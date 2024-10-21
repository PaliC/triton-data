class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 256, 256]", arg1_1: "f32[16, 3, 3, 3]", arg2_1: "f32[16, 1, 1, 1]", arg3_1: "f32[16]", arg4_1: "f32[32, 16, 3, 3]", arg5_1: "f32[32, 1, 1, 1]", arg6_1: "f32[32]", arg7_1: "f32[64, 32, 3, 3]", arg8_1: "f32[64, 1, 1, 1]", arg9_1: "f32[64]", arg10_1: "f32[128, 64, 3, 3]", arg11_1: "f32[128, 1, 1, 1]", arg12_1: "f32[128]", arg13_1: "f32[256, 128, 1, 1]", arg14_1: "f32[256, 1, 1, 1]", arg15_1: "f32[256]", arg16_1: "f32[128, 128, 1, 1]", arg17_1: "f32[128, 1, 1, 1]", arg18_1: "f32[128]", arg19_1: "f32[128, 128, 3, 3]", arg20_1: "f32[128, 1, 1, 1]", arg21_1: "f32[128]", arg22_1: "f32[128, 128, 3, 3]", arg23_1: "f32[128, 1, 1, 1]", arg24_1: "f32[128]", arg25_1: "f32[256, 128, 1, 1]", arg26_1: "f32[256, 1, 1, 1]", arg27_1: "f32[256]", arg28_1: "f32[128, 256, 1, 1]", arg29_1: "f32[128]", arg30_1: "f32[256, 128, 1, 1]", arg31_1: "f32[256]", arg32_1: "f32[]", arg33_1: "f32[512, 256, 1, 1]", arg34_1: "f32[512, 1, 1, 1]", arg35_1: "f32[512]", arg36_1: "f32[256, 256, 1, 1]", arg37_1: "f32[256, 1, 1, 1]", arg38_1: "f32[256]", arg39_1: "f32[256, 128, 3, 3]", arg40_1: "f32[256, 1, 1, 1]", arg41_1: "f32[256]", arg42_1: "f32[256, 128, 3, 3]", arg43_1: "f32[256, 1, 1, 1]", arg44_1: "f32[256]", arg45_1: "f32[512, 256, 1, 1]", arg46_1: "f32[512, 1, 1, 1]", arg47_1: "f32[512]", arg48_1: "f32[256, 512, 1, 1]", arg49_1: "f32[256]", arg50_1: "f32[512, 256, 1, 1]", arg51_1: "f32[512]", arg52_1: "f32[]", arg53_1: "f32[256, 512, 1, 1]", arg54_1: "f32[256, 1, 1, 1]", arg55_1: "f32[256]", arg56_1: "f32[256, 128, 3, 3]", arg57_1: "f32[256, 1, 1, 1]", arg58_1: "f32[256]", arg59_1: "f32[256, 128, 3, 3]", arg60_1: "f32[256, 1, 1, 1]", arg61_1: "f32[256]", arg62_1: "f32[512, 256, 1, 1]", arg63_1: "f32[512, 1, 1, 1]", arg64_1: "f32[512]", arg65_1: "f32[256, 512, 1, 1]", arg66_1: "f32[256]", arg67_1: "f32[512, 256, 1, 1]", arg68_1: "f32[512]", arg69_1: "f32[]", arg70_1: "f32[1536, 512, 1, 1]", arg71_1: "f32[1536, 1, 1, 1]", arg72_1: "f32[1536]", arg73_1: "f32[768, 512, 1, 1]", arg74_1: "f32[768, 1, 1, 1]", arg75_1: "f32[768]", arg76_1: "f32[768, 128, 3, 3]", arg77_1: "f32[768, 1, 1, 1]", arg78_1: "f32[768]", arg79_1: "f32[768, 128, 3, 3]", arg80_1: "f32[768, 1, 1, 1]", arg81_1: "f32[768]", arg82_1: "f32[1536, 768, 1, 1]", arg83_1: "f32[1536, 1, 1, 1]", arg84_1: "f32[1536]", arg85_1: "f32[768, 1536, 1, 1]", arg86_1: "f32[768]", arg87_1: "f32[1536, 768, 1, 1]", arg88_1: "f32[1536]", arg89_1: "f32[]", arg90_1: "f32[768, 1536, 1, 1]", arg91_1: "f32[768, 1, 1, 1]", arg92_1: "f32[768]", arg93_1: "f32[768, 128, 3, 3]", arg94_1: "f32[768, 1, 1, 1]", arg95_1: "f32[768]", arg96_1: "f32[768, 128, 3, 3]", arg97_1: "f32[768, 1, 1, 1]", arg98_1: "f32[768]", arg99_1: "f32[1536, 768, 1, 1]", arg100_1: "f32[1536, 1, 1, 1]", arg101_1: "f32[1536]", arg102_1: "f32[768, 1536, 1, 1]", arg103_1: "f32[768]", arg104_1: "f32[1536, 768, 1, 1]", arg105_1: "f32[1536]", arg106_1: "f32[]", arg107_1: "f32[768, 1536, 1, 1]", arg108_1: "f32[768, 1, 1, 1]", arg109_1: "f32[768]", arg110_1: "f32[768, 128, 3, 3]", arg111_1: "f32[768, 1, 1, 1]", arg112_1: "f32[768]", arg113_1: "f32[768, 128, 3, 3]", arg114_1: "f32[768, 1, 1, 1]", arg115_1: "f32[768]", arg116_1: "f32[1536, 768, 1, 1]", arg117_1: "f32[1536, 1, 1, 1]", arg118_1: "f32[1536]", arg119_1: "f32[768, 1536, 1, 1]", arg120_1: "f32[768]", arg121_1: "f32[1536, 768, 1, 1]", arg122_1: "f32[1536]", arg123_1: "f32[]", arg124_1: "f32[768, 1536, 1, 1]", arg125_1: "f32[768, 1, 1, 1]", arg126_1: "f32[768]", arg127_1: "f32[768, 128, 3, 3]", arg128_1: "f32[768, 1, 1, 1]", arg129_1: "f32[768]", arg130_1: "f32[768, 128, 3, 3]", arg131_1: "f32[768, 1, 1, 1]", arg132_1: "f32[768]", arg133_1: "f32[1536, 768, 1, 1]", arg134_1: "f32[1536, 1, 1, 1]", arg135_1: "f32[1536]", arg136_1: "f32[768, 1536, 1, 1]", arg137_1: "f32[768]", arg138_1: "f32[1536, 768, 1, 1]", arg139_1: "f32[1536]", arg140_1: "f32[]", arg141_1: "f32[768, 1536, 1, 1]", arg142_1: "f32[768, 1, 1, 1]", arg143_1: "f32[768]", arg144_1: "f32[768, 128, 3, 3]", arg145_1: "f32[768, 1, 1, 1]", arg146_1: "f32[768]", arg147_1: "f32[768, 128, 3, 3]", arg148_1: "f32[768, 1, 1, 1]", arg149_1: "f32[768]", arg150_1: "f32[1536, 768, 1, 1]", arg151_1: "f32[1536, 1, 1, 1]", arg152_1: "f32[1536]", arg153_1: "f32[768, 1536, 1, 1]", arg154_1: "f32[768]", arg155_1: "f32[1536, 768, 1, 1]", arg156_1: "f32[1536]", arg157_1: "f32[]", arg158_1: "f32[768, 1536, 1, 1]", arg159_1: "f32[768, 1, 1, 1]", arg160_1: "f32[768]", arg161_1: "f32[768, 128, 3, 3]", arg162_1: "f32[768, 1, 1, 1]", arg163_1: "f32[768]", arg164_1: "f32[768, 128, 3, 3]", arg165_1: "f32[768, 1, 1, 1]", arg166_1: "f32[768]", arg167_1: "f32[1536, 768, 1, 1]", arg168_1: "f32[1536, 1, 1, 1]", arg169_1: "f32[1536]", arg170_1: "f32[768, 1536, 1, 1]", arg171_1: "f32[768]", arg172_1: "f32[1536, 768, 1, 1]", arg173_1: "f32[1536]", arg174_1: "f32[]", arg175_1: "f32[1536, 1536, 1, 1]", arg176_1: "f32[1536, 1, 1, 1]", arg177_1: "f32[1536]", arg178_1: "f32[768, 1536, 1, 1]", arg179_1: "f32[768, 1, 1, 1]", arg180_1: "f32[768]", arg181_1: "f32[768, 128, 3, 3]", arg182_1: "f32[768, 1, 1, 1]", arg183_1: "f32[768]", arg184_1: "f32[768, 128, 3, 3]", arg185_1: "f32[768, 1, 1, 1]", arg186_1: "f32[768]", arg187_1: "f32[1536, 768, 1, 1]", arg188_1: "f32[1536, 1, 1, 1]", arg189_1: "f32[1536]", arg190_1: "f32[768, 1536, 1, 1]", arg191_1: "f32[768]", arg192_1: "f32[1536, 768, 1, 1]", arg193_1: "f32[1536]", arg194_1: "f32[]", arg195_1: "f32[768, 1536, 1, 1]", arg196_1: "f32[768, 1, 1, 1]", arg197_1: "f32[768]", arg198_1: "f32[768, 128, 3, 3]", arg199_1: "f32[768, 1, 1, 1]", arg200_1: "f32[768]", arg201_1: "f32[768, 128, 3, 3]", arg202_1: "f32[768, 1, 1, 1]", arg203_1: "f32[768]", arg204_1: "f32[1536, 768, 1, 1]", arg205_1: "f32[1536, 1, 1, 1]", arg206_1: "f32[1536]", arg207_1: "f32[768, 1536, 1, 1]", arg208_1: "f32[768]", arg209_1: "f32[1536, 768, 1, 1]", arg210_1: "f32[1536]", arg211_1: "f32[]", arg212_1: "f32[768, 1536, 1, 1]", arg213_1: "f32[768, 1, 1, 1]", arg214_1: "f32[768]", arg215_1: "f32[768, 128, 3, 3]", arg216_1: "f32[768, 1, 1, 1]", arg217_1: "f32[768]", arg218_1: "f32[768, 128, 3, 3]", arg219_1: "f32[768, 1, 1, 1]", arg220_1: "f32[768]", arg221_1: "f32[1536, 768, 1, 1]", arg222_1: "f32[1536, 1, 1, 1]", arg223_1: "f32[1536]", arg224_1: "f32[768, 1536, 1, 1]", arg225_1: "f32[768]", arg226_1: "f32[1536, 768, 1, 1]", arg227_1: "f32[1536]", arg228_1: "f32[]", arg229_1: "f32[3072, 1536, 1, 1]", arg230_1: "f32[3072, 1, 1, 1]", arg231_1: "f32[3072]", arg232_1: "f32[1000, 3072]", arg233_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_172: "f32[1, 16, 27]" = torch.ops.aten.reshape.default(arg1_1, [1, 16, -1]);  arg1_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_57 = torch.ops.aten.var_mean.correction(view_172, [0, 2], correction = 0, keepdim = True)
        getitem_114: "f32[1, 16, 1]" = var_mean_57[0]
        getitem_115: "f32[1, 16, 1]" = var_mean_57[1];  var_mean_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_175: "f32[1, 32, 144]" = torch.ops.aten.reshape.default(arg4_1, [1, 32, -1]);  arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_58 = torch.ops.aten.var_mean.correction(view_175, [0, 2], correction = 0, keepdim = True)
        getitem_116: "f32[1, 32, 1]" = var_mean_58[0]
        getitem_117: "f32[1, 32, 1]" = var_mean_58[1];  var_mean_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_178: "f32[1, 64, 288]" = torch.ops.aten.reshape.default(arg7_1, [1, 64, -1]);  arg7_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_59 = torch.ops.aten.var_mean.correction(view_178, [0, 2], correction = 0, keepdim = True)
        getitem_118: "f32[1, 64, 1]" = var_mean_59[0]
        getitem_119: "f32[1, 64, 1]" = var_mean_59[1];  var_mean_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_181: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(arg10_1, [1, 128, -1]);  arg10_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_60 = torch.ops.aten.var_mean.correction(view_181, [0, 2], correction = 0, keepdim = True)
        getitem_120: "f32[1, 128, 1]" = var_mean_60[0]
        getitem_121: "f32[1, 128, 1]" = var_mean_60[1];  var_mean_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_184: "f32[1, 256, 128]" = torch.ops.aten.reshape.default(arg13_1, [1, 256, -1]);  arg13_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_61 = torch.ops.aten.var_mean.correction(view_184, [0, 2], correction = 0, keepdim = True)
        getitem_122: "f32[1, 256, 1]" = var_mean_61[0]
        getitem_123: "f32[1, 256, 1]" = var_mean_61[1];  var_mean_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_187: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(arg16_1, [1, 128, -1]);  arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_62 = torch.ops.aten.var_mean.correction(view_187, [0, 2], correction = 0, keepdim = True)
        getitem_124: "f32[1, 128, 1]" = var_mean_62[0]
        getitem_125: "f32[1, 128, 1]" = var_mean_62[1];  var_mean_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_190: "f32[1, 128, 1152]" = torch.ops.aten.reshape.default(arg19_1, [1, 128, -1]);  arg19_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_63 = torch.ops.aten.var_mean.correction(view_190, [0, 2], correction = 0, keepdim = True)
        getitem_126: "f32[1, 128, 1]" = var_mean_63[0]
        getitem_127: "f32[1, 128, 1]" = var_mean_63[1];  var_mean_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_193: "f32[1, 128, 1152]" = torch.ops.aten.reshape.default(arg22_1, [1, 128, -1]);  arg22_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_64 = torch.ops.aten.var_mean.correction(view_193, [0, 2], correction = 0, keepdim = True)
        getitem_128: "f32[1, 128, 1]" = var_mean_64[0]
        getitem_129: "f32[1, 128, 1]" = var_mean_64[1];  var_mean_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_196: "f32[1, 256, 128]" = torch.ops.aten.reshape.default(arg25_1, [1, 256, -1]);  arg25_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_65 = torch.ops.aten.var_mean.correction(view_196, [0, 2], correction = 0, keepdim = True)
        getitem_130: "f32[1, 256, 1]" = var_mean_65[0]
        getitem_131: "f32[1, 256, 1]" = var_mean_65[1];  var_mean_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_199: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(arg33_1, [1, 512, -1]);  arg33_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_66 = torch.ops.aten.var_mean.correction(view_199, [0, 2], correction = 0, keepdim = True)
        getitem_132: "f32[1, 512, 1]" = var_mean_66[0]
        getitem_133: "f32[1, 512, 1]" = var_mean_66[1];  var_mean_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_202: "f32[1, 256, 256]" = torch.ops.aten.reshape.default(arg36_1, [1, 256, -1]);  arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_67 = torch.ops.aten.var_mean.correction(view_202, [0, 2], correction = 0, keepdim = True)
        getitem_134: "f32[1, 256, 1]" = var_mean_67[0]
        getitem_135: "f32[1, 256, 1]" = var_mean_67[1];  var_mean_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_205: "f32[1, 256, 1152]" = torch.ops.aten.reshape.default(arg39_1, [1, 256, -1]);  arg39_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_68 = torch.ops.aten.var_mean.correction(view_205, [0, 2], correction = 0, keepdim = True)
        getitem_136: "f32[1, 256, 1]" = var_mean_68[0]
        getitem_137: "f32[1, 256, 1]" = var_mean_68[1];  var_mean_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_208: "f32[1, 256, 1152]" = torch.ops.aten.reshape.default(arg42_1, [1, 256, -1]);  arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_69 = torch.ops.aten.var_mean.correction(view_208, [0, 2], correction = 0, keepdim = True)
        getitem_138: "f32[1, 256, 1]" = var_mean_69[0]
        getitem_139: "f32[1, 256, 1]" = var_mean_69[1];  var_mean_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_211: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(arg45_1, [1, 512, -1]);  arg45_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_70 = torch.ops.aten.var_mean.correction(view_211, [0, 2], correction = 0, keepdim = True)
        getitem_140: "f32[1, 512, 1]" = var_mean_70[0]
        getitem_141: "f32[1, 512, 1]" = var_mean_70[1];  var_mean_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_214: "f32[1, 256, 512]" = torch.ops.aten.reshape.default(arg53_1, [1, 256, -1]);  arg53_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_71 = torch.ops.aten.var_mean.correction(view_214, [0, 2], correction = 0, keepdim = True)
        getitem_142: "f32[1, 256, 1]" = var_mean_71[0]
        getitem_143: "f32[1, 256, 1]" = var_mean_71[1];  var_mean_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_217: "f32[1, 256, 1152]" = torch.ops.aten.reshape.default(arg56_1, [1, 256, -1]);  arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_72 = torch.ops.aten.var_mean.correction(view_217, [0, 2], correction = 0, keepdim = True)
        getitem_144: "f32[1, 256, 1]" = var_mean_72[0]
        getitem_145: "f32[1, 256, 1]" = var_mean_72[1];  var_mean_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_220: "f32[1, 256, 1152]" = torch.ops.aten.reshape.default(arg59_1, [1, 256, -1]);  arg59_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_73 = torch.ops.aten.var_mean.correction(view_220, [0, 2], correction = 0, keepdim = True)
        getitem_146: "f32[1, 256, 1]" = var_mean_73[0]
        getitem_147: "f32[1, 256, 1]" = var_mean_73[1];  var_mean_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_223: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(arg62_1, [1, 512, -1]);  arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_74 = torch.ops.aten.var_mean.correction(view_223, [0, 2], correction = 0, keepdim = True)
        getitem_148: "f32[1, 512, 1]" = var_mean_74[0]
        getitem_149: "f32[1, 512, 1]" = var_mean_74[1];  var_mean_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_226: "f32[1, 1536, 512]" = torch.ops.aten.reshape.default(arg70_1, [1, 1536, -1]);  arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_75 = torch.ops.aten.var_mean.correction(view_226, [0, 2], correction = 0, keepdim = True)
        getitem_150: "f32[1, 1536, 1]" = var_mean_75[0]
        getitem_151: "f32[1, 1536, 1]" = var_mean_75[1];  var_mean_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_229: "f32[1, 768, 512]" = torch.ops.aten.reshape.default(arg73_1, [1, 768, -1]);  arg73_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_76 = torch.ops.aten.var_mean.correction(view_229, [0, 2], correction = 0, keepdim = True)
        getitem_152: "f32[1, 768, 1]" = var_mean_76[0]
        getitem_153: "f32[1, 768, 1]" = var_mean_76[1];  var_mean_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_232: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg76_1, [1, 768, -1]);  arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_77 = torch.ops.aten.var_mean.correction(view_232, [0, 2], correction = 0, keepdim = True)
        getitem_154: "f32[1, 768, 1]" = var_mean_77[0]
        getitem_155: "f32[1, 768, 1]" = var_mean_77[1];  var_mean_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_235: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg79_1, [1, 768, -1]);  arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_78 = torch.ops.aten.var_mean.correction(view_235, [0, 2], correction = 0, keepdim = True)
        getitem_156: "f32[1, 768, 1]" = var_mean_78[0]
        getitem_157: "f32[1, 768, 1]" = var_mean_78[1];  var_mean_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_238: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(arg82_1, [1, 1536, -1]);  arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_79 = torch.ops.aten.var_mean.correction(view_238, [0, 2], correction = 0, keepdim = True)
        getitem_158: "f32[1, 1536, 1]" = var_mean_79[0]
        getitem_159: "f32[1, 1536, 1]" = var_mean_79[1];  var_mean_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_241: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(arg90_1, [1, 768, -1]);  arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_80 = torch.ops.aten.var_mean.correction(view_241, [0, 2], correction = 0, keepdim = True)
        getitem_160: "f32[1, 768, 1]" = var_mean_80[0]
        getitem_161: "f32[1, 768, 1]" = var_mean_80[1];  var_mean_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_244: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg93_1, [1, 768, -1]);  arg93_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_81 = torch.ops.aten.var_mean.correction(view_244, [0, 2], correction = 0, keepdim = True)
        getitem_162: "f32[1, 768, 1]" = var_mean_81[0]
        getitem_163: "f32[1, 768, 1]" = var_mean_81[1];  var_mean_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_247: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg96_1, [1, 768, -1]);  arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_82 = torch.ops.aten.var_mean.correction(view_247, [0, 2], correction = 0, keepdim = True)
        getitem_164: "f32[1, 768, 1]" = var_mean_82[0]
        getitem_165: "f32[1, 768, 1]" = var_mean_82[1];  var_mean_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_250: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(arg99_1, [1, 1536, -1]);  arg99_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_83 = torch.ops.aten.var_mean.correction(view_250, [0, 2], correction = 0, keepdim = True)
        getitem_166: "f32[1, 1536, 1]" = var_mean_83[0]
        getitem_167: "f32[1, 1536, 1]" = var_mean_83[1];  var_mean_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_253: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(arg107_1, [1, 768, -1]);  arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_84 = torch.ops.aten.var_mean.correction(view_253, [0, 2], correction = 0, keepdim = True)
        getitem_168: "f32[1, 768, 1]" = var_mean_84[0]
        getitem_169: "f32[1, 768, 1]" = var_mean_84[1];  var_mean_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_256: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg110_1, [1, 768, -1]);  arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_85 = torch.ops.aten.var_mean.correction(view_256, [0, 2], correction = 0, keepdim = True)
        getitem_170: "f32[1, 768, 1]" = var_mean_85[0]
        getitem_171: "f32[1, 768, 1]" = var_mean_85[1];  var_mean_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_259: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg113_1, [1, 768, -1]);  arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_86 = torch.ops.aten.var_mean.correction(view_259, [0, 2], correction = 0, keepdim = True)
        getitem_172: "f32[1, 768, 1]" = var_mean_86[0]
        getitem_173: "f32[1, 768, 1]" = var_mean_86[1];  var_mean_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_262: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(arg116_1, [1, 1536, -1]);  arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_87 = torch.ops.aten.var_mean.correction(view_262, [0, 2], correction = 0, keepdim = True)
        getitem_174: "f32[1, 1536, 1]" = var_mean_87[0]
        getitem_175: "f32[1, 1536, 1]" = var_mean_87[1];  var_mean_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_265: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(arg124_1, [1, 768, -1]);  arg124_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_88 = torch.ops.aten.var_mean.correction(view_265, [0, 2], correction = 0, keepdim = True)
        getitem_176: "f32[1, 768, 1]" = var_mean_88[0]
        getitem_177: "f32[1, 768, 1]" = var_mean_88[1];  var_mean_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_268: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg127_1, [1, 768, -1]);  arg127_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_89 = torch.ops.aten.var_mean.correction(view_268, [0, 2], correction = 0, keepdim = True)
        getitem_178: "f32[1, 768, 1]" = var_mean_89[0]
        getitem_179: "f32[1, 768, 1]" = var_mean_89[1];  var_mean_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_271: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg130_1, [1, 768, -1]);  arg130_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_90 = torch.ops.aten.var_mean.correction(view_271, [0, 2], correction = 0, keepdim = True)
        getitem_180: "f32[1, 768, 1]" = var_mean_90[0]
        getitem_181: "f32[1, 768, 1]" = var_mean_90[1];  var_mean_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_274: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(arg133_1, [1, 1536, -1]);  arg133_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_91 = torch.ops.aten.var_mean.correction(view_274, [0, 2], correction = 0, keepdim = True)
        getitem_182: "f32[1, 1536, 1]" = var_mean_91[0]
        getitem_183: "f32[1, 1536, 1]" = var_mean_91[1];  var_mean_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_277: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(arg141_1, [1, 768, -1]);  arg141_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_92 = torch.ops.aten.var_mean.correction(view_277, [0, 2], correction = 0, keepdim = True)
        getitem_184: "f32[1, 768, 1]" = var_mean_92[0]
        getitem_185: "f32[1, 768, 1]" = var_mean_92[1];  var_mean_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_280: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg144_1, [1, 768, -1]);  arg144_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_93 = torch.ops.aten.var_mean.correction(view_280, [0, 2], correction = 0, keepdim = True)
        getitem_186: "f32[1, 768, 1]" = var_mean_93[0]
        getitem_187: "f32[1, 768, 1]" = var_mean_93[1];  var_mean_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_283: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg147_1, [1, 768, -1]);  arg147_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_94 = torch.ops.aten.var_mean.correction(view_283, [0, 2], correction = 0, keepdim = True)
        getitem_188: "f32[1, 768, 1]" = var_mean_94[0]
        getitem_189: "f32[1, 768, 1]" = var_mean_94[1];  var_mean_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_286: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(arg150_1, [1, 1536, -1]);  arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_95 = torch.ops.aten.var_mean.correction(view_286, [0, 2], correction = 0, keepdim = True)
        getitem_190: "f32[1, 1536, 1]" = var_mean_95[0]
        getitem_191: "f32[1, 1536, 1]" = var_mean_95[1];  var_mean_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_289: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(arg158_1, [1, 768, -1]);  arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_96 = torch.ops.aten.var_mean.correction(view_289, [0, 2], correction = 0, keepdim = True)
        getitem_192: "f32[1, 768, 1]" = var_mean_96[0]
        getitem_193: "f32[1, 768, 1]" = var_mean_96[1];  var_mean_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_292: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg161_1, [1, 768, -1]);  arg161_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_97 = torch.ops.aten.var_mean.correction(view_292, [0, 2], correction = 0, keepdim = True)
        getitem_194: "f32[1, 768, 1]" = var_mean_97[0]
        getitem_195: "f32[1, 768, 1]" = var_mean_97[1];  var_mean_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_295: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg164_1, [1, 768, -1]);  arg164_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_98 = torch.ops.aten.var_mean.correction(view_295, [0, 2], correction = 0, keepdim = True)
        getitem_196: "f32[1, 768, 1]" = var_mean_98[0]
        getitem_197: "f32[1, 768, 1]" = var_mean_98[1];  var_mean_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_298: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(arg167_1, [1, 1536, -1]);  arg167_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_99 = torch.ops.aten.var_mean.correction(view_298, [0, 2], correction = 0, keepdim = True)
        getitem_198: "f32[1, 1536, 1]" = var_mean_99[0]
        getitem_199: "f32[1, 1536, 1]" = var_mean_99[1];  var_mean_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_301: "f32[1, 1536, 1536]" = torch.ops.aten.reshape.default(arg175_1, [1, 1536, -1]);  arg175_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_100 = torch.ops.aten.var_mean.correction(view_301, [0, 2], correction = 0, keepdim = True)
        getitem_200: "f32[1, 1536, 1]" = var_mean_100[0]
        getitem_201: "f32[1, 1536, 1]" = var_mean_100[1];  var_mean_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_304: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(arg178_1, [1, 768, -1]);  arg178_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_101 = torch.ops.aten.var_mean.correction(view_304, [0, 2], correction = 0, keepdim = True)
        getitem_202: "f32[1, 768, 1]" = var_mean_101[0]
        getitem_203: "f32[1, 768, 1]" = var_mean_101[1];  var_mean_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_307: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg181_1, [1, 768, -1]);  arg181_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_102 = torch.ops.aten.var_mean.correction(view_307, [0, 2], correction = 0, keepdim = True)
        getitem_204: "f32[1, 768, 1]" = var_mean_102[0]
        getitem_205: "f32[1, 768, 1]" = var_mean_102[1];  var_mean_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_310: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg184_1, [1, 768, -1]);  arg184_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_103 = torch.ops.aten.var_mean.correction(view_310, [0, 2], correction = 0, keepdim = True)
        getitem_206: "f32[1, 768, 1]" = var_mean_103[0]
        getitem_207: "f32[1, 768, 1]" = var_mean_103[1];  var_mean_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_313: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(arg187_1, [1, 1536, -1]);  arg187_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_104 = torch.ops.aten.var_mean.correction(view_313, [0, 2], correction = 0, keepdim = True)
        getitem_208: "f32[1, 1536, 1]" = var_mean_104[0]
        getitem_209: "f32[1, 1536, 1]" = var_mean_104[1];  var_mean_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_316: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(arg195_1, [1, 768, -1]);  arg195_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_105 = torch.ops.aten.var_mean.correction(view_316, [0, 2], correction = 0, keepdim = True)
        getitem_210: "f32[1, 768, 1]" = var_mean_105[0]
        getitem_211: "f32[1, 768, 1]" = var_mean_105[1];  var_mean_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_319: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg198_1, [1, 768, -1]);  arg198_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_106 = torch.ops.aten.var_mean.correction(view_319, [0, 2], correction = 0, keepdim = True)
        getitem_212: "f32[1, 768, 1]" = var_mean_106[0]
        getitem_213: "f32[1, 768, 1]" = var_mean_106[1];  var_mean_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_322: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg201_1, [1, 768, -1]);  arg201_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_107 = torch.ops.aten.var_mean.correction(view_322, [0, 2], correction = 0, keepdim = True)
        getitem_214: "f32[1, 768, 1]" = var_mean_107[0]
        getitem_215: "f32[1, 768, 1]" = var_mean_107[1];  var_mean_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_325: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(arg204_1, [1, 1536, -1]);  arg204_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_108 = torch.ops.aten.var_mean.correction(view_325, [0, 2], correction = 0, keepdim = True)
        getitem_216: "f32[1, 1536, 1]" = var_mean_108[0]
        getitem_217: "f32[1, 1536, 1]" = var_mean_108[1];  var_mean_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_328: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(arg212_1, [1, 768, -1]);  arg212_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_109 = torch.ops.aten.var_mean.correction(view_328, [0, 2], correction = 0, keepdim = True)
        getitem_218: "f32[1, 768, 1]" = var_mean_109[0]
        getitem_219: "f32[1, 768, 1]" = var_mean_109[1];  var_mean_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_331: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg215_1, [1, 768, -1]);  arg215_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_110 = torch.ops.aten.var_mean.correction(view_331, [0, 2], correction = 0, keepdim = True)
        getitem_220: "f32[1, 768, 1]" = var_mean_110[0]
        getitem_221: "f32[1, 768, 1]" = var_mean_110[1];  var_mean_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_334: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(arg218_1, [1, 768, -1]);  arg218_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_111 = torch.ops.aten.var_mean.correction(view_334, [0, 2], correction = 0, keepdim = True)
        getitem_222: "f32[1, 768, 1]" = var_mean_111[0]
        getitem_223: "f32[1, 768, 1]" = var_mean_111[1];  var_mean_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_337: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(arg221_1, [1, 1536, -1]);  arg221_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_112 = torch.ops.aten.var_mean.correction(view_337, [0, 2], correction = 0, keepdim = True)
        getitem_224: "f32[1, 1536, 1]" = var_mean_112[0]
        getitem_225: "f32[1, 1536, 1]" = var_mean_112[1];  var_mean_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:130 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_340: "f32[1, 3072, 1536]" = torch.ops.aten.reshape.default(arg229_1, [1, 3072, -1]);  arg229_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        var_mean_113 = torch.ops.aten.var_mean.correction(view_340, [0, 2], correction = 0, keepdim = True)
        getitem_226: "f32[1, 3072, 1]" = var_mean_113[0]
        getitem_227: "f32[1, 3072, 1]" = var_mean_113[1];  var_mean_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_5: "f32[8, 3, 257, 257]" = torch.ops.aten.constant_pad_nd.default(arg0_1, [0, 1, 0, 1], 0.0);  arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_57: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view_172, getitem_115);  view_172 = getitem_115 = None
        add_121: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
        rsqrt_57: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        mul_440: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = rsqrt_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_439: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg2_1, 0.19245008972987526);  arg2_1 = None
        view_173: "f32[16]" = torch.ops.aten.reshape.default(mul_439, [-1]);  mul_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_57: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(view_173, -1);  view_173 = None
        mul_441: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(mul_440, unsqueeze_57);  mul_440 = unsqueeze_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_174: "f32[16, 3, 3, 3]" = torch.ops.aten.reshape.default(mul_441, [16, 3, 3, 3]);  mul_441 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_81: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(constant_pad_nd_5, view_174, arg3_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  constant_pad_nd_5 = view_174 = arg3_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_442: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_81, 0.5)
        mul_443: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_81, 0.7071067811865476);  convolution_81 = None
        erf_52: "f32[8, 16, 128, 128]" = torch.ops.aten.erf.default(mul_443);  mul_443 = None
        add_122: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
        mul_444: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_442, add_122);  mul_442 = add_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_445: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_444, 1.7015043497085571);  mul_444 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_58: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_175, getitem_117);  view_175 = getitem_117 = None
        add_123: "f32[1, 32, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_58: "f32[1, 32, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        mul_447: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = rsqrt_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_446: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg5_1, 0.08333333333333333);  arg5_1 = None
        view_176: "f32[32]" = torch.ops.aten.reshape.default(mul_446, [-1]);  mul_446 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_58: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(view_176, -1);  view_176 = None
        mul_448: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(mul_447, unsqueeze_58);  mul_447 = unsqueeze_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_177: "f32[32, 16, 3, 3]" = torch.ops.aten.reshape.default(mul_448, [32, 16, 3, 3]);  mul_448 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_82: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(mul_445, view_177, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_445 = view_177 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_449: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_82, 0.5)
        mul_450: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_82, 0.7071067811865476);  convolution_82 = None
        erf_53: "f32[8, 32, 128, 128]" = torch.ops.aten.erf.default(mul_450);  mul_450 = None
        add_124: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
        mul_451: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_449, add_124);  mul_449 = add_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_452: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_451, 1.7015043497085571);  mul_451 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_59: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_178, getitem_119);  view_178 = getitem_119 = None
        add_125: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
        rsqrt_59: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        mul_454: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = rsqrt_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_453: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg8_1, 0.05892556509887896);  arg8_1 = None
        view_179: "f32[64]" = torch.ops.aten.reshape.default(mul_453, [-1]);  mul_453 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_59: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_179, -1);  view_179 = None
        mul_455: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(mul_454, unsqueeze_59);  mul_454 = unsqueeze_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_180: "f32[64, 32, 3, 3]" = torch.ops.aten.reshape.default(mul_455, [64, 32, 3, 3]);  mul_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_83: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(mul_452, view_180, arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_452 = view_180 = arg9_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_456: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_83, 0.5)
        mul_457: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_83, 0.7071067811865476);  convolution_83 = None
        erf_54: "f32[8, 64, 128, 128]" = torch.ops.aten.erf.default(mul_457);  mul_457 = None
        add_126: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
        mul_458: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_456, add_126);  mul_456 = add_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_459: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_458, 1.7015043497085571);  mul_458 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_6: "f32[8, 64, 129, 129]" = torch.ops.aten.constant_pad_nd.default(mul_459, [0, 1, 0, 1], 0.0);  mul_459 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_60: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_181, getitem_121);  view_181 = getitem_121 = None
        add_127: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
        rsqrt_60: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        mul_461: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = rsqrt_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_460: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg11_1, 0.041666666666666664);  arg11_1 = None
        view_182: "f32[128]" = torch.ops.aten.reshape.default(mul_460, [-1]);  mul_460 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_182, -1);  view_182 = None
        mul_462: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_461, unsqueeze_60);  mul_461 = unsqueeze_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_183: "f32[128, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_462, [128, 64, 3, 3]);  mul_462 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_84: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(constant_pad_nd_6, view_183, arg12_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  constant_pad_nd_6 = view_183 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_463: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_84, 0.5)
        mul_464: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_84, 0.7071067811865476);  convolution_84 = None
        erf_55: "f32[8, 128, 64, 64]" = torch.ops.aten.erf.default(mul_464);  mul_464 = None
        add_128: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
        mul_465: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_463, add_128);  mul_463 = add_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_466: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_465, 1.7015043497085571);  mul_465 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        mul_467: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_466, 1.0);  mul_466 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_62: "f32[1, 128, 128]" = torch.ops.aten.sub.Tensor(view_187, getitem_125);  view_187 = getitem_125 = None
        add_130: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
        rsqrt_62: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        mul_472: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = rsqrt_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_471: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg17_1, 0.08838834764831845);  arg17_1 = None
        view_188: "f32[128]" = torch.ops.aten.reshape.default(mul_471, [-1]);  mul_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_188, -1);  view_188 = None
        mul_473: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(mul_472, unsqueeze_62);  mul_472 = unsqueeze_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_189: "f32[128, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_473, [128, 128, 1, 1]);  mul_473 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_86: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(mul_467, view_189, arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_189 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_474: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_86, 0.5)
        mul_475: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_86, 0.7071067811865476);  convolution_86 = None
        erf_56: "f32[8, 128, 64, 64]" = torch.ops.aten.erf.default(mul_475);  mul_475 = None
        add_131: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
        mul_476: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_474, add_131);  mul_474 = add_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_477: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_476, 1.7015043497085571);  mul_476 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_63: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_190, getitem_127);  view_190 = getitem_127 = None
        add_132: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
        rsqrt_63: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        mul_479: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = rsqrt_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_478: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg20_1, 0.02946278254943948);  arg20_1 = None
        view_191: "f32[128]" = torch.ops.aten.reshape.default(mul_478, [-1]);  mul_478 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_63: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_191, -1);  view_191 = None
        mul_480: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(mul_479, unsqueeze_63);  mul_479 = unsqueeze_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_192: "f32[128, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_480, [128, 128, 3, 3]);  mul_480 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_87: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(mul_477, view_192, arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_477 = view_192 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_481: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_87, 0.5)
        mul_482: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_87, 0.7071067811865476);  convolution_87 = None
        erf_57: "f32[8, 128, 64, 64]" = torch.ops.aten.erf.default(mul_482);  mul_482 = None
        add_133: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
        mul_483: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_481, add_133);  mul_481 = add_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_484: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_483, 1.7015043497085571);  mul_483 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_64: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_193, getitem_129);  view_193 = getitem_129 = None
        add_134: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_64: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        mul_486: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = rsqrt_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_485: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg23_1, 0.02946278254943948);  arg23_1 = None
        view_194: "f32[128]" = torch.ops.aten.reshape.default(mul_485, [-1]);  mul_485 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_64: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_194, -1);  view_194 = None
        mul_487: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(mul_486, unsqueeze_64);  mul_486 = unsqueeze_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_195: "f32[128, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_487, [128, 128, 3, 3]);  mul_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_88: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(mul_484, view_195, arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_484 = view_195 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_488: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_88, 0.5)
        mul_489: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_88, 0.7071067811865476);  convolution_88 = None
        erf_58: "f32[8, 128, 64, 64]" = torch.ops.aten.erf.default(mul_489);  mul_489 = None
        add_135: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
        mul_490: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_488, add_135);  mul_488 = add_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_491: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_490, 1.7015043497085571);  mul_490 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_65: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_196, getitem_131);  view_196 = getitem_131 = None
        add_136: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_65: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_493: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = rsqrt_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_492: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg26_1, 0.08838834764831845);  arg26_1 = None
        view_197: "f32[256]" = torch.ops.aten.reshape.default(mul_492, [-1]);  mul_492 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_65: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_197, -1);  view_197 = None
        mul_494: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(mul_493, unsqueeze_65);  mul_493 = unsqueeze_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_198: "f32[256, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_494, [256, 128, 1, 1]);  mul_494 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_89: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_491, view_198, arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_491 = view_198 = arg27_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_13: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(convolution_89, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_90: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_13, arg28_1, arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_13 = arg28_1 = arg29_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_12: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(convolution_90);  convolution_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_91: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_12, arg30_1, arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_12 = arg30_1 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_12: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_91);  convolution_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_495: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_89, sigmoid_12);  convolution_89 = sigmoid_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_496: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_495, 2.0);  mul_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:198 in forward, code: out.mul_(self.skipinit_gain)
        mul_497: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_496, arg32_1);  mul_496 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_498: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_497, 0.2);  mul_497 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_61: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_184, getitem_123);  view_184 = getitem_123 = None
        add_129: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
        rsqrt_61: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_469: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = rsqrt_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_468: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg14_1, 0.08838834764831845);  arg14_1 = None
        view_185: "f32[256]" = torch.ops.aten.reshape.default(mul_468, [-1]);  mul_468 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_61: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_185, -1);  view_185 = None
        mul_470: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_61);  mul_469 = unsqueeze_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_186: "f32[256, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_470, [256, 128, 1, 1]);  mul_470 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_85: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_467, view_186, arg15_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_467 = view_186 = arg15_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        add_137: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_498, convolution_85);  mul_498 = convolution_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_499: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_137, 0.5)
        mul_500: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_137, 0.7071067811865476);  add_137 = None
        erf_59: "f32[8, 256, 64, 64]" = torch.ops.aten.erf.default(mul_500);  mul_500 = None
        add_138: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
        mul_501: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_499, add_138);  mul_499 = add_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_502: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_501, 1.7015043497085571);  mul_501 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        mul_503: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_502, 0.9805806756909201);  mul_502 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_67: "f32[1, 256, 256]" = torch.ops.aten.sub.Tensor(view_202, getitem_135);  view_202 = getitem_135 = None
        add_140: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05);  getitem_134 = None
        rsqrt_67: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_508: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = rsqrt_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_507: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg37_1, 0.0625);  arg37_1 = None
        view_203: "f32[256]" = torch.ops.aten.reshape.default(mul_507, [-1]);  mul_507 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_67: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_203, -1);  view_203 = None
        mul_509: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(mul_508, unsqueeze_67);  mul_508 = unsqueeze_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_204: "f32[256, 256, 1, 1]" = torch.ops.aten.reshape.default(mul_509, [256, 256, 1, 1]);  mul_509 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_93: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_503, view_204, arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_204 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_510: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_93, 0.5)
        mul_511: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_93, 0.7071067811865476);  convolution_93 = None
        erf_60: "f32[8, 256, 64, 64]" = torch.ops.aten.erf.default(mul_511);  mul_511 = None
        add_141: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(erf_60, 1);  erf_60 = None
        mul_512: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_510, add_141);  mul_510 = add_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_513: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_512, 1.7015043497085571);  mul_512 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_7: "f32[8, 256, 65, 65]" = torch.ops.aten.constant_pad_nd.default(mul_513, [0, 1, 0, 1], 0.0);  mul_513 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_68: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_205, getitem_137);  view_205 = getitem_137 = None
        add_142: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05);  getitem_136 = None
        rsqrt_68: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        mul_515: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = rsqrt_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_514: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg40_1, 0.02946278254943948);  arg40_1 = None
        view_206: "f32[256]" = torch.ops.aten.reshape.default(mul_514, [-1]);  mul_514 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_68: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_206, -1);  view_206 = None
        mul_516: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(mul_515, unsqueeze_68);  mul_515 = unsqueeze_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_207: "f32[256, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_516, [256, 128, 3, 3]);  mul_516 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_94: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(constant_pad_nd_7, view_207, arg41_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 2);  constant_pad_nd_7 = view_207 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_517: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_94, 0.5)
        mul_518: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_94, 0.7071067811865476);  convolution_94 = None
        erf_61: "f32[8, 256, 32, 32]" = torch.ops.aten.erf.default(mul_518);  mul_518 = None
        add_143: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(erf_61, 1);  erf_61 = None
        mul_519: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_517, add_143);  mul_517 = add_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_520: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_519, 1.7015043497085571);  mul_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_69: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_208, getitem_139);  view_208 = getitem_139 = None
        add_144: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
        rsqrt_69: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        mul_522: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = rsqrt_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_521: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg43_1, 0.02946278254943948);  arg43_1 = None
        view_209: "f32[256]" = torch.ops.aten.reshape.default(mul_521, [-1]);  mul_521 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_69: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_209, -1);  view_209 = None
        mul_523: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(mul_522, unsqueeze_69);  mul_522 = unsqueeze_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_210: "f32[256, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_523, [256, 128, 3, 3]);  mul_523 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_95: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_520, view_210, arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_520 = view_210 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_524: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_95, 0.5)
        mul_525: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_95, 0.7071067811865476);  convolution_95 = None
        erf_62: "f32[8, 256, 32, 32]" = torch.ops.aten.erf.default(mul_525);  mul_525 = None
        add_145: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(erf_62, 1);  erf_62 = None
        mul_526: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_524, add_145);  mul_524 = add_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_527: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_526, 1.7015043497085571);  mul_526 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_70: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_211, getitem_141);  view_211 = getitem_141 = None
        add_146: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05);  getitem_140 = None
        rsqrt_70: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        mul_529: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = rsqrt_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_528: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg46_1, 0.0625);  arg46_1 = None
        view_212: "f32[512]" = torch.ops.aten.reshape.default(mul_528, [-1]);  mul_528 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_70: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_212, -1);  view_212 = None
        mul_530: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_529, unsqueeze_70);  mul_529 = unsqueeze_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_213: "f32[512, 256, 1, 1]" = torch.ops.aten.reshape.default(mul_530, [512, 256, 1, 1]);  mul_530 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_96: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_527, view_213, arg47_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_527 = view_213 = arg47_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_14: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_96, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_97: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_14, arg48_1, arg49_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_14 = arg48_1 = arg49_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_13: "f32[8, 256, 1, 1]" = torch.ops.aten.relu.default(convolution_97);  convolution_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_98: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_13, arg50_1, arg51_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_13 = arg50_1 = arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_13: "f32[8, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_98);  convolution_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_531: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_96, sigmoid_13);  convolution_96 = sigmoid_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_532: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_531, 2.0);  mul_531 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:198 in forward, code: out.mul_(self.skipinit_gain)
        mul_533: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_532, arg52_1);  mul_532 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_534: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_533, 0.2);  mul_533 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:101 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_3: "f32[8, 256, 32, 32]" = torch.ops.aten.avg_pool2d.default(mul_503, [2, 2], [2, 2], [0, 0], True, False);  mul_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_66: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_199, getitem_133);  view_199 = getitem_133 = None
        add_139: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_66: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        mul_505: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = rsqrt_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_504: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg34_1, 0.0625);  arg34_1 = None
        view_200: "f32[512]" = torch.ops.aten.reshape.default(mul_504, [-1]);  mul_504 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_66: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_200, -1);  view_200 = None
        mul_506: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_505, unsqueeze_66);  mul_505 = unsqueeze_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_201: "f32[512, 256, 1, 1]" = torch.ops.aten.reshape.default(mul_506, [512, 256, 1, 1]);  mul_506 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_92: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(avg_pool2d_3, view_201, arg35_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_3 = view_201 = arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        add_147: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_534, convolution_92);  mul_534 = convolution_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_535: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_147, 0.5)
        mul_536: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_147, 0.7071067811865476)
        erf_63: "f32[8, 512, 32, 32]" = torch.ops.aten.erf.default(mul_536);  mul_536 = None
        add_148: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(erf_63, 1);  erf_63 = None
        mul_537: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_535, add_148);  mul_535 = add_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_538: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_537, 1.7015043497085571);  mul_537 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        mul_539: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_538, 0.9805806756909201);  mul_538 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_71: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(view_214, getitem_143);  view_214 = getitem_143 = None
        add_149: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
        rsqrt_71: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_541: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = rsqrt_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_540: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg54_1, 0.04419417382415922);  arg54_1 = None
        view_215: "f32[256]" = torch.ops.aten.reshape.default(mul_540, [-1]);  mul_540 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_71: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_215, -1);  view_215 = None
        mul_542: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(mul_541, unsqueeze_71);  mul_541 = unsqueeze_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_216: "f32[256, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_542, [256, 512, 1, 1]);  mul_542 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_99: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_539, view_216, arg55_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_539 = view_216 = arg55_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_543: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_99, 0.5)
        mul_544: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_99, 0.7071067811865476);  convolution_99 = None
        erf_64: "f32[8, 256, 32, 32]" = torch.ops.aten.erf.default(mul_544);  mul_544 = None
        add_150: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(erf_64, 1);  erf_64 = None
        mul_545: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_543, add_150);  mul_543 = add_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_546: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_545, 1.7015043497085571);  mul_545 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_72: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_217, getitem_145);  view_217 = getitem_145 = None
        add_151: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_72: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        mul_548: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = rsqrt_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_547: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg57_1, 0.02946278254943948);  arg57_1 = None
        view_218: "f32[256]" = torch.ops.aten.reshape.default(mul_547, [-1]);  mul_547 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_72: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_218, -1);  view_218 = None
        mul_549: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(mul_548, unsqueeze_72);  mul_548 = unsqueeze_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_219: "f32[256, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_549, [256, 128, 3, 3]);  mul_549 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_100: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_546, view_219, arg58_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_546 = view_219 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_550: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_100, 0.5)
        mul_551: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_100, 0.7071067811865476);  convolution_100 = None
        erf_65: "f32[8, 256, 32, 32]" = torch.ops.aten.erf.default(mul_551);  mul_551 = None
        add_152: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(erf_65, 1);  erf_65 = None
        mul_552: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_550, add_152);  mul_550 = add_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_553: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_552, 1.7015043497085571);  mul_552 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_73: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_220, getitem_147);  view_220 = getitem_147 = None
        add_153: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
        rsqrt_73: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        mul_555: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = rsqrt_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_554: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg60_1, 0.02946278254943948);  arg60_1 = None
        view_221: "f32[256]" = torch.ops.aten.reshape.default(mul_554, [-1]);  mul_554 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_73: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_221, -1);  view_221 = None
        mul_556: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(mul_555, unsqueeze_73);  mul_555 = unsqueeze_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_222: "f32[256, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_556, [256, 128, 3, 3]);  mul_556 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_101: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_553, view_222, arg61_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_553 = view_222 = arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_557: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_101, 0.5)
        mul_558: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_101, 0.7071067811865476);  convolution_101 = None
        erf_66: "f32[8, 256, 32, 32]" = torch.ops.aten.erf.default(mul_558);  mul_558 = None
        add_154: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(erf_66, 1);  erf_66 = None
        mul_559: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_557, add_154);  mul_557 = add_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_560: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_559, 1.7015043497085571);  mul_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_74: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_223, getitem_149);  view_223 = getitem_149 = None
        add_155: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
        rsqrt_74: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        mul_562: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = rsqrt_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_561: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg63_1, 0.0625);  arg63_1 = None
        view_224: "f32[512]" = torch.ops.aten.reshape.default(mul_561, [-1]);  mul_561 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_74: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_224, -1);  view_224 = None
        mul_563: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_562, unsqueeze_74);  mul_562 = unsqueeze_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_225: "f32[512, 256, 1, 1]" = torch.ops.aten.reshape.default(mul_563, [512, 256, 1, 1]);  mul_563 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_102: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_560, view_225, arg64_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_560 = view_225 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_15: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_102, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_103: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_15, arg65_1, arg66_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_15 = arg65_1 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_14: "f32[8, 256, 1, 1]" = torch.ops.aten.relu.default(convolution_103);  convolution_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_104: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_14, arg67_1, arg68_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_14 = arg67_1 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_14: "f32[8, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_104);  convolution_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_564: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_102, sigmoid_14);  convolution_102 = sigmoid_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_565: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_564, 2.0);  mul_564 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:198 in forward, code: out.mul_(self.skipinit_gain)
        mul_566: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_565, arg69_1);  mul_565 = arg69_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_567: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_566, 0.2);  mul_566 = None
        add_156: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_567, add_147);  mul_567 = add_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_568: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_156, 0.5)
        mul_569: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_156, 0.7071067811865476);  add_156 = None
        erf_67: "f32[8, 512, 32, 32]" = torch.ops.aten.erf.default(mul_569);  mul_569 = None
        add_157: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(erf_67, 1);  erf_67 = None
        mul_570: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_568, add_157);  mul_568 = add_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_571: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_570, 1.7015043497085571);  mul_570 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        mul_572: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_571, 0.9622504486493761);  mul_571 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_76: "f32[1, 768, 512]" = torch.ops.aten.sub.Tensor(view_229, getitem_153);  view_229 = getitem_153 = None
        add_159: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05);  getitem_152 = None
        rsqrt_76: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        mul_577: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = rsqrt_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_576: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg74_1, 0.04419417382415922);  arg74_1 = None
        view_230: "f32[768]" = torch.ops.aten.reshape.default(mul_576, [-1]);  mul_576 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_76: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_230, -1);  view_230 = None
        mul_578: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(mul_577, unsqueeze_76);  mul_577 = unsqueeze_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_231: "f32[768, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_578, [768, 512, 1, 1]);  mul_578 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_106: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(mul_572, view_231, arg75_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_231 = arg75_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_579: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_106, 0.5)
        mul_580: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_106, 0.7071067811865476);  convolution_106 = None
        erf_68: "f32[8, 768, 32, 32]" = torch.ops.aten.erf.default(mul_580);  mul_580 = None
        add_160: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(erf_68, 1);  erf_68 = None
        mul_581: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_579, add_160);  mul_579 = add_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_582: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_581, 1.7015043497085571);  mul_581 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_8: "f32[8, 768, 33, 33]" = torch.ops.aten.constant_pad_nd.default(mul_582, [0, 1, 0, 1], 0.0);  mul_582 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_77: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_232, getitem_155);  view_232 = getitem_155 = None
        add_161: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_77: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        mul_584: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = rsqrt_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_583: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg77_1, 0.02946278254943948);  arg77_1 = None
        view_233: "f32[768]" = torch.ops.aten.reshape.default(mul_583, [-1]);  mul_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_77: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_233, -1);  view_233 = None
        mul_585: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_584, unsqueeze_77);  mul_584 = unsqueeze_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_234: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_585, [768, 128, 3, 3]);  mul_585 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_107: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(constant_pad_nd_8, view_234, arg78_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6);  constant_pad_nd_8 = view_234 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_586: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_107, 0.5)
        mul_587: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_107, 0.7071067811865476);  convolution_107 = None
        erf_69: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_587);  mul_587 = None
        add_162: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_69, 1);  erf_69 = None
        mul_588: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_586, add_162);  mul_586 = add_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_589: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_588, 1.7015043497085571);  mul_588 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_78: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_235, getitem_157);  view_235 = getitem_157 = None
        add_163: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_78: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        mul_591: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = rsqrt_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_590: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg80_1, 0.02946278254943948);  arg80_1 = None
        view_236: "f32[768]" = torch.ops.aten.reshape.default(mul_590, [-1]);  mul_590 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_78: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_236, -1);  view_236 = None
        mul_592: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_591, unsqueeze_78);  mul_591 = unsqueeze_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_237: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_592, [768, 128, 3, 3]);  mul_592 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_108: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_589, view_237, arg81_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_589 = view_237 = arg81_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_593: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_108, 0.5)
        mul_594: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_108, 0.7071067811865476);  convolution_108 = None
        erf_70: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_594);  mul_594 = None
        add_164: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_70, 1);  erf_70 = None
        mul_595: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_593, add_164);  mul_593 = add_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_596: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_595, 1.7015043497085571);  mul_595 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_79: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_238, getitem_159);  view_238 = getitem_159 = None
        add_165: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_79: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        mul_598: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = rsqrt_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_597: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg83_1, 0.03608439182435161);  arg83_1 = None
        view_239: "f32[1536]" = torch.ops.aten.reshape.default(mul_597, [-1]);  mul_597 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_79: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_239, -1);  view_239 = None
        mul_599: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_598, unsqueeze_79);  mul_598 = unsqueeze_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_240: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_599, [1536, 768, 1, 1]);  mul_599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_109: "f32[8, 1536, 16, 16]" = torch.ops.aten.convolution.default(mul_596, view_240, arg84_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_596 = view_240 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_16: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_109, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_110: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_16, arg85_1, arg86_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_16 = arg85_1 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_15: "f32[8, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_110);  convolution_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_111: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_15, arg87_1, arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_15 = arg87_1 = arg88_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_15: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_111);  convolution_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_600: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_109, sigmoid_15);  convolution_109 = sigmoid_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_601: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_600, 2.0);  mul_600 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:198 in forward, code: out.mul_(self.skipinit_gain)
        mul_602: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_601, arg89_1);  mul_601 = arg89_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_603: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_602, 0.2);  mul_602 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:101 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_4: "f32[8, 512, 16, 16]" = torch.ops.aten.avg_pool2d.default(mul_572, [2, 2], [2, 2], [0, 0], True, False);  mul_572 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_75: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_226, getitem_151);  view_226 = getitem_151 = None
        add_158: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05);  getitem_150 = None
        rsqrt_75: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        mul_574: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = rsqrt_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_573: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg71_1, 0.04419417382415922);  arg71_1 = None
        view_227: "f32[1536]" = torch.ops.aten.reshape.default(mul_573, [-1]);  mul_573 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_75: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_227, -1);  view_227 = None
        mul_575: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_75);  mul_574 = unsqueeze_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_228: "f32[1536, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_575, [1536, 512, 1, 1]);  mul_575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_105: "f32[8, 1536, 16, 16]" = torch.ops.aten.convolution.default(avg_pool2d_4, view_228, arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_4 = view_228 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        add_166: "f32[8, 1536, 16, 16]" = torch.ops.aten.add.Tensor(mul_603, convolution_105);  mul_603 = convolution_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_604: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(add_166, 0.5)
        mul_605: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(add_166, 0.7071067811865476)
        erf_71: "f32[8, 1536, 16, 16]" = torch.ops.aten.erf.default(mul_605);  mul_605 = None
        add_167: "f32[8, 1536, 16, 16]" = torch.ops.aten.add.Tensor(erf_71, 1);  erf_71 = None
        mul_606: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_604, add_167);  mul_604 = add_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_607: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_606, 1.7015043497085571);  mul_606 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        mul_608: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_607, 0.9805806756909201);  mul_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_80: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_241, getitem_161);  view_241 = getitem_161 = None
        add_168: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_80: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        mul_610: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = rsqrt_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_609: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg91_1, 0.02551551815399144);  arg91_1 = None
        view_242: "f32[768]" = torch.ops.aten.reshape.default(mul_609, [-1]);  mul_609 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_80: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_242, -1);  view_242 = None
        mul_611: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_610, unsqueeze_80);  mul_610 = unsqueeze_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_243: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_611, [768, 1536, 1, 1]);  mul_611 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_112: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_608, view_243, arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_608 = view_243 = arg92_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_612: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_112, 0.5)
        mul_613: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_112, 0.7071067811865476);  convolution_112 = None
        erf_72: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_613);  mul_613 = None
        add_169: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_72, 1);  erf_72 = None
        mul_614: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_612, add_169);  mul_612 = add_169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_615: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_614, 1.7015043497085571);  mul_614 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_81: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_244, getitem_163);  view_244 = getitem_163 = None
        add_170: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
        rsqrt_81: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        mul_617: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = rsqrt_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_616: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg94_1, 0.02946278254943948);  arg94_1 = None
        view_245: "f32[768]" = torch.ops.aten.reshape.default(mul_616, [-1]);  mul_616 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_81: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_245, -1);  view_245 = None
        mul_618: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_617, unsqueeze_81);  mul_617 = unsqueeze_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_246: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_618, [768, 128, 3, 3]);  mul_618 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_113: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_615, view_246, arg95_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_615 = view_246 = arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_619: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_113, 0.5)
        mul_620: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_113, 0.7071067811865476);  convolution_113 = None
        erf_73: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_620);  mul_620 = None
        add_171: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_73, 1);  erf_73 = None
        mul_621: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_619, add_171);  mul_619 = add_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_622: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_621, 1.7015043497085571);  mul_621 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_82: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_247, getitem_165);  view_247 = getitem_165 = None
        add_172: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-05);  getitem_164 = None
        rsqrt_82: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        mul_624: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = rsqrt_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_623: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg97_1, 0.02946278254943948);  arg97_1 = None
        view_248: "f32[768]" = torch.ops.aten.reshape.default(mul_623, [-1]);  mul_623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_82: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_248, -1);  view_248 = None
        mul_625: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_624, unsqueeze_82);  mul_624 = unsqueeze_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_249: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_625, [768, 128, 3, 3]);  mul_625 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_114: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_622, view_249, arg98_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_622 = view_249 = arg98_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_626: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_114, 0.5)
        mul_627: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_114, 0.7071067811865476);  convolution_114 = None
        erf_74: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_627);  mul_627 = None
        add_173: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_74, 1);  erf_74 = None
        mul_628: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_626, add_173);  mul_626 = add_173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_629: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_628, 1.7015043497085571);  mul_628 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_83: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_250, getitem_167);  view_250 = getitem_167 = None
        add_174: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05);  getitem_166 = None
        rsqrt_83: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_631: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = rsqrt_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_630: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg100_1, 0.03608439182435161);  arg100_1 = None
        view_251: "f32[1536]" = torch.ops.aten.reshape.default(mul_630, [-1]);  mul_630 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_83: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_251, -1);  view_251 = None
        mul_632: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_631, unsqueeze_83);  mul_631 = unsqueeze_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_252: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_632, [1536, 768, 1, 1]);  mul_632 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_115: "f32[8, 1536, 16, 16]" = torch.ops.aten.convolution.default(mul_629, view_252, arg101_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_629 = view_252 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_17: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_115, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_116: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_17, arg102_1, arg103_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_17 = arg102_1 = arg103_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_16: "f32[8, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_116);  convolution_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_117: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_16, arg104_1, arg105_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_16 = arg104_1 = arg105_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_16: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_117);  convolution_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_633: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_115, sigmoid_16);  convolution_115 = sigmoid_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_634: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_633, 2.0);  mul_633 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:198 in forward, code: out.mul_(self.skipinit_gain)
        mul_635: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_634, arg106_1);  mul_634 = arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_636: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_635, 0.2);  mul_635 = None
        add_175: "f32[8, 1536, 16, 16]" = torch.ops.aten.add.Tensor(mul_636, add_166);  mul_636 = add_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_637: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(add_175, 0.5)
        mul_638: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(add_175, 0.7071067811865476)
        erf_75: "f32[8, 1536, 16, 16]" = torch.ops.aten.erf.default(mul_638);  mul_638 = None
        add_176: "f32[8, 1536, 16, 16]" = torch.ops.aten.add.Tensor(erf_75, 1);  erf_75 = None
        mul_639: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_637, add_176);  mul_637 = add_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_640: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_639, 1.7015043497085571);  mul_639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        mul_641: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_640, 0.9622504486493761);  mul_640 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_84: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_253, getitem_169);  view_253 = getitem_169 = None
        add_177: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-05);  getitem_168 = None
        rsqrt_84: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        mul_643: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = rsqrt_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_642: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg108_1, 0.02551551815399144);  arg108_1 = None
        view_254: "f32[768]" = torch.ops.aten.reshape.default(mul_642, [-1]);  mul_642 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_84: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_254, -1);  view_254 = None
        mul_644: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_643, unsqueeze_84);  mul_643 = unsqueeze_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_255: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_644, [768, 1536, 1, 1]);  mul_644 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_118: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_641, view_255, arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_641 = view_255 = arg109_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_645: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_118, 0.5)
        mul_646: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_118, 0.7071067811865476);  convolution_118 = None
        erf_76: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_646);  mul_646 = None
        add_178: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_76, 1);  erf_76 = None
        mul_647: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_645, add_178);  mul_645 = add_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_648: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_647, 1.7015043497085571);  mul_647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_85: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_256, getitem_171);  view_256 = getitem_171 = None
        add_179: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-05);  getitem_170 = None
        rsqrt_85: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
        mul_650: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = rsqrt_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_649: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg111_1, 0.02946278254943948);  arg111_1 = None
        view_257: "f32[768]" = torch.ops.aten.reshape.default(mul_649, [-1]);  mul_649 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_85: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_257, -1);  view_257 = None
        mul_651: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_650, unsqueeze_85);  mul_650 = unsqueeze_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_258: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_651, [768, 128, 3, 3]);  mul_651 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_119: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_648, view_258, arg112_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_648 = view_258 = arg112_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_652: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_119, 0.5)
        mul_653: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_119, 0.7071067811865476);  convolution_119 = None
        erf_77: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_653);  mul_653 = None
        add_180: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_77, 1);  erf_77 = None
        mul_654: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_652, add_180);  mul_652 = add_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_655: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_654, 1.7015043497085571);  mul_654 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_86: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_259, getitem_173);  view_259 = getitem_173 = None
        add_181: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05);  getitem_172 = None
        rsqrt_86: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        mul_657: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = rsqrt_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_656: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg114_1, 0.02946278254943948);  arg114_1 = None
        view_260: "f32[768]" = torch.ops.aten.reshape.default(mul_656, [-1]);  mul_656 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_86: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_260, -1);  view_260 = None
        mul_658: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_657, unsqueeze_86);  mul_657 = unsqueeze_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_261: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_658, [768, 128, 3, 3]);  mul_658 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_120: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_655, view_261, arg115_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_655 = view_261 = arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_659: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_120, 0.5)
        mul_660: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_120, 0.7071067811865476);  convolution_120 = None
        erf_78: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_660);  mul_660 = None
        add_182: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_78, 1);  erf_78 = None
        mul_661: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_659, add_182);  mul_659 = add_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_662: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_661, 1.7015043497085571);  mul_661 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_87: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_262, getitem_175);  view_262 = getitem_175 = None
        add_183: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_87: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
        mul_664: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = rsqrt_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_663: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg117_1, 0.03608439182435161);  arg117_1 = None
        view_263: "f32[1536]" = torch.ops.aten.reshape.default(mul_663, [-1]);  mul_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_87: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_263, -1);  view_263 = None
        mul_665: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_664, unsqueeze_87);  mul_664 = unsqueeze_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_264: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_665, [1536, 768, 1, 1]);  mul_665 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_121: "f32[8, 1536, 16, 16]" = torch.ops.aten.convolution.default(mul_662, view_264, arg118_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_662 = view_264 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_18: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_121, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_122: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_18, arg119_1, arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_18 = arg119_1 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_17: "f32[8, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_122);  convolution_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_123: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_17, arg121_1, arg122_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg121_1 = arg122_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_17: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_123);  convolution_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_666: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_121, sigmoid_17);  convolution_121 = sigmoid_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_667: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_666, 2.0);  mul_666 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:198 in forward, code: out.mul_(self.skipinit_gain)
        mul_668: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_667, arg123_1);  mul_667 = arg123_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_669: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_668, 0.2);  mul_668 = None
        add_184: "f32[8, 1536, 16, 16]" = torch.ops.aten.add.Tensor(mul_669, add_175);  mul_669 = add_175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_670: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(add_184, 0.5)
        mul_671: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(add_184, 0.7071067811865476)
        erf_79: "f32[8, 1536, 16, 16]" = torch.ops.aten.erf.default(mul_671);  mul_671 = None
        add_185: "f32[8, 1536, 16, 16]" = torch.ops.aten.add.Tensor(erf_79, 1);  erf_79 = None
        mul_672: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_670, add_185);  mul_670 = add_185 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_673: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_672, 1.7015043497085571);  mul_672 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        mul_674: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_673, 0.9449111825230679);  mul_673 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_88: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_265, getitem_177);  view_265 = getitem_177 = None
        add_186: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_88: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        mul_676: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = rsqrt_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_675: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg125_1, 0.02551551815399144);  arg125_1 = None
        view_266: "f32[768]" = torch.ops.aten.reshape.default(mul_675, [-1]);  mul_675 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_88: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_266, -1);  view_266 = None
        mul_677: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_676, unsqueeze_88);  mul_676 = unsqueeze_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_267: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_677, [768, 1536, 1, 1]);  mul_677 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_124: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_674, view_267, arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_674 = view_267 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_678: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_124, 0.5)
        mul_679: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_124, 0.7071067811865476);  convolution_124 = None
        erf_80: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_679);  mul_679 = None
        add_187: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_80, 1);  erf_80 = None
        mul_680: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_678, add_187);  mul_678 = add_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_681: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_680, 1.7015043497085571);  mul_680 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_89: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_268, getitem_179);  view_268 = getitem_179 = None
        add_188: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
        rsqrt_89: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        mul_683: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = rsqrt_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_682: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg128_1, 0.02946278254943948);  arg128_1 = None
        view_269: "f32[768]" = torch.ops.aten.reshape.default(mul_682, [-1]);  mul_682 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_89: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_269, -1);  view_269 = None
        mul_684: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_683, unsqueeze_89);  mul_683 = unsqueeze_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_270: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_684, [768, 128, 3, 3]);  mul_684 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_125: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_681, view_270, arg129_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_681 = view_270 = arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_685: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_125, 0.5)
        mul_686: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_125, 0.7071067811865476);  convolution_125 = None
        erf_81: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_686);  mul_686 = None
        add_189: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_81, 1);  erf_81 = None
        mul_687: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_685, add_189);  mul_685 = add_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_688: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_687, 1.7015043497085571);  mul_687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_90: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_271, getitem_181);  view_271 = getitem_181 = None
        add_190: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-05);  getitem_180 = None
        rsqrt_90: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        mul_690: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = rsqrt_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_689: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg131_1, 0.02946278254943948);  arg131_1 = None
        view_272: "f32[768]" = torch.ops.aten.reshape.default(mul_689, [-1]);  mul_689 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_90: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_272, -1);  view_272 = None
        mul_691: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_690, unsqueeze_90);  mul_690 = unsqueeze_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_273: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_691, [768, 128, 3, 3]);  mul_691 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_126: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_688, view_273, arg132_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_688 = view_273 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_692: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_126, 0.5)
        mul_693: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_126, 0.7071067811865476);  convolution_126 = None
        erf_82: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_693);  mul_693 = None
        add_191: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_82, 1);  erf_82 = None
        mul_694: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_692, add_191);  mul_692 = add_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_695: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_694, 1.7015043497085571);  mul_694 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_91: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_274, getitem_183);  view_274 = getitem_183 = None
        add_192: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-05);  getitem_182 = None
        rsqrt_91: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_697: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = rsqrt_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_696: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg134_1, 0.03608439182435161);  arg134_1 = None
        view_275: "f32[1536]" = torch.ops.aten.reshape.default(mul_696, [-1]);  mul_696 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_91: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_275, -1);  view_275 = None
        mul_698: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_697, unsqueeze_91);  mul_697 = unsqueeze_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_276: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_698, [1536, 768, 1, 1]);  mul_698 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_127: "f32[8, 1536, 16, 16]" = torch.ops.aten.convolution.default(mul_695, view_276, arg135_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_695 = view_276 = arg135_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_19: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_127, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_128: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_19, arg136_1, arg137_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_19 = arg136_1 = arg137_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_18: "f32[8, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_128);  convolution_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_129: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_18, arg138_1, arg139_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_18 = arg138_1 = arg139_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_18: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_129);  convolution_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_699: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_127, sigmoid_18);  convolution_127 = sigmoid_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_700: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_699, 2.0);  mul_699 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:198 in forward, code: out.mul_(self.skipinit_gain)
        mul_701: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_700, arg140_1);  mul_700 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_702: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_701, 0.2);  mul_701 = None
        add_193: "f32[8, 1536, 16, 16]" = torch.ops.aten.add.Tensor(mul_702, add_184);  mul_702 = add_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_703: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(add_193, 0.5)
        mul_704: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(add_193, 0.7071067811865476)
        erf_83: "f32[8, 1536, 16, 16]" = torch.ops.aten.erf.default(mul_704);  mul_704 = None
        add_194: "f32[8, 1536, 16, 16]" = torch.ops.aten.add.Tensor(erf_83, 1);  erf_83 = None
        mul_705: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_703, add_194);  mul_703 = add_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_706: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_705, 1.7015043497085571);  mul_705 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        mul_707: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_706, 0.9284766908852592);  mul_706 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_92: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_277, getitem_185);  view_277 = getitem_185 = None
        add_195: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-05);  getitem_184 = None
        rsqrt_92: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        mul_709: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = rsqrt_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_708: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg142_1, 0.02551551815399144);  arg142_1 = None
        view_278: "f32[768]" = torch.ops.aten.reshape.default(mul_708, [-1]);  mul_708 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_92: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_278, -1);  view_278 = None
        mul_710: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_709, unsqueeze_92);  mul_709 = unsqueeze_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_279: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_710, [768, 1536, 1, 1]);  mul_710 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_130: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_707, view_279, arg143_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_707 = view_279 = arg143_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_711: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_130, 0.5)
        mul_712: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_130, 0.7071067811865476);  convolution_130 = None
        erf_84: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_712);  mul_712 = None
        add_196: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_84, 1);  erf_84 = None
        mul_713: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_711, add_196);  mul_711 = add_196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_714: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_713, 1.7015043497085571);  mul_713 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_93: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_280, getitem_187);  view_280 = getitem_187 = None
        add_197: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_93: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        mul_716: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = rsqrt_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_715: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg145_1, 0.02946278254943948);  arg145_1 = None
        view_281: "f32[768]" = torch.ops.aten.reshape.default(mul_715, [-1]);  mul_715 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_93: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_281, -1);  view_281 = None
        mul_717: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_716, unsqueeze_93);  mul_716 = unsqueeze_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_282: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_717, [768, 128, 3, 3]);  mul_717 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_131: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_714, view_282, arg146_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_714 = view_282 = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_718: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_131, 0.5)
        mul_719: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_131, 0.7071067811865476);  convolution_131 = None
        erf_85: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_719);  mul_719 = None
        add_198: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_85, 1);  erf_85 = None
        mul_720: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_718, add_198);  mul_718 = add_198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_721: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_720, 1.7015043497085571);  mul_720 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_94: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_283, getitem_189);  view_283 = getitem_189 = None
        add_199: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-05);  getitem_188 = None
        rsqrt_94: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        mul_723: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = rsqrt_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_722: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg148_1, 0.02946278254943948);  arg148_1 = None
        view_284: "f32[768]" = torch.ops.aten.reshape.default(mul_722, [-1]);  mul_722 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_94: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_284, -1);  view_284 = None
        mul_724: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_723, unsqueeze_94);  mul_723 = unsqueeze_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_285: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_724, [768, 128, 3, 3]);  mul_724 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_132: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_721, view_285, arg149_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_721 = view_285 = arg149_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_725: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_132, 0.5)
        mul_726: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_132, 0.7071067811865476);  convolution_132 = None
        erf_86: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_726);  mul_726 = None
        add_200: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_86, 1);  erf_86 = None
        mul_727: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_725, add_200);  mul_725 = add_200 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_728: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_727, 1.7015043497085571);  mul_727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_95: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_286, getitem_191);  view_286 = getitem_191 = None
        add_201: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_95: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        mul_730: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = rsqrt_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_729: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg151_1, 0.03608439182435161);  arg151_1 = None
        view_287: "f32[1536]" = torch.ops.aten.reshape.default(mul_729, [-1]);  mul_729 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_95: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_287, -1);  view_287 = None
        mul_731: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_730, unsqueeze_95);  mul_730 = unsqueeze_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_288: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_731, [1536, 768, 1, 1]);  mul_731 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_133: "f32[8, 1536, 16, 16]" = torch.ops.aten.convolution.default(mul_728, view_288, arg152_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_728 = view_288 = arg152_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_20: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_133, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_134: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_20, arg153_1, arg154_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_20 = arg153_1 = arg154_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_19: "f32[8, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_134);  convolution_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_135: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_19, arg155_1, arg156_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_19 = arg155_1 = arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_19: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_135);  convolution_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_732: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_133, sigmoid_19);  convolution_133 = sigmoid_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_733: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_732, 2.0);  mul_732 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:198 in forward, code: out.mul_(self.skipinit_gain)
        mul_734: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_733, arg157_1);  mul_733 = arg157_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_735: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_734, 0.2);  mul_734 = None
        add_202: "f32[8, 1536, 16, 16]" = torch.ops.aten.add.Tensor(mul_735, add_193);  mul_735 = add_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_736: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(add_202, 0.5)
        mul_737: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(add_202, 0.7071067811865476)
        erf_87: "f32[8, 1536, 16, 16]" = torch.ops.aten.erf.default(mul_737);  mul_737 = None
        add_203: "f32[8, 1536, 16, 16]" = torch.ops.aten.add.Tensor(erf_87, 1);  erf_87 = None
        mul_738: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_736, add_203);  mul_736 = add_203 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_739: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_738, 1.7015043497085571);  mul_738 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        mul_740: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_739, 0.9128709291752768);  mul_739 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_96: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_289, getitem_193);  view_289 = getitem_193 = None
        add_204: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_96: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
        mul_742: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = rsqrt_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_741: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg159_1, 0.02551551815399144);  arg159_1 = None
        view_290: "f32[768]" = torch.ops.aten.reshape.default(mul_741, [-1]);  mul_741 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_96: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_290, -1);  view_290 = None
        mul_743: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_742, unsqueeze_96);  mul_742 = unsqueeze_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_291: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_743, [768, 1536, 1, 1]);  mul_743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_136: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_740, view_291, arg160_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_740 = view_291 = arg160_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_744: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_136, 0.5)
        mul_745: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_136, 0.7071067811865476);  convolution_136 = None
        erf_88: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_745);  mul_745 = None
        add_205: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_88, 1);  erf_88 = None
        mul_746: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_744, add_205);  mul_744 = add_205 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_747: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_746, 1.7015043497085571);  mul_746 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_97: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_292, getitem_195);  view_292 = getitem_195 = None
        add_206: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-05);  getitem_194 = None
        rsqrt_97: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        mul_749: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = rsqrt_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_748: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg162_1, 0.02946278254943948);  arg162_1 = None
        view_293: "f32[768]" = torch.ops.aten.reshape.default(mul_748, [-1]);  mul_748 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_97: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_293, -1);  view_293 = None
        mul_750: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_749, unsqueeze_97);  mul_749 = unsqueeze_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_294: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_750, [768, 128, 3, 3]);  mul_750 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_137: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_747, view_294, arg163_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_747 = view_294 = arg163_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_751: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_137, 0.5)
        mul_752: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_137, 0.7071067811865476);  convolution_137 = None
        erf_89: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_752);  mul_752 = None
        add_207: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_89, 1);  erf_89 = None
        mul_753: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_751, add_207);  mul_751 = add_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_754: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_753, 1.7015043497085571);  mul_753 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_98: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_295, getitem_197);  view_295 = getitem_197 = None
        add_208: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_98: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
        mul_756: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = rsqrt_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_755: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg165_1, 0.02946278254943948);  arg165_1 = None
        view_296: "f32[768]" = torch.ops.aten.reshape.default(mul_755, [-1]);  mul_755 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_98: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_296, -1);  view_296 = None
        mul_757: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_756, unsqueeze_98);  mul_756 = unsqueeze_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_297: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_757, [768, 128, 3, 3]);  mul_757 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_138: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_754, view_297, arg166_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_754 = view_297 = arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_758: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_138, 0.5)
        mul_759: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_138, 0.7071067811865476);  convolution_138 = None
        erf_90: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_759);  mul_759 = None
        add_209: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_90, 1);  erf_90 = None
        mul_760: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_758, add_209);  mul_758 = add_209 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_761: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_760, 1.7015043497085571);  mul_760 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_99: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_298, getitem_199);  view_298 = getitem_199 = None
        add_210: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_99: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        mul_763: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = rsqrt_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_762: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg168_1, 0.03608439182435161);  arg168_1 = None
        view_299: "f32[1536]" = torch.ops.aten.reshape.default(mul_762, [-1]);  mul_762 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_99: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_299, -1);  view_299 = None
        mul_764: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_763, unsqueeze_99);  mul_763 = unsqueeze_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_300: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_764, [1536, 768, 1, 1]);  mul_764 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_139: "f32[8, 1536, 16, 16]" = torch.ops.aten.convolution.default(mul_761, view_300, arg169_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_761 = view_300 = arg169_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_21: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_139, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_140: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_21, arg170_1, arg171_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_21 = arg170_1 = arg171_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_20: "f32[8, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_140);  convolution_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_141: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_20, arg172_1, arg173_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_20 = arg172_1 = arg173_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_20: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_141);  convolution_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_765: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_139, sigmoid_20);  convolution_139 = sigmoid_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_766: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_765, 2.0);  mul_765 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:198 in forward, code: out.mul_(self.skipinit_gain)
        mul_767: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_766, arg174_1);  mul_766 = arg174_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_768: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_767, 0.2);  mul_767 = None
        add_211: "f32[8, 1536, 16, 16]" = torch.ops.aten.add.Tensor(mul_768, add_202);  mul_768 = add_202 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_769: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(add_211, 0.5)
        mul_770: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(add_211, 0.7071067811865476);  add_211 = None
        erf_91: "f32[8, 1536, 16, 16]" = torch.ops.aten.erf.default(mul_770);  mul_770 = None
        add_212: "f32[8, 1536, 16, 16]" = torch.ops.aten.add.Tensor(erf_91, 1);  erf_91 = None
        mul_771: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_769, add_212);  mul_769 = add_212 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_772: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_771, 1.7015043497085571);  mul_771 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        mul_773: "f32[8, 1536, 16, 16]" = torch.ops.aten.mul.Tensor(mul_772, 0.8980265101338745);  mul_772 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_101: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_304, getitem_203);  view_304 = getitem_203 = None
        add_214: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_202, 1e-05);  getitem_202 = None
        rsqrt_101: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        mul_778: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = rsqrt_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_777: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg179_1, 0.02551551815399144);  arg179_1 = None
        view_305: "f32[768]" = torch.ops.aten.reshape.default(mul_777, [-1]);  mul_777 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_101: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_305, -1);  view_305 = None
        mul_779: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_778, unsqueeze_101);  mul_778 = unsqueeze_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_306: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_779, [768, 1536, 1, 1]);  mul_779 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_143: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_773, view_306, arg180_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_306 = arg180_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_780: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_143, 0.5)
        mul_781: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_143, 0.7071067811865476);  convolution_143 = None
        erf_92: "f32[8, 768, 16, 16]" = torch.ops.aten.erf.default(mul_781);  mul_781 = None
        add_215: "f32[8, 768, 16, 16]" = torch.ops.aten.add.Tensor(erf_92, 1);  erf_92 = None
        mul_782: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_780, add_215);  mul_780 = add_215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_783: "f32[8, 768, 16, 16]" = torch.ops.aten.mul.Tensor(mul_782, 1.7015043497085571);  mul_782 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_9: "f32[8, 768, 17, 17]" = torch.ops.aten.constant_pad_nd.default(mul_783, [0, 1, 0, 1], 0.0);  mul_783 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_102: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_307, getitem_205);  view_307 = getitem_205 = None
        add_216: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_204, 1e-05);  getitem_204 = None
        rsqrt_102: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        mul_785: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = rsqrt_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_784: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg182_1, 0.02946278254943948);  arg182_1 = None
        view_308: "f32[768]" = torch.ops.aten.reshape.default(mul_784, [-1]);  mul_784 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_102: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_308, -1);  view_308 = None
        mul_786: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_785, unsqueeze_102);  mul_785 = unsqueeze_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_309: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_786, [768, 128, 3, 3]);  mul_786 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_144: "f32[8, 768, 8, 8]" = torch.ops.aten.convolution.default(constant_pad_nd_9, view_309, arg183_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6);  constant_pad_nd_9 = view_309 = arg183_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_787: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_144, 0.5)
        mul_788: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_144, 0.7071067811865476);  convolution_144 = None
        erf_93: "f32[8, 768, 8, 8]" = torch.ops.aten.erf.default(mul_788);  mul_788 = None
        add_217: "f32[8, 768, 8, 8]" = torch.ops.aten.add.Tensor(erf_93, 1);  erf_93 = None
        mul_789: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_787, add_217);  mul_787 = add_217 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_790: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_789, 1.7015043497085571);  mul_789 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_103: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_310, getitem_207);  view_310 = getitem_207 = None
        add_218: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_103: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        mul_792: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = rsqrt_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_791: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg185_1, 0.02946278254943948);  arg185_1 = None
        view_311: "f32[768]" = torch.ops.aten.reshape.default(mul_791, [-1]);  mul_791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_103: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_311, -1);  view_311 = None
        mul_793: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_792, unsqueeze_103);  mul_792 = unsqueeze_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_312: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_793, [768, 128, 3, 3]);  mul_793 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_145: "f32[8, 768, 8, 8]" = torch.ops.aten.convolution.default(mul_790, view_312, arg186_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_790 = view_312 = arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_794: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_145, 0.5)
        mul_795: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_145, 0.7071067811865476);  convolution_145 = None
        erf_94: "f32[8, 768, 8, 8]" = torch.ops.aten.erf.default(mul_795);  mul_795 = None
        add_219: "f32[8, 768, 8, 8]" = torch.ops.aten.add.Tensor(erf_94, 1);  erf_94 = None
        mul_796: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_794, add_219);  mul_794 = add_219 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_797: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_796, 1.7015043497085571);  mul_796 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_104: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_313, getitem_209);  view_313 = getitem_209 = None
        add_220: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_104: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        mul_799: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_104);  sub_104 = rsqrt_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_798: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg188_1, 0.03608439182435161);  arg188_1 = None
        view_314: "f32[1536]" = torch.ops.aten.reshape.default(mul_798, [-1]);  mul_798 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_104: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_314, -1);  view_314 = None
        mul_800: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_799, unsqueeze_104);  mul_799 = unsqueeze_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_315: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_800, [1536, 768, 1, 1]);  mul_800 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_146: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_797, view_315, arg189_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_797 = view_315 = arg189_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_22: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_146, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_147: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_22, arg190_1, arg191_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_22 = arg190_1 = arg191_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_21: "f32[8, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_147);  convolution_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_148: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_21, arg192_1, arg193_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_21 = arg192_1 = arg193_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_21: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_148);  convolution_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_801: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_146, sigmoid_21);  convolution_146 = sigmoid_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_802: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_801, 2.0);  mul_801 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:198 in forward, code: out.mul_(self.skipinit_gain)
        mul_803: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_802, arg194_1);  mul_802 = arg194_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_804: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_803, 0.2);  mul_803 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:101 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_5: "f32[8, 1536, 8, 8]" = torch.ops.aten.avg_pool2d.default(mul_773, [2, 2], [2, 2], [0, 0], True, False);  mul_773 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_100: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_301, getitem_201);  view_301 = getitem_201 = None
        add_213: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-05);  getitem_200 = None
        rsqrt_100: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        mul_775: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = rsqrt_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_774: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg176_1, 0.02551551815399144);  arg176_1 = None
        view_302: "f32[1536]" = torch.ops.aten.reshape.default(mul_774, [-1]);  mul_774 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_100: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_302, -1);  view_302 = None
        mul_776: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(mul_775, unsqueeze_100);  mul_775 = unsqueeze_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_303: "f32[1536, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_776, [1536, 1536, 1, 1]);  mul_776 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_142: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(avg_pool2d_5, view_303, arg177_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_5 = view_303 = arg177_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        add_221: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_804, convolution_142);  mul_804 = convolution_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_805: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_221, 0.5)
        mul_806: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_221, 0.7071067811865476)
        erf_95: "f32[8, 1536, 8, 8]" = torch.ops.aten.erf.default(mul_806);  mul_806 = None
        add_222: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(erf_95, 1);  erf_95 = None
        mul_807: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_805, add_222);  mul_805 = add_222 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_808: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_807, 1.7015043497085571);  mul_807 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        mul_809: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_808, 0.9805806756909201);  mul_808 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_105: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_316, getitem_211);  view_316 = getitem_211 = None
        add_223: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_105: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        mul_811: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_105);  sub_105 = rsqrt_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_810: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg196_1, 0.02551551815399144);  arg196_1 = None
        view_317: "f32[768]" = torch.ops.aten.reshape.default(mul_810, [-1]);  mul_810 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_105: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_317, -1);  view_317 = None
        mul_812: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_811, unsqueeze_105);  mul_811 = unsqueeze_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_318: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_812, [768, 1536, 1, 1]);  mul_812 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_149: "f32[8, 768, 8, 8]" = torch.ops.aten.convolution.default(mul_809, view_318, arg197_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_809 = view_318 = arg197_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_813: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_149, 0.5)
        mul_814: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_149, 0.7071067811865476);  convolution_149 = None
        erf_96: "f32[8, 768, 8, 8]" = torch.ops.aten.erf.default(mul_814);  mul_814 = None
        add_224: "f32[8, 768, 8, 8]" = torch.ops.aten.add.Tensor(erf_96, 1);  erf_96 = None
        mul_815: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_813, add_224);  mul_813 = add_224 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_816: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_815, 1.7015043497085571);  mul_815 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_106: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_319, getitem_213);  view_319 = getitem_213 = None
        add_225: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-05);  getitem_212 = None
        rsqrt_106: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        mul_818: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_106);  sub_106 = rsqrt_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_817: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg199_1, 0.02946278254943948);  arg199_1 = None
        view_320: "f32[768]" = torch.ops.aten.reshape.default(mul_817, [-1]);  mul_817 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_106: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_320, -1);  view_320 = None
        mul_819: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_818, unsqueeze_106);  mul_818 = unsqueeze_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_321: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_819, [768, 128, 3, 3]);  mul_819 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_150: "f32[8, 768, 8, 8]" = torch.ops.aten.convolution.default(mul_816, view_321, arg200_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_816 = view_321 = arg200_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_820: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_150, 0.5)
        mul_821: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_150, 0.7071067811865476);  convolution_150 = None
        erf_97: "f32[8, 768, 8, 8]" = torch.ops.aten.erf.default(mul_821);  mul_821 = None
        add_226: "f32[8, 768, 8, 8]" = torch.ops.aten.add.Tensor(erf_97, 1);  erf_97 = None
        mul_822: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_820, add_226);  mul_820 = add_226 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_823: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_822, 1.7015043497085571);  mul_822 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_107: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_322, getitem_215);  view_322 = getitem_215 = None
        add_227: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_214, 1e-05);  getitem_214 = None
        rsqrt_107: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
        mul_825: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_107);  sub_107 = rsqrt_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_824: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg202_1, 0.02946278254943948);  arg202_1 = None
        view_323: "f32[768]" = torch.ops.aten.reshape.default(mul_824, [-1]);  mul_824 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_107: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_323, -1);  view_323 = None
        mul_826: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_825, unsqueeze_107);  mul_825 = unsqueeze_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_324: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_826, [768, 128, 3, 3]);  mul_826 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_151: "f32[8, 768, 8, 8]" = torch.ops.aten.convolution.default(mul_823, view_324, arg203_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_823 = view_324 = arg203_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_827: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_151, 0.5)
        mul_828: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_151, 0.7071067811865476);  convolution_151 = None
        erf_98: "f32[8, 768, 8, 8]" = torch.ops.aten.erf.default(mul_828);  mul_828 = None
        add_228: "f32[8, 768, 8, 8]" = torch.ops.aten.add.Tensor(erf_98, 1);  erf_98 = None
        mul_829: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_827, add_228);  mul_827 = add_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_830: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_829, 1.7015043497085571);  mul_829 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_108: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_325, getitem_217);  view_325 = getitem_217 = None
        add_229: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_216, 1e-05);  getitem_216 = None
        rsqrt_108: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        mul_832: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_108);  sub_108 = rsqrt_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_831: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg205_1, 0.03608439182435161);  arg205_1 = None
        view_326: "f32[1536]" = torch.ops.aten.reshape.default(mul_831, [-1]);  mul_831 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_108: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_326, -1);  view_326 = None
        mul_833: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_832, unsqueeze_108);  mul_832 = unsqueeze_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_327: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_833, [1536, 768, 1, 1]);  mul_833 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_152: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_830, view_327, arg206_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_830 = view_327 = arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_23: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_152, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_153: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_23, arg207_1, arg208_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_23 = arg207_1 = arg208_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_22: "f32[8, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_153);  convolution_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_154: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_22, arg209_1, arg210_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_22 = arg209_1 = arg210_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_22: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_154);  convolution_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_834: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_152, sigmoid_22);  convolution_152 = sigmoid_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_835: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_834, 2.0);  mul_834 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:198 in forward, code: out.mul_(self.skipinit_gain)
        mul_836: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_835, arg211_1);  mul_835 = arg211_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_837: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_836, 0.2);  mul_836 = None
        add_230: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_837, add_221);  mul_837 = add_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_838: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_230, 0.5)
        mul_839: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_230, 0.7071067811865476)
        erf_99: "f32[8, 1536, 8, 8]" = torch.ops.aten.erf.default(mul_839);  mul_839 = None
        add_231: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(erf_99, 1);  erf_99 = None
        mul_840: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_838, add_231);  mul_838 = add_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_841: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_840, 1.7015043497085571);  mul_840 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        mul_842: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_841, 0.9622504486493761);  mul_841 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_109: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_328, getitem_219);  view_328 = getitem_219 = None
        add_232: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-05);  getitem_218 = None
        rsqrt_109: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
        mul_844: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_109);  sub_109 = rsqrt_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_843: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg213_1, 0.02551551815399144);  arg213_1 = None
        view_329: "f32[768]" = torch.ops.aten.reshape.default(mul_843, [-1]);  mul_843 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_109: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_329, -1);  view_329 = None
        mul_845: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(mul_844, unsqueeze_109);  mul_844 = unsqueeze_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_330: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_845, [768, 1536, 1, 1]);  mul_845 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_155: "f32[8, 768, 8, 8]" = torch.ops.aten.convolution.default(mul_842, view_330, arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_842 = view_330 = arg214_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_846: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_155, 0.5)
        mul_847: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_155, 0.7071067811865476);  convolution_155 = None
        erf_100: "f32[8, 768, 8, 8]" = torch.ops.aten.erf.default(mul_847);  mul_847 = None
        add_233: "f32[8, 768, 8, 8]" = torch.ops.aten.add.Tensor(erf_100, 1);  erf_100 = None
        mul_848: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_846, add_233);  mul_846 = add_233 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_849: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_848, 1.7015043497085571);  mul_848 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_110: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_331, getitem_221);  view_331 = getitem_221 = None
        add_234: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-05);  getitem_220 = None
        rsqrt_110: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
        mul_851: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_110);  sub_110 = rsqrt_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_850: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg216_1, 0.02946278254943948);  arg216_1 = None
        view_332: "f32[768]" = torch.ops.aten.reshape.default(mul_850, [-1]);  mul_850 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_110: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_332, -1);  view_332 = None
        mul_852: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_851, unsqueeze_110);  mul_851 = unsqueeze_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_333: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_852, [768, 128, 3, 3]);  mul_852 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_156: "f32[8, 768, 8, 8]" = torch.ops.aten.convolution.default(mul_849, view_333, arg217_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_849 = view_333 = arg217_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_853: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_156, 0.5)
        mul_854: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_156, 0.7071067811865476);  convolution_156 = None
        erf_101: "f32[8, 768, 8, 8]" = torch.ops.aten.erf.default(mul_854);  mul_854 = None
        add_235: "f32[8, 768, 8, 8]" = torch.ops.aten.add.Tensor(erf_101, 1);  erf_101 = None
        mul_855: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_853, add_235);  mul_853 = add_235 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_856: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_855, 1.7015043497085571);  mul_855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_111: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_334, getitem_223);  view_334 = getitem_223 = None
        add_236: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-05);  getitem_222 = None
        rsqrt_111: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        mul_858: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_111);  sub_111 = rsqrt_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_857: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg219_1, 0.02946278254943948);  arg219_1 = None
        view_335: "f32[768]" = torch.ops.aten.reshape.default(mul_857, [-1]);  mul_857 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_111: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_335, -1);  view_335 = None
        mul_859: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_858, unsqueeze_111);  mul_858 = unsqueeze_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_336: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_859, [768, 128, 3, 3]);  mul_859 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_157: "f32[8, 768, 8, 8]" = torch.ops.aten.convolution.default(mul_856, view_336, arg220_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_856 = view_336 = arg220_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_860: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_157, 0.5)
        mul_861: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_157, 0.7071067811865476);  convolution_157 = None
        erf_102: "f32[8, 768, 8, 8]" = torch.ops.aten.erf.default(mul_861);  mul_861 = None
        add_237: "f32[8, 768, 8, 8]" = torch.ops.aten.add.Tensor(erf_102, 1);  erf_102 = None
        mul_862: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_860, add_237);  mul_860 = add_237 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_863: "f32[8, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_862, 1.7015043497085571);  mul_862 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_112: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_337, getitem_225);  view_337 = getitem_225 = None
        add_238: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-05);  getitem_224 = None
        rsqrt_112: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
        mul_865: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_112);  sub_112 = rsqrt_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_864: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg222_1, 0.03608439182435161);  arg222_1 = None
        view_338: "f32[1536]" = torch.ops.aten.reshape.default(mul_864, [-1]);  mul_864 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_112: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_338, -1);  view_338 = None
        mul_866: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_865, unsqueeze_112);  mul_865 = unsqueeze_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_339: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_866, [1536, 768, 1, 1]);  mul_866 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_158: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_863, view_339, arg223_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_863 = view_339 = arg223_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_24: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_158, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_159: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_24, arg224_1, arg225_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_24 = arg224_1 = arg225_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_23: "f32[8, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_159);  convolution_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_160: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_23, arg226_1, arg227_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_23 = arg226_1 = arg227_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_23: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_160);  convolution_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_867: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_158, sigmoid_23);  convolution_158 = sigmoid_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_868: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_867, 2.0);  mul_867 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:198 in forward, code: out.mul_(self.skipinit_gain)
        mul_869: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_868, arg228_1);  mul_868 = arg228_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_870: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_869, 0.2);  mul_869 = None
        add_239: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_870, add_230);  mul_870 = add_230 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        sub_113: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(view_340, getitem_227);  view_340 = getitem_227 = None
        add_240: "f32[1, 3072, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-05);  getitem_226 = None
        rsqrt_113: "f32[1, 3072, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        mul_872: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_113);  sub_113 = rsqrt_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:131 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_871: "f32[3072, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg230_1, 0.02551551815399144);  arg230_1 = None
        view_341: "f32[3072]" = torch.ops.aten.reshape.default(mul_871, [-1]);  mul_871 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:129 in forward, code: weight = F.batch_norm(
        unsqueeze_113: "f32[3072, 1]" = torch.ops.aten.unsqueeze.default(view_341, -1);  view_341 = None
        mul_873: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(mul_872, unsqueeze_113);  mul_872 = unsqueeze_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:132 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_342: "f32[3072, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_873, [3072, 1536, 1, 1]);  mul_873 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:133 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_161: "f32[8, 3072, 8, 8]" = torch.ops.aten.convolution.default(add_239, view_342, arg231_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_239 = view_342 = arg231_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_874: "f32[8, 3072, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_161, 0.5)
        mul_875: "f32[8, 3072, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_161, 0.7071067811865476);  convolution_161 = None
        erf_103: "f32[8, 3072, 8, 8]" = torch.ops.aten.erf.default(mul_875);  mul_875 = None
        add_241: "f32[8, 3072, 8, 8]" = torch.ops.aten.add.Tensor(erf_103, 1);  erf_103 = None
        mul_876: "f32[8, 3072, 8, 8]" = torch.ops.aten.mul.Tensor(mul_874, add_241);  mul_874 = add_241 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:71 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_877: "f32[8, 3072, 8, 8]" = torch.ops.aten.mul.Tensor(mul_876, 1.7015043497085571);  mul_876 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_25: "f32[8, 3072, 1, 1]" = torch.ops.aten.mean.dim(mul_877, [-1, -2], True);  mul_877 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_343: "f32[8, 3072]" = torch.ops.aten.reshape.default(mean_25, [8, 3072]);  mean_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:132 in forward, code: x = self.fc(x)
        permute_1: "f32[3072, 1000]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg233_1, view_343, permute_1);  arg233_1 = view_343 = permute_1 = None
        return (addmm_1,)
        