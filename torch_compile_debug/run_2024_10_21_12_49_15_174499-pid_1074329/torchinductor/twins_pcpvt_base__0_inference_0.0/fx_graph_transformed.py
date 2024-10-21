class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 224, 224]", arg1_1: "f32[64, 3, 4, 4]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64]", arg7_1: "f32[64, 64]", arg8_1: "f32[64]", arg9_1: "f32[64, 64, 8, 8]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "f32[64]", arg13_1: "f32[128, 64]", arg14_1: "f32[128]", arg15_1: "f32[64, 64]", arg16_1: "f32[64]", arg17_1: "f32[64]", arg18_1: "f32[64]", arg19_1: "f32[512, 64]", arg20_1: "f32[512]", arg21_1: "f32[64, 512]", arg22_1: "f32[64]", arg23_1: "f32[64, 1, 3, 3]", arg24_1: "f32[64]", arg25_1: "f32[64]", arg26_1: "f32[64]", arg27_1: "f32[64, 64]", arg28_1: "f32[64]", arg29_1: "f32[64, 64, 8, 8]", arg30_1: "f32[64]", arg31_1: "f32[64]", arg32_1: "f32[64]", arg33_1: "f32[128, 64]", arg34_1: "f32[128]", arg35_1: "f32[64, 64]", arg36_1: "f32[64]", arg37_1: "f32[64]", arg38_1: "f32[64]", arg39_1: "f32[512, 64]", arg40_1: "f32[512]", arg41_1: "f32[64, 512]", arg42_1: "f32[64]", arg43_1: "f32[64]", arg44_1: "f32[64]", arg45_1: "f32[64, 64]", arg46_1: "f32[64]", arg47_1: "f32[64, 64, 8, 8]", arg48_1: "f32[64]", arg49_1: "f32[64]", arg50_1: "f32[64]", arg51_1: "f32[128, 64]", arg52_1: "f32[128]", arg53_1: "f32[64, 64]", arg54_1: "f32[64]", arg55_1: "f32[64]", arg56_1: "f32[64]", arg57_1: "f32[512, 64]", arg58_1: "f32[512]", arg59_1: "f32[64, 512]", arg60_1: "f32[64]", arg61_1: "f32[128, 64, 2, 2]", arg62_1: "f32[128]", arg63_1: "f32[128]", arg64_1: "f32[128]", arg65_1: "f32[128]", arg66_1: "f32[128]", arg67_1: "f32[128, 128]", arg68_1: "f32[128]", arg69_1: "f32[128, 128, 4, 4]", arg70_1: "f32[128]", arg71_1: "f32[128]", arg72_1: "f32[128]", arg73_1: "f32[256, 128]", arg74_1: "f32[256]", arg75_1: "f32[128, 128]", arg76_1: "f32[128]", arg77_1: "f32[128]", arg78_1: "f32[128]", arg79_1: "f32[1024, 128]", arg80_1: "f32[1024]", arg81_1: "f32[128, 1024]", arg82_1: "f32[128]", arg83_1: "f32[128, 1, 3, 3]", arg84_1: "f32[128]", arg85_1: "f32[128]", arg86_1: "f32[128]", arg87_1: "f32[128, 128]", arg88_1: "f32[128]", arg89_1: "f32[128, 128, 4, 4]", arg90_1: "f32[128]", arg91_1: "f32[128]", arg92_1: "f32[128]", arg93_1: "f32[256, 128]", arg94_1: "f32[256]", arg95_1: "f32[128, 128]", arg96_1: "f32[128]", arg97_1: "f32[128]", arg98_1: "f32[128]", arg99_1: "f32[1024, 128]", arg100_1: "f32[1024]", arg101_1: "f32[128, 1024]", arg102_1: "f32[128]", arg103_1: "f32[128]", arg104_1: "f32[128]", arg105_1: "f32[128, 128]", arg106_1: "f32[128]", arg107_1: "f32[128, 128, 4, 4]", arg108_1: "f32[128]", arg109_1: "f32[128]", arg110_1: "f32[128]", arg111_1: "f32[256, 128]", arg112_1: "f32[256]", arg113_1: "f32[128, 128]", arg114_1: "f32[128]", arg115_1: "f32[128]", arg116_1: "f32[128]", arg117_1: "f32[1024, 128]", arg118_1: "f32[1024]", arg119_1: "f32[128, 1024]", arg120_1: "f32[128]", arg121_1: "f32[128]", arg122_1: "f32[128]", arg123_1: "f32[128, 128]", arg124_1: "f32[128]", arg125_1: "f32[128, 128, 4, 4]", arg126_1: "f32[128]", arg127_1: "f32[128]", arg128_1: "f32[128]", arg129_1: "f32[256, 128]", arg130_1: "f32[256]", arg131_1: "f32[128, 128]", arg132_1: "f32[128]", arg133_1: "f32[128]", arg134_1: "f32[128]", arg135_1: "f32[1024, 128]", arg136_1: "f32[1024]", arg137_1: "f32[128, 1024]", arg138_1: "f32[128]", arg139_1: "f32[320, 128, 2, 2]", arg140_1: "f32[320]", arg141_1: "f32[320]", arg142_1: "f32[320]", arg143_1: "f32[320]", arg144_1: "f32[320]", arg145_1: "f32[320, 320]", arg146_1: "f32[320]", arg147_1: "f32[320, 320, 2, 2]", arg148_1: "f32[320]", arg149_1: "f32[320]", arg150_1: "f32[320]", arg151_1: "f32[640, 320]", arg152_1: "f32[640]", arg153_1: "f32[320, 320]", arg154_1: "f32[320]", arg155_1: "f32[320]", arg156_1: "f32[320]", arg157_1: "f32[1280, 320]", arg158_1: "f32[1280]", arg159_1: "f32[320, 1280]", arg160_1: "f32[320]", arg161_1: "f32[320, 1, 3, 3]", arg162_1: "f32[320]", arg163_1: "f32[320]", arg164_1: "f32[320]", arg165_1: "f32[320, 320]", arg166_1: "f32[320]", arg167_1: "f32[320, 320, 2, 2]", arg168_1: "f32[320]", arg169_1: "f32[320]", arg170_1: "f32[320]", arg171_1: "f32[640, 320]", arg172_1: "f32[640]", arg173_1: "f32[320, 320]", arg174_1: "f32[320]", arg175_1: "f32[320]", arg176_1: "f32[320]", arg177_1: "f32[1280, 320]", arg178_1: "f32[1280]", arg179_1: "f32[320, 1280]", arg180_1: "f32[320]", arg181_1: "f32[320]", arg182_1: "f32[320]", arg183_1: "f32[320, 320]", arg184_1: "f32[320]", arg185_1: "f32[320, 320, 2, 2]", arg186_1: "f32[320]", arg187_1: "f32[320]", arg188_1: "f32[320]", arg189_1: "f32[640, 320]", arg190_1: "f32[640]", arg191_1: "f32[320, 320]", arg192_1: "f32[320]", arg193_1: "f32[320]", arg194_1: "f32[320]", arg195_1: "f32[1280, 320]", arg196_1: "f32[1280]", arg197_1: "f32[320, 1280]", arg198_1: "f32[320]", arg199_1: "f32[320]", arg200_1: "f32[320]", arg201_1: "f32[320, 320]", arg202_1: "f32[320]", arg203_1: "f32[320, 320, 2, 2]", arg204_1: "f32[320]", arg205_1: "f32[320]", arg206_1: "f32[320]", arg207_1: "f32[640, 320]", arg208_1: "f32[640]", arg209_1: "f32[320, 320]", arg210_1: "f32[320]", arg211_1: "f32[320]", arg212_1: "f32[320]", arg213_1: "f32[1280, 320]", arg214_1: "f32[1280]", arg215_1: "f32[320, 1280]", arg216_1: "f32[320]", arg217_1: "f32[320]", arg218_1: "f32[320]", arg219_1: "f32[320, 320]", arg220_1: "f32[320]", arg221_1: "f32[320, 320, 2, 2]", arg222_1: "f32[320]", arg223_1: "f32[320]", arg224_1: "f32[320]", arg225_1: "f32[640, 320]", arg226_1: "f32[640]", arg227_1: "f32[320, 320]", arg228_1: "f32[320]", arg229_1: "f32[320]", arg230_1: "f32[320]", arg231_1: "f32[1280, 320]", arg232_1: "f32[1280]", arg233_1: "f32[320, 1280]", arg234_1: "f32[320]", arg235_1: "f32[320]", arg236_1: "f32[320]", arg237_1: "f32[320, 320]", arg238_1: "f32[320]", arg239_1: "f32[320, 320, 2, 2]", arg240_1: "f32[320]", arg241_1: "f32[320]", arg242_1: "f32[320]", arg243_1: "f32[640, 320]", arg244_1: "f32[640]", arg245_1: "f32[320, 320]", arg246_1: "f32[320]", arg247_1: "f32[320]", arg248_1: "f32[320]", arg249_1: "f32[1280, 320]", arg250_1: "f32[1280]", arg251_1: "f32[320, 1280]", arg252_1: "f32[320]", arg253_1: "f32[320]", arg254_1: "f32[320]", arg255_1: "f32[320, 320]", arg256_1: "f32[320]", arg257_1: "f32[320, 320, 2, 2]", arg258_1: "f32[320]", arg259_1: "f32[320]", arg260_1: "f32[320]", arg261_1: "f32[640, 320]", arg262_1: "f32[640]", arg263_1: "f32[320, 320]", arg264_1: "f32[320]", arg265_1: "f32[320]", arg266_1: "f32[320]", arg267_1: "f32[1280, 320]", arg268_1: "f32[1280]", arg269_1: "f32[320, 1280]", arg270_1: "f32[320]", arg271_1: "f32[320]", arg272_1: "f32[320]", arg273_1: "f32[320, 320]", arg274_1: "f32[320]", arg275_1: "f32[320, 320, 2, 2]", arg276_1: "f32[320]", arg277_1: "f32[320]", arg278_1: "f32[320]", arg279_1: "f32[640, 320]", arg280_1: "f32[640]", arg281_1: "f32[320, 320]", arg282_1: "f32[320]", arg283_1: "f32[320]", arg284_1: "f32[320]", arg285_1: "f32[1280, 320]", arg286_1: "f32[1280]", arg287_1: "f32[320, 1280]", arg288_1: "f32[320]", arg289_1: "f32[320]", arg290_1: "f32[320]", arg291_1: "f32[320, 320]", arg292_1: "f32[320]", arg293_1: "f32[320, 320, 2, 2]", arg294_1: "f32[320]", arg295_1: "f32[320]", arg296_1: "f32[320]", arg297_1: "f32[640, 320]", arg298_1: "f32[640]", arg299_1: "f32[320, 320]", arg300_1: "f32[320]", arg301_1: "f32[320]", arg302_1: "f32[320]", arg303_1: "f32[1280, 320]", arg304_1: "f32[1280]", arg305_1: "f32[320, 1280]", arg306_1: "f32[320]", arg307_1: "f32[320]", arg308_1: "f32[320]", arg309_1: "f32[320, 320]", arg310_1: "f32[320]", arg311_1: "f32[320, 320, 2, 2]", arg312_1: "f32[320]", arg313_1: "f32[320]", arg314_1: "f32[320]", arg315_1: "f32[640, 320]", arg316_1: "f32[640]", arg317_1: "f32[320, 320]", arg318_1: "f32[320]", arg319_1: "f32[320]", arg320_1: "f32[320]", arg321_1: "f32[1280, 320]", arg322_1: "f32[1280]", arg323_1: "f32[320, 1280]", arg324_1: "f32[320]", arg325_1: "f32[320]", arg326_1: "f32[320]", arg327_1: "f32[320, 320]", arg328_1: "f32[320]", arg329_1: "f32[320, 320, 2, 2]", arg330_1: "f32[320]", arg331_1: "f32[320]", arg332_1: "f32[320]", arg333_1: "f32[640, 320]", arg334_1: "f32[640]", arg335_1: "f32[320, 320]", arg336_1: "f32[320]", arg337_1: "f32[320]", arg338_1: "f32[320]", arg339_1: "f32[1280, 320]", arg340_1: "f32[1280]", arg341_1: "f32[320, 1280]", arg342_1: "f32[320]", arg343_1: "f32[320]", arg344_1: "f32[320]", arg345_1: "f32[320, 320]", arg346_1: "f32[320]", arg347_1: "f32[320, 320, 2, 2]", arg348_1: "f32[320]", arg349_1: "f32[320]", arg350_1: "f32[320]", arg351_1: "f32[640, 320]", arg352_1: "f32[640]", arg353_1: "f32[320, 320]", arg354_1: "f32[320]", arg355_1: "f32[320]", arg356_1: "f32[320]", arg357_1: "f32[1280, 320]", arg358_1: "f32[1280]", arg359_1: "f32[320, 1280]", arg360_1: "f32[320]", arg361_1: "f32[320]", arg362_1: "f32[320]", arg363_1: "f32[320, 320]", arg364_1: "f32[320]", arg365_1: "f32[320, 320, 2, 2]", arg366_1: "f32[320]", arg367_1: "f32[320]", arg368_1: "f32[320]", arg369_1: "f32[640, 320]", arg370_1: "f32[640]", arg371_1: "f32[320, 320]", arg372_1: "f32[320]", arg373_1: "f32[320]", arg374_1: "f32[320]", arg375_1: "f32[1280, 320]", arg376_1: "f32[1280]", arg377_1: "f32[320, 1280]", arg378_1: "f32[320]", arg379_1: "f32[320]", arg380_1: "f32[320]", arg381_1: "f32[320, 320]", arg382_1: "f32[320]", arg383_1: "f32[320, 320, 2, 2]", arg384_1: "f32[320]", arg385_1: "f32[320]", arg386_1: "f32[320]", arg387_1: "f32[640, 320]", arg388_1: "f32[640]", arg389_1: "f32[320, 320]", arg390_1: "f32[320]", arg391_1: "f32[320]", arg392_1: "f32[320]", arg393_1: "f32[1280, 320]", arg394_1: "f32[1280]", arg395_1: "f32[320, 1280]", arg396_1: "f32[320]", arg397_1: "f32[320]", arg398_1: "f32[320]", arg399_1: "f32[320, 320]", arg400_1: "f32[320]", arg401_1: "f32[320, 320, 2, 2]", arg402_1: "f32[320]", arg403_1: "f32[320]", arg404_1: "f32[320]", arg405_1: "f32[640, 320]", arg406_1: "f32[640]", arg407_1: "f32[320, 320]", arg408_1: "f32[320]", arg409_1: "f32[320]", arg410_1: "f32[320]", arg411_1: "f32[1280, 320]", arg412_1: "f32[1280]", arg413_1: "f32[320, 1280]", arg414_1: "f32[320]", arg415_1: "f32[320]", arg416_1: "f32[320]", arg417_1: "f32[320, 320]", arg418_1: "f32[320]", arg419_1: "f32[320, 320, 2, 2]", arg420_1: "f32[320]", arg421_1: "f32[320]", arg422_1: "f32[320]", arg423_1: "f32[640, 320]", arg424_1: "f32[640]", arg425_1: "f32[320, 320]", arg426_1: "f32[320]", arg427_1: "f32[320]", arg428_1: "f32[320]", arg429_1: "f32[1280, 320]", arg430_1: "f32[1280]", arg431_1: "f32[320, 1280]", arg432_1: "f32[320]", arg433_1: "f32[320]", arg434_1: "f32[320]", arg435_1: "f32[320, 320]", arg436_1: "f32[320]", arg437_1: "f32[320, 320, 2, 2]", arg438_1: "f32[320]", arg439_1: "f32[320]", arg440_1: "f32[320]", arg441_1: "f32[640, 320]", arg442_1: "f32[640]", arg443_1: "f32[320, 320]", arg444_1: "f32[320]", arg445_1: "f32[320]", arg446_1: "f32[320]", arg447_1: "f32[1280, 320]", arg448_1: "f32[1280]", arg449_1: "f32[320, 1280]", arg450_1: "f32[320]", arg451_1: "f32[320]", arg452_1: "f32[320]", arg453_1: "f32[320, 320]", arg454_1: "f32[320]", arg455_1: "f32[320, 320, 2, 2]", arg456_1: "f32[320]", arg457_1: "f32[320]", arg458_1: "f32[320]", arg459_1: "f32[640, 320]", arg460_1: "f32[640]", arg461_1: "f32[320, 320]", arg462_1: "f32[320]", arg463_1: "f32[320]", arg464_1: "f32[320]", arg465_1: "f32[1280, 320]", arg466_1: "f32[1280]", arg467_1: "f32[320, 1280]", arg468_1: "f32[320]", arg469_1: "f32[512, 320, 2, 2]", arg470_1: "f32[512]", arg471_1: "f32[512]", arg472_1: "f32[512]", arg473_1: "f32[512]", arg474_1: "f32[512]", arg475_1: "f32[512, 512]", arg476_1: "f32[512]", arg477_1: "f32[1024, 512]", arg478_1: "f32[1024]", arg479_1: "f32[512, 512]", arg480_1: "f32[512]", arg481_1: "f32[512]", arg482_1: "f32[512]", arg483_1: "f32[2048, 512]", arg484_1: "f32[2048]", arg485_1: "f32[512, 2048]", arg486_1: "f32[512]", arg487_1: "f32[512, 1, 3, 3]", arg488_1: "f32[512]", arg489_1: "f32[512]", arg490_1: "f32[512]", arg491_1: "f32[512, 512]", arg492_1: "f32[512]", arg493_1: "f32[1024, 512]", arg494_1: "f32[1024]", arg495_1: "f32[512, 512]", arg496_1: "f32[512]", arg497_1: "f32[512]", arg498_1: "f32[512]", arg499_1: "f32[2048, 512]", arg500_1: "f32[2048]", arg501_1: "f32[512, 2048]", arg502_1: "f32[512]", arg503_1: "f32[512]", arg504_1: "f32[512]", arg505_1: "f32[512, 512]", arg506_1: "f32[512]", arg507_1: "f32[1024, 512]", arg508_1: "f32[1024]", arg509_1: "f32[512, 512]", arg510_1: "f32[512]", arg511_1: "f32[512]", arg512_1: "f32[512]", arg513_1: "f32[2048, 512]", arg514_1: "f32[2048]", arg515_1: "f32[512, 2048]", arg516_1: "f32[512]", arg517_1: "f32[512]", arg518_1: "f32[512]", arg519_1: "f32[1000, 512]", arg520_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:275 in forward, code: x = self.proj(x).flatten(2).transpose(1, 2)
        convolution_33: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        view_433: "f32[8, 64, 3136]" = torch.ops.aten.reshape.default(convolution_33, [8, 64, 3136]);  convolution_33 = None
        permute_294: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_433, [0, 2, 1]);  view_433 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:276 in forward, code: x = self.norm(x)
        clone_96: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
        var_mean_86 = torch.ops.aten.var_mean.correction(clone_96, [2], correction = 0, keepdim = True)
        getitem_340: "f32[8, 3136, 1]" = var_mean_86[0]
        getitem_341: "f32[8, 3136, 1]" = var_mean_86[1];  var_mean_86 = None
        sub_86: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_96, getitem_341);  clone_96 = getitem_341 = None
        add_260: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_340, 1e-05);  getitem_340 = None
        rsqrt_86: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
        mul_256: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = rsqrt_86 = None
        mul_257: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_256, arg3_1);  mul_256 = arg3_1 = None
        add_261: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_257, arg4_1);  mul_257 = arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_87 = torch.ops.aten.var_mean.correction(add_261, [2], correction = 0, keepdim = True)
        getitem_342: "f32[8, 3136, 1]" = var_mean_87[0]
        getitem_343: "f32[8, 3136, 1]" = var_mean_87[1];  var_mean_87 = None
        sub_87: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(add_261, getitem_343);  getitem_343 = None
        add_262: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_342, 1e-06);  getitem_342 = None
        rsqrt_87: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
        mul_258: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = rsqrt_87 = None
        mul_259: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_258, arg5_1);  mul_258 = arg5_1 = None
        add_263: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_259, arg6_1);  mul_259 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_297: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(add_263, [0, 2, 1])
        view_437: "f32[8, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_297, [8, 64, 56, 56]);  permute_297 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_34: "f32[8, 64, 7, 7]" = torch.ops.aten.convolution.default(view_437, arg9_1, arg10_1, [8, 8], [0, 0], [1, 1], False, [0, 0], 1);  view_437 = arg9_1 = arg10_1 = None
        view_438: "f32[8, 64, 49]" = torch.ops.aten.reshape.default(convolution_34, [8, 64, 49]);  convolution_34 = None
        permute_298: "f32[8, 49, 64]" = torch.ops.aten.permute.default(view_438, [0, 2, 1]);  view_438 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_88 = torch.ops.aten.var_mean.correction(permute_298, [2], correction = 0, keepdim = True)
        getitem_344: "f32[8, 49, 1]" = var_mean_88[0]
        getitem_345: "f32[8, 49, 1]" = var_mean_88[1];  var_mean_88 = None
        sub_88: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(permute_298, getitem_345);  permute_298 = getitem_345 = None
        add_264: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_344, 1e-05);  getitem_344 = None
        rsqrt_88: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
        mul_260: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = rsqrt_88 = None
        mul_261: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_260, arg11_1);  mul_260 = arg11_1 = None
        add_265: "f32[8, 49, 64]" = torch.ops.aten.add.Tensor(mul_261, arg12_1);  mul_261 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_439: "f32[392, 64]" = torch.ops.aten.reshape.default(add_265, [392, 64]);  add_265 = None
        permute_299: "f32[64, 128]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_142: "f32[392, 128]" = torch.ops.aten.addmm.default(arg14_1, view_439, permute_299);  arg14_1 = view_439 = permute_299 = None
        view_440: "f32[8, 49, 128]" = torch.ops.aten.reshape.default(addmm_142, [8, 49, 128]);  addmm_142 = None
        view_441: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.reshape.default(view_440, [8, -1, 2, 1, 64]);  view_440 = None
        permute_300: "f32[2, 8, 1, 49, 64]" = torch.ops.aten.permute.default(view_441, [2, 0, 3, 1, 4]);  view_441 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_28 = torch.ops.aten.unbind.int(permute_300);  permute_300 = None
        getitem_346: "f32[8, 1, 49, 64]" = unbind_28[0]
        getitem_347: "f32[8, 1, 49, 64]" = unbind_28[1];  unbind_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_434: "f32[25088, 64]" = torch.ops.aten.reshape.default(add_263, [25088, 64]);  add_263 = None
        permute_295: "f32[64, 64]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_141: "f32[25088, 64]" = torch.ops.aten.addmm.default(arg8_1, view_434, permute_295);  arg8_1 = view_434 = permute_295 = None
        view_435: "f32[8, 3136, 64]" = torch.ops.aten.reshape.default(addmm_141, [8, 3136, 64]);  addmm_141 = None
        view_436: "f32[8, 3136, 1, 64]" = torch.ops.aten.reshape.default(view_435, [8, 3136, 1, 64]);  view_435 = None
        permute_296: "f32[8, 1, 3136, 64]" = torch.ops.aten.permute.default(view_436, [0, 2, 1, 3]);  view_436 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_28 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_296, getitem_346, getitem_347, None, False);  permute_296 = getitem_346 = getitem_347 = None
        getitem_348: "f32[8, 1, 3136, 64]" = _scaled_dot_product_efficient_attention_28[0];  _scaled_dot_product_efficient_attention_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_301: "f32[8, 3136, 1, 64]" = torch.ops.aten.permute.default(getitem_348, [0, 2, 1, 3]);  getitem_348 = None
        view_442: "f32[8, 3136, 64]" = torch.ops.aten.reshape.default(permute_301, [8, 3136, 64]);  permute_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_443: "f32[25088, 64]" = torch.ops.aten.reshape.default(view_442, [25088, 64]);  view_442 = None
        permute_302: "f32[64, 64]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        
        # No stacktrace found for following nodes
        mm_default_83: "f32[25088, 64]" = torch.ops.aten.mm.default(view_443, permute_302);  view_443 = permute_302 = None
        add_tensor_83: "f32[25088, 64]" = torch.ops.aten.add.Tensor(mm_default_83, arg16_1);  mm_default_83 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_444: "f32[8, 3136, 64]" = torch.ops.aten.reshape.default(add_tensor_83, [8, 3136, 64]);  add_tensor_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_266: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_261, view_444);  add_261 = view_444 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_89 = torch.ops.aten.var_mean.correction(add_266, [2], correction = 0, keepdim = True)
        getitem_352: "f32[8, 3136, 1]" = var_mean_89[0]
        getitem_353: "f32[8, 3136, 1]" = var_mean_89[1];  var_mean_89 = None
        sub_89: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(add_266, getitem_353);  getitem_353 = None
        add_267: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_352, 1e-06);  getitem_352 = None
        rsqrt_89: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
        mul_262: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = rsqrt_89 = None
        mul_263: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_262, arg17_1);  mul_262 = arg17_1 = None
        add_268: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_263, arg18_1);  mul_263 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_445: "f32[25088, 64]" = torch.ops.aten.reshape.default(add_268, [25088, 64]);  add_268 = None
        permute_303: "f32[64, 512]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        
        # No stacktrace found for following nodes
        mm_default_82: "f32[25088, 512]" = torch.ops.aten.mm.default(view_445, permute_303);  view_445 = permute_303 = None
        add_tensor_82: "f32[25088, 512]" = torch.ops.aten.add.Tensor(mm_default_82, arg20_1);  mm_default_82 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_446: "f32[8, 3136, 512]" = torch.ops.aten.reshape.default(add_tensor_82, [8, 3136, 512]);  add_tensor_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_264: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_446, 0.5)
        mul_265: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_446, 0.7071067811865476);  view_446 = None
        erf_28: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_265);  mul_265 = None
        add_269: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_266: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_264, add_269);  mul_264 = add_269 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_447: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_266, [25088, 512]);  mul_266 = None
        permute_304: "f32[512, 64]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        
        # No stacktrace found for following nodes
        mm_default_81: "f32[25088, 64]" = torch.ops.aten.mm.default(view_447, permute_304);  view_447 = permute_304 = None
        add_tensor_81: "f32[25088, 64]" = torch.ops.aten.add.Tensor(mm_default_81, arg22_1);  mm_default_81 = arg22_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_448: "f32[8, 3136, 64]" = torch.ops.aten.reshape.default(add_tensor_81, [8, 3136, 64]);  add_tensor_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_270: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_266, view_448);  add_266 = view_448 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:243 in forward, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
        permute_305: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(add_270, [0, 2, 1]);  add_270 = None
        view_449: "f32[8, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_305, [8, 64, 56, 56]);  permute_305 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:244 in forward, code: x = self.proj(cnn_feat_token)
        convolution_35: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(view_449, arg23_1, arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  arg23_1 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:246 in forward, code: x += cnn_feat_token
        add_271: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(convolution_35, view_449);  convolution_35 = view_449 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        view_451: "f32[8, 64, 3136]" = torch.ops.aten.reshape.default(add_271, [8, 64, 3136]);  add_271 = None
        permute_307: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None
        var_mean_90 = torch.ops.aten.var_mean.correction(permute_307, [2], correction = 0, keepdim = True)
        getitem_354: "f32[8, 3136, 1]" = var_mean_90[0]
        getitem_355: "f32[8, 3136, 1]" = var_mean_90[1];  var_mean_90 = None
        sub_90: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(permute_307, getitem_355);  getitem_355 = None
        add_272: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_354, 1e-06);  getitem_354 = None
        rsqrt_90: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
        mul_267: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = rsqrt_90 = None
        mul_268: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_267, arg25_1);  mul_267 = arg25_1 = None
        add_273: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_268, arg26_1);  mul_268 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_310: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(add_273, [0, 2, 1])
        view_455: "f32[8, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_310, [8, 64, 56, 56]);  permute_310 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_36: "f32[8, 64, 7, 7]" = torch.ops.aten.convolution.default(view_455, arg29_1, arg30_1, [8, 8], [0, 0], [1, 1], False, [0, 0], 1);  view_455 = arg29_1 = arg30_1 = None
        view_456: "f32[8, 64, 49]" = torch.ops.aten.reshape.default(convolution_36, [8, 64, 49]);  convolution_36 = None
        permute_311: "f32[8, 49, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1]);  view_456 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_91 = torch.ops.aten.var_mean.correction(permute_311, [2], correction = 0, keepdim = True)
        getitem_356: "f32[8, 49, 1]" = var_mean_91[0]
        getitem_357: "f32[8, 49, 1]" = var_mean_91[1];  var_mean_91 = None
        sub_91: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(permute_311, getitem_357);  permute_311 = getitem_357 = None
        add_274: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_356, 1e-05);  getitem_356 = None
        rsqrt_91: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_274);  add_274 = None
        mul_269: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = rsqrt_91 = None
        mul_270: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_269, arg31_1);  mul_269 = arg31_1 = None
        add_275: "f32[8, 49, 64]" = torch.ops.aten.add.Tensor(mul_270, arg32_1);  mul_270 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_457: "f32[392, 64]" = torch.ops.aten.reshape.default(add_275, [392, 64]);  add_275 = None
        permute_312: "f32[64, 128]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_147: "f32[392, 128]" = torch.ops.aten.addmm.default(arg34_1, view_457, permute_312);  arg34_1 = view_457 = permute_312 = None
        view_458: "f32[8, 49, 128]" = torch.ops.aten.reshape.default(addmm_147, [8, 49, 128]);  addmm_147 = None
        view_459: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.reshape.default(view_458, [8, -1, 2, 1, 64]);  view_458 = None
        permute_313: "f32[2, 8, 1, 49, 64]" = torch.ops.aten.permute.default(view_459, [2, 0, 3, 1, 4]);  view_459 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_29 = torch.ops.aten.unbind.int(permute_313);  permute_313 = None
        getitem_358: "f32[8, 1, 49, 64]" = unbind_29[0]
        getitem_359: "f32[8, 1, 49, 64]" = unbind_29[1];  unbind_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_452: "f32[25088, 64]" = torch.ops.aten.reshape.default(add_273, [25088, 64]);  add_273 = None
        permute_308: "f32[64, 64]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_146: "f32[25088, 64]" = torch.ops.aten.addmm.default(arg28_1, view_452, permute_308);  arg28_1 = view_452 = permute_308 = None
        view_453: "f32[8, 3136, 64]" = torch.ops.aten.reshape.default(addmm_146, [8, 3136, 64]);  addmm_146 = None
        view_454: "f32[8, 3136, 1, 64]" = torch.ops.aten.reshape.default(view_453, [8, 3136, 1, 64]);  view_453 = None
        permute_309: "f32[8, 1, 3136, 64]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_29 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_309, getitem_358, getitem_359, None, False);  permute_309 = getitem_358 = getitem_359 = None
        getitem_360: "f32[8, 1, 3136, 64]" = _scaled_dot_product_efficient_attention_29[0];  _scaled_dot_product_efficient_attention_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_314: "f32[8, 3136, 1, 64]" = torch.ops.aten.permute.default(getitem_360, [0, 2, 1, 3]);  getitem_360 = None
        view_460: "f32[8, 3136, 64]" = torch.ops.aten.reshape.default(permute_314, [8, 3136, 64]);  permute_314 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_461: "f32[25088, 64]" = torch.ops.aten.reshape.default(view_460, [25088, 64]);  view_460 = None
        permute_315: "f32[64, 64]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        
        # No stacktrace found for following nodes
        mm_default_80: "f32[25088, 64]" = torch.ops.aten.mm.default(view_461, permute_315);  view_461 = permute_315 = None
        add_tensor_80: "f32[25088, 64]" = torch.ops.aten.add.Tensor(mm_default_80, arg36_1);  mm_default_80 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_462: "f32[8, 3136, 64]" = torch.ops.aten.reshape.default(add_tensor_80, [8, 3136, 64]);  add_tensor_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_276: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(permute_307, view_462);  permute_307 = view_462 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_92 = torch.ops.aten.var_mean.correction(add_276, [2], correction = 0, keepdim = True)
        getitem_364: "f32[8, 3136, 1]" = var_mean_92[0]
        getitem_365: "f32[8, 3136, 1]" = var_mean_92[1];  var_mean_92 = None
        sub_92: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(add_276, getitem_365);  getitem_365 = None
        add_277: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_364, 1e-06);  getitem_364 = None
        rsqrt_92: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_277);  add_277 = None
        mul_271: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = rsqrt_92 = None
        mul_272: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_271, arg37_1);  mul_271 = arg37_1 = None
        add_278: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_272, arg38_1);  mul_272 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_463: "f32[25088, 64]" = torch.ops.aten.reshape.default(add_278, [25088, 64]);  add_278 = None
        permute_316: "f32[64, 512]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        
        # No stacktrace found for following nodes
        mm_default_79: "f32[25088, 512]" = torch.ops.aten.mm.default(view_463, permute_316);  view_463 = permute_316 = None
        add_tensor_79: "f32[25088, 512]" = torch.ops.aten.add.Tensor(mm_default_79, arg40_1);  mm_default_79 = arg40_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_464: "f32[8, 3136, 512]" = torch.ops.aten.reshape.default(add_tensor_79, [8, 3136, 512]);  add_tensor_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_273: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_464, 0.5)
        mul_274: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_464, 0.7071067811865476);  view_464 = None
        erf_29: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_274);  mul_274 = None
        add_279: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_275: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_273, add_279);  mul_273 = add_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_465: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_275, [25088, 512]);  mul_275 = None
        permute_317: "f32[512, 64]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        
        # No stacktrace found for following nodes
        mm_default_78: "f32[25088, 64]" = torch.ops.aten.mm.default(view_465, permute_317);  view_465 = permute_317 = None
        add_tensor_78: "f32[25088, 64]" = torch.ops.aten.add.Tensor(mm_default_78, arg42_1);  mm_default_78 = arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_466: "f32[8, 3136, 64]" = torch.ops.aten.reshape.default(add_tensor_78, [8, 3136, 64]);  add_tensor_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_280: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_276, view_466);  add_276 = view_466 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_93 = torch.ops.aten.var_mean.correction(add_280, [2], correction = 0, keepdim = True)
        getitem_366: "f32[8, 3136, 1]" = var_mean_93[0]
        getitem_367: "f32[8, 3136, 1]" = var_mean_93[1];  var_mean_93 = None
        sub_93: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(add_280, getitem_367);  getitem_367 = None
        add_281: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_366, 1e-06);  getitem_366 = None
        rsqrt_93: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_281);  add_281 = None
        mul_276: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = rsqrt_93 = None
        mul_277: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_276, arg43_1);  mul_276 = arg43_1 = None
        add_282: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_277, arg44_1);  mul_277 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_320: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(add_282, [0, 2, 1])
        view_470: "f32[8, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_320, [8, 64, 56, 56]);  permute_320 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_37: "f32[8, 64, 7, 7]" = torch.ops.aten.convolution.default(view_470, arg47_1, arg48_1, [8, 8], [0, 0], [1, 1], False, [0, 0], 1);  view_470 = arg47_1 = arg48_1 = None
        view_471: "f32[8, 64, 49]" = torch.ops.aten.reshape.default(convolution_37, [8, 64, 49]);  convolution_37 = None
        permute_321: "f32[8, 49, 64]" = torch.ops.aten.permute.default(view_471, [0, 2, 1]);  view_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_94 = torch.ops.aten.var_mean.correction(permute_321, [2], correction = 0, keepdim = True)
        getitem_368: "f32[8, 49, 1]" = var_mean_94[0]
        getitem_369: "f32[8, 49, 1]" = var_mean_94[1];  var_mean_94 = None
        sub_94: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(permute_321, getitem_369);  permute_321 = getitem_369 = None
        add_283: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_368, 1e-05);  getitem_368 = None
        rsqrt_94: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
        mul_278: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = rsqrt_94 = None
        mul_279: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_278, arg49_1);  mul_278 = arg49_1 = None
        add_284: "f32[8, 49, 64]" = torch.ops.aten.add.Tensor(mul_279, arg50_1);  mul_279 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_472: "f32[392, 64]" = torch.ops.aten.reshape.default(add_284, [392, 64]);  add_284 = None
        permute_322: "f32[64, 128]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_152: "f32[392, 128]" = torch.ops.aten.addmm.default(arg52_1, view_472, permute_322);  arg52_1 = view_472 = permute_322 = None
        view_473: "f32[8, 49, 128]" = torch.ops.aten.reshape.default(addmm_152, [8, 49, 128]);  addmm_152 = None
        view_474: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.reshape.default(view_473, [8, -1, 2, 1, 64]);  view_473 = None
        permute_323: "f32[2, 8, 1, 49, 64]" = torch.ops.aten.permute.default(view_474, [2, 0, 3, 1, 4]);  view_474 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_30 = torch.ops.aten.unbind.int(permute_323);  permute_323 = None
        getitem_370: "f32[8, 1, 49, 64]" = unbind_30[0]
        getitem_371: "f32[8, 1, 49, 64]" = unbind_30[1];  unbind_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_467: "f32[25088, 64]" = torch.ops.aten.reshape.default(add_282, [25088, 64]);  add_282 = None
        permute_318: "f32[64, 64]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_151: "f32[25088, 64]" = torch.ops.aten.addmm.default(arg46_1, view_467, permute_318);  arg46_1 = view_467 = permute_318 = None
        view_468: "f32[8, 3136, 64]" = torch.ops.aten.reshape.default(addmm_151, [8, 3136, 64]);  addmm_151 = None
        view_469: "f32[8, 3136, 1, 64]" = torch.ops.aten.reshape.default(view_468, [8, 3136, 1, 64]);  view_468 = None
        permute_319: "f32[8, 1, 3136, 64]" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_30 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_319, getitem_370, getitem_371, None, False);  permute_319 = getitem_370 = getitem_371 = None
        getitem_372: "f32[8, 1, 3136, 64]" = _scaled_dot_product_efficient_attention_30[0];  _scaled_dot_product_efficient_attention_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_324: "f32[8, 3136, 1, 64]" = torch.ops.aten.permute.default(getitem_372, [0, 2, 1, 3]);  getitem_372 = None
        view_475: "f32[8, 3136, 64]" = torch.ops.aten.reshape.default(permute_324, [8, 3136, 64]);  permute_324 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_476: "f32[25088, 64]" = torch.ops.aten.reshape.default(view_475, [25088, 64]);  view_475 = None
        permute_325: "f32[64, 64]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        
        # No stacktrace found for following nodes
        mm_default_77: "f32[25088, 64]" = torch.ops.aten.mm.default(view_476, permute_325);  view_476 = permute_325 = None
        add_tensor_77: "f32[25088, 64]" = torch.ops.aten.add.Tensor(mm_default_77, arg54_1);  mm_default_77 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_477: "f32[8, 3136, 64]" = torch.ops.aten.reshape.default(add_tensor_77, [8, 3136, 64]);  add_tensor_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_285: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_280, view_477);  add_280 = view_477 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_95 = torch.ops.aten.var_mean.correction(add_285, [2], correction = 0, keepdim = True)
        getitem_376: "f32[8, 3136, 1]" = var_mean_95[0]
        getitem_377: "f32[8, 3136, 1]" = var_mean_95[1];  var_mean_95 = None
        sub_95: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(add_285, getitem_377);  getitem_377 = None
        add_286: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_376, 1e-06);  getitem_376 = None
        rsqrt_95: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
        mul_280: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = rsqrt_95 = None
        mul_281: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_280, arg55_1);  mul_280 = arg55_1 = None
        add_287: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_281, arg56_1);  mul_281 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_478: "f32[25088, 64]" = torch.ops.aten.reshape.default(add_287, [25088, 64]);  add_287 = None
        permute_326: "f32[64, 512]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        
        # No stacktrace found for following nodes
        mm_default_76: "f32[25088, 512]" = torch.ops.aten.mm.default(view_478, permute_326);  view_478 = permute_326 = None
        add_tensor_76: "f32[25088, 512]" = torch.ops.aten.add.Tensor(mm_default_76, arg58_1);  mm_default_76 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_479: "f32[8, 3136, 512]" = torch.ops.aten.reshape.default(add_tensor_76, [8, 3136, 512]);  add_tensor_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_282: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_479, 0.5)
        mul_283: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_479, 0.7071067811865476);  view_479 = None
        erf_30: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_283);  mul_283 = None
        add_288: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_284: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_282, add_288);  mul_282 = add_288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_480: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_284, [25088, 512]);  mul_284 = None
        permute_327: "f32[512, 64]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        
        # No stacktrace found for following nodes
        mm_default_75: "f32[25088, 64]" = torch.ops.aten.mm.default(view_480, permute_327);  view_480 = permute_327 = None
        add_tensor_75: "f32[25088, 64]" = torch.ops.aten.add.Tensor(mm_default_75, arg60_1);  mm_default_75 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_481: "f32[8, 3136, 64]" = torch.ops.aten.reshape.default(add_tensor_75, [8, 3136, 64]);  add_tensor_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_289: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_285, view_481);  add_285 = view_481 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:489 in forward_features, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
        view_482: "f32[8, 56, 56, 64]" = torch.ops.aten.reshape.default(add_289, [8, 56, 56, -1]);  add_289 = None
        permute_328: "f32[8, 64, 56, 56]" = torch.ops.aten.permute.default(view_482, [0, 3, 1, 2]);  view_482 = None
        clone_107: "f32[8, 64, 56, 56]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:275 in forward, code: x = self.proj(x).flatten(2).transpose(1, 2)
        convolution_38: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(clone_107, arg61_1, arg62_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_107 = arg61_1 = arg62_1 = None
        view_483: "f32[8, 128, 784]" = torch.ops.aten.reshape.default(convolution_38, [8, 128, 784]);  convolution_38 = None
        permute_329: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_483, [0, 2, 1]);  view_483 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:276 in forward, code: x = self.norm(x)
        clone_108: "f32[8, 784, 128]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
        var_mean_96 = torch.ops.aten.var_mean.correction(clone_108, [2], correction = 0, keepdim = True)
        getitem_378: "f32[8, 784, 1]" = var_mean_96[0]
        getitem_379: "f32[8, 784, 1]" = var_mean_96[1];  var_mean_96 = None
        sub_96: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_108, getitem_379);  clone_108 = getitem_379 = None
        add_290: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_378, 1e-05);  getitem_378 = None
        rsqrt_96: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
        mul_285: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = rsqrt_96 = None
        mul_286: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_285, arg63_1);  mul_285 = arg63_1 = None
        add_291: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_286, arg64_1);  mul_286 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_97 = torch.ops.aten.var_mean.correction(add_291, [2], correction = 0, keepdim = True)
        getitem_380: "f32[8, 784, 1]" = var_mean_97[0]
        getitem_381: "f32[8, 784, 1]" = var_mean_97[1];  var_mean_97 = None
        sub_97: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_291, getitem_381);  getitem_381 = None
        add_292: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_380, 1e-06);  getitem_380 = None
        rsqrt_97: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        mul_287: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = rsqrt_97 = None
        mul_288: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_287, arg65_1);  mul_287 = arg65_1 = None
        add_293: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_288, arg66_1);  mul_288 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_332: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_293, [0, 2, 1])
        view_487: "f32[8, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_332, [8, 128, 28, 28]);  permute_332 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_39: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(view_487, arg69_1, arg70_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  view_487 = arg69_1 = arg70_1 = None
        view_488: "f32[8, 128, 49]" = torch.ops.aten.reshape.default(convolution_39, [8, 128, 49]);  convolution_39 = None
        permute_333: "f32[8, 49, 128]" = torch.ops.aten.permute.default(view_488, [0, 2, 1]);  view_488 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_98 = torch.ops.aten.var_mean.correction(permute_333, [2], correction = 0, keepdim = True)
        getitem_382: "f32[8, 49, 1]" = var_mean_98[0]
        getitem_383: "f32[8, 49, 1]" = var_mean_98[1];  var_mean_98 = None
        sub_98: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(permute_333, getitem_383);  permute_333 = getitem_383 = None
        add_294: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_382, 1e-05);  getitem_382 = None
        rsqrt_98: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
        mul_289: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = rsqrt_98 = None
        mul_290: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_289, arg71_1);  mul_289 = arg71_1 = None
        add_295: "f32[8, 49, 128]" = torch.ops.aten.add.Tensor(mul_290, arg72_1);  mul_290 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_489: "f32[392, 128]" = torch.ops.aten.reshape.default(add_295, [392, 128]);  add_295 = None
        permute_334: "f32[128, 256]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_157: "f32[392, 256]" = torch.ops.aten.addmm.default(arg74_1, view_489, permute_334);  arg74_1 = view_489 = permute_334 = None
        view_490: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(addmm_157, [8, 49, 256]);  addmm_157 = None
        view_491: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.reshape.default(view_490, [8, -1, 2, 2, 64]);  view_490 = None
        permute_335: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.permute.default(view_491, [2, 0, 3, 1, 4]);  view_491 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_31 = torch.ops.aten.unbind.int(permute_335);  permute_335 = None
        getitem_384: "f32[8, 2, 49, 64]" = unbind_31[0]
        getitem_385: "f32[8, 2, 49, 64]" = unbind_31[1];  unbind_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_484: "f32[6272, 128]" = torch.ops.aten.reshape.default(add_293, [6272, 128]);  add_293 = None
        permute_330: "f32[128, 128]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_156: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg68_1, view_484, permute_330);  arg68_1 = view_484 = permute_330 = None
        view_485: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(addmm_156, [8, 784, 128]);  addmm_156 = None
        view_486: "f32[8, 784, 2, 64]" = torch.ops.aten.reshape.default(view_485, [8, 784, 2, 64]);  view_485 = None
        permute_331: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_31 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_331, getitem_384, getitem_385, None, False);  permute_331 = getitem_384 = getitem_385 = None
        getitem_386: "f32[8, 2, 784, 64]" = _scaled_dot_product_efficient_attention_31[0];  _scaled_dot_product_efficient_attention_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_336: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_386, [0, 2, 1, 3]);  getitem_386 = None
        view_492: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(permute_336, [8, 784, 128]);  permute_336 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_493: "f32[6272, 128]" = torch.ops.aten.reshape.default(view_492, [6272, 128]);  view_492 = None
        permute_337: "f32[128, 128]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        
        # No stacktrace found for following nodes
        mm_default_74: "f32[6272, 128]" = torch.ops.aten.mm.default(view_493, permute_337);  view_493 = permute_337 = None
        add_tensor_74: "f32[6272, 128]" = torch.ops.aten.add.Tensor(mm_default_74, arg76_1);  mm_default_74 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_494: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(add_tensor_74, [8, 784, 128]);  add_tensor_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_296: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_291, view_494);  add_291 = view_494 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_99 = torch.ops.aten.var_mean.correction(add_296, [2], correction = 0, keepdim = True)
        getitem_390: "f32[8, 784, 1]" = var_mean_99[0]
        getitem_391: "f32[8, 784, 1]" = var_mean_99[1];  var_mean_99 = None
        sub_99: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_296, getitem_391);  getitem_391 = None
        add_297: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_390, 1e-06);  getitem_390 = None
        rsqrt_99: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_297);  add_297 = None
        mul_291: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = rsqrt_99 = None
        mul_292: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_291, arg77_1);  mul_291 = arg77_1 = None
        add_298: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_292, arg78_1);  mul_292 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_495: "f32[6272, 128]" = torch.ops.aten.reshape.default(add_298, [6272, 128]);  add_298 = None
        permute_338: "f32[128, 1024]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        
        # No stacktrace found for following nodes
        mm_default_73: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_495, permute_338);  view_495 = permute_338 = None
        add_tensor_73: "f32[6272, 1024]" = torch.ops.aten.add.Tensor(mm_default_73, arg80_1);  mm_default_73 = arg80_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_496: "f32[8, 784, 1024]" = torch.ops.aten.reshape.default(add_tensor_73, [8, 784, 1024]);  add_tensor_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_293: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_496, 0.5)
        mul_294: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_496, 0.7071067811865476);  view_496 = None
        erf_31: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_294);  mul_294 = None
        add_299: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_295: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_293, add_299);  mul_293 = add_299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_497: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_295, [6272, 1024]);  mul_295 = None
        permute_339: "f32[1024, 128]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        
        # No stacktrace found for following nodes
        mm_default_72: "f32[6272, 128]" = torch.ops.aten.mm.default(view_497, permute_339);  view_497 = permute_339 = None
        add_tensor_72: "f32[6272, 128]" = torch.ops.aten.add.Tensor(mm_default_72, arg82_1);  mm_default_72 = arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_498: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(add_tensor_72, [8, 784, 128]);  add_tensor_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_300: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_296, view_498);  add_296 = view_498 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:243 in forward, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
        permute_340: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_300, [0, 2, 1]);  add_300 = None
        view_499: "f32[8, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_340, [8, 128, 28, 28]);  permute_340 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:244 in forward, code: x = self.proj(cnn_feat_token)
        convolution_40: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(view_499, arg83_1, arg84_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  arg83_1 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:246 in forward, code: x += cnn_feat_token
        add_301: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(convolution_40, view_499);  convolution_40 = view_499 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        view_501: "f32[8, 128, 784]" = torch.ops.aten.reshape.default(add_301, [8, 128, 784]);  add_301 = None
        permute_342: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_501, [0, 2, 1]);  view_501 = None
        var_mean_100 = torch.ops.aten.var_mean.correction(permute_342, [2], correction = 0, keepdim = True)
        getitem_392: "f32[8, 784, 1]" = var_mean_100[0]
        getitem_393: "f32[8, 784, 1]" = var_mean_100[1];  var_mean_100 = None
        sub_100: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(permute_342, getitem_393);  getitem_393 = None
        add_302: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_392, 1e-06);  getitem_392 = None
        rsqrt_100: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_302);  add_302 = None
        mul_296: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = rsqrt_100 = None
        mul_297: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_296, arg85_1);  mul_296 = arg85_1 = None
        add_303: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_297, arg86_1);  mul_297 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_345: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_303, [0, 2, 1])
        view_505: "f32[8, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_345, [8, 128, 28, 28]);  permute_345 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_41: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(view_505, arg89_1, arg90_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  view_505 = arg89_1 = arg90_1 = None
        view_506: "f32[8, 128, 49]" = torch.ops.aten.reshape.default(convolution_41, [8, 128, 49]);  convolution_41 = None
        permute_346: "f32[8, 49, 128]" = torch.ops.aten.permute.default(view_506, [0, 2, 1]);  view_506 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_101 = torch.ops.aten.var_mean.correction(permute_346, [2], correction = 0, keepdim = True)
        getitem_394: "f32[8, 49, 1]" = var_mean_101[0]
        getitem_395: "f32[8, 49, 1]" = var_mean_101[1];  var_mean_101 = None
        sub_101: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(permute_346, getitem_395);  permute_346 = getitem_395 = None
        add_304: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_394, 1e-05);  getitem_394 = None
        rsqrt_101: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_304);  add_304 = None
        mul_298: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = rsqrt_101 = None
        mul_299: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_298, arg91_1);  mul_298 = arg91_1 = None
        add_305: "f32[8, 49, 128]" = torch.ops.aten.add.Tensor(mul_299, arg92_1);  mul_299 = arg92_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_507: "f32[392, 128]" = torch.ops.aten.reshape.default(add_305, [392, 128]);  add_305 = None
        permute_347: "f32[128, 256]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_162: "f32[392, 256]" = torch.ops.aten.addmm.default(arg94_1, view_507, permute_347);  arg94_1 = view_507 = permute_347 = None
        view_508: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(addmm_162, [8, 49, 256]);  addmm_162 = None
        view_509: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.reshape.default(view_508, [8, -1, 2, 2, 64]);  view_508 = None
        permute_348: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.permute.default(view_509, [2, 0, 3, 1, 4]);  view_509 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_32 = torch.ops.aten.unbind.int(permute_348);  permute_348 = None
        getitem_396: "f32[8, 2, 49, 64]" = unbind_32[0]
        getitem_397: "f32[8, 2, 49, 64]" = unbind_32[1];  unbind_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_502: "f32[6272, 128]" = torch.ops.aten.reshape.default(add_303, [6272, 128]);  add_303 = None
        permute_343: "f32[128, 128]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_161: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg88_1, view_502, permute_343);  arg88_1 = view_502 = permute_343 = None
        view_503: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(addmm_161, [8, 784, 128]);  addmm_161 = None
        view_504: "f32[8, 784, 2, 64]" = torch.ops.aten.reshape.default(view_503, [8, 784, 2, 64]);  view_503 = None
        permute_344: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_32 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_344, getitem_396, getitem_397, None, False);  permute_344 = getitem_396 = getitem_397 = None
        getitem_398: "f32[8, 2, 784, 64]" = _scaled_dot_product_efficient_attention_32[0];  _scaled_dot_product_efficient_attention_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_349: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_398, [0, 2, 1, 3]);  getitem_398 = None
        view_510: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(permute_349, [8, 784, 128]);  permute_349 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_511: "f32[6272, 128]" = torch.ops.aten.reshape.default(view_510, [6272, 128]);  view_510 = None
        permute_350: "f32[128, 128]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        
        # No stacktrace found for following nodes
        mm_default_71: "f32[6272, 128]" = torch.ops.aten.mm.default(view_511, permute_350);  view_511 = permute_350 = None
        add_tensor_71: "f32[6272, 128]" = torch.ops.aten.add.Tensor(mm_default_71, arg96_1);  mm_default_71 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_512: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(add_tensor_71, [8, 784, 128]);  add_tensor_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_306: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(permute_342, view_512);  permute_342 = view_512 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_102 = torch.ops.aten.var_mean.correction(add_306, [2], correction = 0, keepdim = True)
        getitem_402: "f32[8, 784, 1]" = var_mean_102[0]
        getitem_403: "f32[8, 784, 1]" = var_mean_102[1];  var_mean_102 = None
        sub_102: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_306, getitem_403);  getitem_403 = None
        add_307: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_402, 1e-06);  getitem_402 = None
        rsqrt_102: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_307);  add_307 = None
        mul_300: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = rsqrt_102 = None
        mul_301: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_300, arg97_1);  mul_300 = arg97_1 = None
        add_308: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_301, arg98_1);  mul_301 = arg98_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_513: "f32[6272, 128]" = torch.ops.aten.reshape.default(add_308, [6272, 128]);  add_308 = None
        permute_351: "f32[128, 1024]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        
        # No stacktrace found for following nodes
        mm_default_70: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_513, permute_351);  view_513 = permute_351 = None
        add_tensor_70: "f32[6272, 1024]" = torch.ops.aten.add.Tensor(mm_default_70, arg100_1);  mm_default_70 = arg100_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_514: "f32[8, 784, 1024]" = torch.ops.aten.reshape.default(add_tensor_70, [8, 784, 1024]);  add_tensor_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_302: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_514, 0.5)
        mul_303: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_514, 0.7071067811865476);  view_514 = None
        erf_32: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_303);  mul_303 = None
        add_309: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_304: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_302, add_309);  mul_302 = add_309 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_515: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_304, [6272, 1024]);  mul_304 = None
        permute_352: "f32[1024, 128]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        
        # No stacktrace found for following nodes
        mm_default_69: "f32[6272, 128]" = torch.ops.aten.mm.default(view_515, permute_352);  view_515 = permute_352 = None
        add_tensor_69: "f32[6272, 128]" = torch.ops.aten.add.Tensor(mm_default_69, arg102_1);  mm_default_69 = arg102_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_516: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(add_tensor_69, [8, 784, 128]);  add_tensor_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_310: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_306, view_516);  add_306 = view_516 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_103 = torch.ops.aten.var_mean.correction(add_310, [2], correction = 0, keepdim = True)
        getitem_404: "f32[8, 784, 1]" = var_mean_103[0]
        getitem_405: "f32[8, 784, 1]" = var_mean_103[1];  var_mean_103 = None
        sub_103: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_310, getitem_405);  getitem_405 = None
        add_311: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_404, 1e-06);  getitem_404 = None
        rsqrt_103: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_311);  add_311 = None
        mul_305: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = rsqrt_103 = None
        mul_306: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_305, arg103_1);  mul_305 = arg103_1 = None
        add_312: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_306, arg104_1);  mul_306 = arg104_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_355: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_312, [0, 2, 1])
        view_520: "f32[8, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_355, [8, 128, 28, 28]);  permute_355 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_42: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(view_520, arg107_1, arg108_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  view_520 = arg107_1 = arg108_1 = None
        view_521: "f32[8, 128, 49]" = torch.ops.aten.reshape.default(convolution_42, [8, 128, 49]);  convolution_42 = None
        permute_356: "f32[8, 49, 128]" = torch.ops.aten.permute.default(view_521, [0, 2, 1]);  view_521 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_104 = torch.ops.aten.var_mean.correction(permute_356, [2], correction = 0, keepdim = True)
        getitem_406: "f32[8, 49, 1]" = var_mean_104[0]
        getitem_407: "f32[8, 49, 1]" = var_mean_104[1];  var_mean_104 = None
        sub_104: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(permute_356, getitem_407);  permute_356 = getitem_407 = None
        add_313: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_406, 1e-05);  getitem_406 = None
        rsqrt_104: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        mul_307: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_104);  sub_104 = rsqrt_104 = None
        mul_308: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_307, arg109_1);  mul_307 = arg109_1 = None
        add_314: "f32[8, 49, 128]" = torch.ops.aten.add.Tensor(mul_308, arg110_1);  mul_308 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_522: "f32[392, 128]" = torch.ops.aten.reshape.default(add_314, [392, 128]);  add_314 = None
        permute_357: "f32[128, 256]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_167: "f32[392, 256]" = torch.ops.aten.addmm.default(arg112_1, view_522, permute_357);  arg112_1 = view_522 = permute_357 = None
        view_523: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(addmm_167, [8, 49, 256]);  addmm_167 = None
        view_524: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.reshape.default(view_523, [8, -1, 2, 2, 64]);  view_523 = None
        permute_358: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.permute.default(view_524, [2, 0, 3, 1, 4]);  view_524 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_33 = torch.ops.aten.unbind.int(permute_358);  permute_358 = None
        getitem_408: "f32[8, 2, 49, 64]" = unbind_33[0]
        getitem_409: "f32[8, 2, 49, 64]" = unbind_33[1];  unbind_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_517: "f32[6272, 128]" = torch.ops.aten.reshape.default(add_312, [6272, 128]);  add_312 = None
        permute_353: "f32[128, 128]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_166: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg106_1, view_517, permute_353);  arg106_1 = view_517 = permute_353 = None
        view_518: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(addmm_166, [8, 784, 128]);  addmm_166 = None
        view_519: "f32[8, 784, 2, 64]" = torch.ops.aten.reshape.default(view_518, [8, 784, 2, 64]);  view_518 = None
        permute_354: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_519, [0, 2, 1, 3]);  view_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_33 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_354, getitem_408, getitem_409, None, False);  permute_354 = getitem_408 = getitem_409 = None
        getitem_410: "f32[8, 2, 784, 64]" = _scaled_dot_product_efficient_attention_33[0];  _scaled_dot_product_efficient_attention_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_359: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_410, [0, 2, 1, 3]);  getitem_410 = None
        view_525: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(permute_359, [8, 784, 128]);  permute_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_526: "f32[6272, 128]" = torch.ops.aten.reshape.default(view_525, [6272, 128]);  view_525 = None
        permute_360: "f32[128, 128]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        
        # No stacktrace found for following nodes
        mm_default_68: "f32[6272, 128]" = torch.ops.aten.mm.default(view_526, permute_360);  view_526 = permute_360 = None
        add_tensor_68: "f32[6272, 128]" = torch.ops.aten.add.Tensor(mm_default_68, arg114_1);  mm_default_68 = arg114_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_527: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(add_tensor_68, [8, 784, 128]);  add_tensor_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_315: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_310, view_527);  add_310 = view_527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_105 = torch.ops.aten.var_mean.correction(add_315, [2], correction = 0, keepdim = True)
        getitem_414: "f32[8, 784, 1]" = var_mean_105[0]
        getitem_415: "f32[8, 784, 1]" = var_mean_105[1];  var_mean_105 = None
        sub_105: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_315, getitem_415);  getitem_415 = None
        add_316: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_414, 1e-06);  getitem_414 = None
        rsqrt_105: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_316);  add_316 = None
        mul_309: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_105);  sub_105 = rsqrt_105 = None
        mul_310: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_309, arg115_1);  mul_309 = arg115_1 = None
        add_317: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_310, arg116_1);  mul_310 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_528: "f32[6272, 128]" = torch.ops.aten.reshape.default(add_317, [6272, 128]);  add_317 = None
        permute_361: "f32[128, 1024]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        
        # No stacktrace found for following nodes
        mm_default_67: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_528, permute_361);  view_528 = permute_361 = None
        add_tensor_67: "f32[6272, 1024]" = torch.ops.aten.add.Tensor(mm_default_67, arg118_1);  mm_default_67 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_529: "f32[8, 784, 1024]" = torch.ops.aten.reshape.default(add_tensor_67, [8, 784, 1024]);  add_tensor_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_311: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        mul_312: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476);  view_529 = None
        erf_33: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_312);  mul_312 = None
        add_318: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_313: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_311, add_318);  mul_311 = add_318 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_530: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_313, [6272, 1024]);  mul_313 = None
        permute_362: "f32[1024, 128]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        
        # No stacktrace found for following nodes
        mm_default_66: "f32[6272, 128]" = torch.ops.aten.mm.default(view_530, permute_362);  view_530 = permute_362 = None
        add_tensor_66: "f32[6272, 128]" = torch.ops.aten.add.Tensor(mm_default_66, arg120_1);  mm_default_66 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_531: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(add_tensor_66, [8, 784, 128]);  add_tensor_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_319: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_315, view_531);  add_315 = view_531 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_106 = torch.ops.aten.var_mean.correction(add_319, [2], correction = 0, keepdim = True)
        getitem_416: "f32[8, 784, 1]" = var_mean_106[0]
        getitem_417: "f32[8, 784, 1]" = var_mean_106[1];  var_mean_106 = None
        sub_106: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_319, getitem_417);  getitem_417 = None
        add_320: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_416, 1e-06);  getitem_416 = None
        rsqrt_106: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        mul_314: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_106);  sub_106 = rsqrt_106 = None
        mul_315: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_314, arg121_1);  mul_314 = arg121_1 = None
        add_321: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_315, arg122_1);  mul_315 = arg122_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_365: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_321, [0, 2, 1])
        view_535: "f32[8, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_365, [8, 128, 28, 28]);  permute_365 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_43: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(view_535, arg125_1, arg126_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  view_535 = arg125_1 = arg126_1 = None
        view_536: "f32[8, 128, 49]" = torch.ops.aten.reshape.default(convolution_43, [8, 128, 49]);  convolution_43 = None
        permute_366: "f32[8, 49, 128]" = torch.ops.aten.permute.default(view_536, [0, 2, 1]);  view_536 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_107 = torch.ops.aten.var_mean.correction(permute_366, [2], correction = 0, keepdim = True)
        getitem_418: "f32[8, 49, 1]" = var_mean_107[0]
        getitem_419: "f32[8, 49, 1]" = var_mean_107[1];  var_mean_107 = None
        sub_107: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(permute_366, getitem_419);  permute_366 = getitem_419 = None
        add_322: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_418, 1e-05);  getitem_418 = None
        rsqrt_107: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_322);  add_322 = None
        mul_316: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_107);  sub_107 = rsqrt_107 = None
        mul_317: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_316, arg127_1);  mul_316 = arg127_1 = None
        add_323: "f32[8, 49, 128]" = torch.ops.aten.add.Tensor(mul_317, arg128_1);  mul_317 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_537: "f32[392, 128]" = torch.ops.aten.reshape.default(add_323, [392, 128]);  add_323 = None
        permute_367: "f32[128, 256]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_172: "f32[392, 256]" = torch.ops.aten.addmm.default(arg130_1, view_537, permute_367);  arg130_1 = view_537 = permute_367 = None
        view_538: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(addmm_172, [8, 49, 256]);  addmm_172 = None
        view_539: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.reshape.default(view_538, [8, -1, 2, 2, 64]);  view_538 = None
        permute_368: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.permute.default(view_539, [2, 0, 3, 1, 4]);  view_539 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_34 = torch.ops.aten.unbind.int(permute_368);  permute_368 = None
        getitem_420: "f32[8, 2, 49, 64]" = unbind_34[0]
        getitem_421: "f32[8, 2, 49, 64]" = unbind_34[1];  unbind_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_532: "f32[6272, 128]" = torch.ops.aten.reshape.default(add_321, [6272, 128]);  add_321 = None
        permute_363: "f32[128, 128]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_171: "f32[6272, 128]" = torch.ops.aten.addmm.default(arg124_1, view_532, permute_363);  arg124_1 = view_532 = permute_363 = None
        view_533: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(addmm_171, [8, 784, 128]);  addmm_171 = None
        view_534: "f32[8, 784, 2, 64]" = torch.ops.aten.reshape.default(view_533, [8, 784, 2, 64]);  view_533 = None
        permute_364: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_34 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_364, getitem_420, getitem_421, None, False);  permute_364 = getitem_420 = getitem_421 = None
        getitem_422: "f32[8, 2, 784, 64]" = _scaled_dot_product_efficient_attention_34[0];  _scaled_dot_product_efficient_attention_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_369: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_422, [0, 2, 1, 3]);  getitem_422 = None
        view_540: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(permute_369, [8, 784, 128]);  permute_369 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_541: "f32[6272, 128]" = torch.ops.aten.reshape.default(view_540, [6272, 128]);  view_540 = None
        permute_370: "f32[128, 128]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        
        # No stacktrace found for following nodes
        mm_default_65: "f32[6272, 128]" = torch.ops.aten.mm.default(view_541, permute_370);  view_541 = permute_370 = None
        add_tensor_65: "f32[6272, 128]" = torch.ops.aten.add.Tensor(mm_default_65, arg132_1);  mm_default_65 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_542: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(add_tensor_65, [8, 784, 128]);  add_tensor_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_324: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_319, view_542);  add_319 = view_542 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_108 = torch.ops.aten.var_mean.correction(add_324, [2], correction = 0, keepdim = True)
        getitem_426: "f32[8, 784, 1]" = var_mean_108[0]
        getitem_427: "f32[8, 784, 1]" = var_mean_108[1];  var_mean_108 = None
        sub_108: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_324, getitem_427);  getitem_427 = None
        add_325: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_426, 1e-06);  getitem_426 = None
        rsqrt_108: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_325);  add_325 = None
        mul_318: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_108);  sub_108 = rsqrt_108 = None
        mul_319: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_318, arg133_1);  mul_318 = arg133_1 = None
        add_326: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_319, arg134_1);  mul_319 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_543: "f32[6272, 128]" = torch.ops.aten.reshape.default(add_326, [6272, 128]);  add_326 = None
        permute_371: "f32[128, 1024]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        
        # No stacktrace found for following nodes
        mm_default_64: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_543, permute_371);  view_543 = permute_371 = None
        add_tensor_64: "f32[6272, 1024]" = torch.ops.aten.add.Tensor(mm_default_64, arg136_1);  mm_default_64 = arg136_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_544: "f32[8, 784, 1024]" = torch.ops.aten.reshape.default(add_tensor_64, [8, 784, 1024]);  add_tensor_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_320: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_544, 0.5)
        mul_321: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_544, 0.7071067811865476);  view_544 = None
        erf_34: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_321);  mul_321 = None
        add_327: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_322: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_320, add_327);  mul_320 = add_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_545: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_322, [6272, 1024]);  mul_322 = None
        permute_372: "f32[1024, 128]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        
        # No stacktrace found for following nodes
        mm_default_63: "f32[6272, 128]" = torch.ops.aten.mm.default(view_545, permute_372);  view_545 = permute_372 = None
        add_tensor_63: "f32[6272, 128]" = torch.ops.aten.add.Tensor(mm_default_63, arg138_1);  mm_default_63 = arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_546: "f32[8, 784, 128]" = torch.ops.aten.reshape.default(add_tensor_63, [8, 784, 128]);  add_tensor_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_328: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_324, view_546);  add_324 = view_546 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:489 in forward_features, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
        view_547: "f32[8, 28, 28, 128]" = torch.ops.aten.reshape.default(add_328, [8, 28, 28, -1]);  add_328 = None
        permute_373: "f32[8, 128, 28, 28]" = torch.ops.aten.permute.default(view_547, [0, 3, 1, 2]);  view_547 = None
        clone_122: "f32[8, 128, 28, 28]" = torch.ops.aten.clone.default(permute_373, memory_format = torch.contiguous_format);  permute_373 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:275 in forward, code: x = self.proj(x).flatten(2).transpose(1, 2)
        convolution_44: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(clone_122, arg139_1, arg140_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_122 = arg139_1 = arg140_1 = None
        view_548: "f32[8, 320, 196]" = torch.ops.aten.reshape.default(convolution_44, [8, 320, 196]);  convolution_44 = None
        permute_374: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_548, [0, 2, 1]);  view_548 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:276 in forward, code: x = self.norm(x)
        clone_123: "f32[8, 196, 320]" = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
        var_mean_109 = torch.ops.aten.var_mean.correction(clone_123, [2], correction = 0, keepdim = True)
        getitem_428: "f32[8, 196, 1]" = var_mean_109[0]
        getitem_429: "f32[8, 196, 1]" = var_mean_109[1];  var_mean_109 = None
        sub_109: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_123, getitem_429);  clone_123 = getitem_429 = None
        add_329: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_428, 1e-05);  getitem_428 = None
        rsqrt_109: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_329);  add_329 = None
        mul_323: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_109);  sub_109 = rsqrt_109 = None
        mul_324: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_323, arg141_1);  mul_323 = arg141_1 = None
        add_330: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_324, arg142_1);  mul_324 = arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_110 = torch.ops.aten.var_mean.correction(add_330, [2], correction = 0, keepdim = True)
        getitem_430: "f32[8, 196, 1]" = var_mean_110[0]
        getitem_431: "f32[8, 196, 1]" = var_mean_110[1];  var_mean_110 = None
        sub_110: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_330, getitem_431);  getitem_431 = None
        add_331: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_430, 1e-06);  getitem_430 = None
        rsqrt_110: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        mul_325: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_110);  sub_110 = rsqrt_110 = None
        mul_326: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_325, arg143_1);  mul_325 = arg143_1 = None
        add_332: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_326, arg144_1);  mul_326 = arg144_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_377: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_332, [0, 2, 1])
        view_552: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_377, [8, 320, 14, 14]);  permute_377 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_45: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_552, arg147_1, arg148_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_552 = arg147_1 = arg148_1 = None
        view_553: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_45, [8, 320, 49]);  convolution_45 = None
        permute_378: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_553, [0, 2, 1]);  view_553 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_111 = torch.ops.aten.var_mean.correction(permute_378, [2], correction = 0, keepdim = True)
        getitem_432: "f32[8, 49, 1]" = var_mean_111[0]
        getitem_433: "f32[8, 49, 1]" = var_mean_111[1];  var_mean_111 = None
        sub_111: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_378, getitem_433);  permute_378 = getitem_433 = None
        add_333: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_432, 1e-05);  getitem_432 = None
        rsqrt_111: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_333);  add_333 = None
        mul_327: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_111);  sub_111 = rsqrt_111 = None
        mul_328: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_327, arg149_1);  mul_327 = arg149_1 = None
        add_334: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_328, arg150_1);  mul_328 = arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_554: "f32[392, 320]" = torch.ops.aten.reshape.default(add_334, [392, 320]);  add_334 = None
        permute_379: "f32[320, 640]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_177: "f32[392, 640]" = torch.ops.aten.addmm.default(arg152_1, view_554, permute_379);  arg152_1 = view_554 = permute_379 = None
        view_555: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_177, [8, 49, 640]);  addmm_177 = None
        view_556: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_555, [8, -1, 2, 5, 64]);  view_555 = None
        permute_380: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_556, [2, 0, 3, 1, 4]);  view_556 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_35 = torch.ops.aten.unbind.int(permute_380);  permute_380 = None
        getitem_434: "f32[8, 5, 49, 64]" = unbind_35[0]
        getitem_435: "f32[8, 5, 49, 64]" = unbind_35[1];  unbind_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_549: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_332, [1568, 320]);  add_332 = None
        permute_375: "f32[320, 320]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_176: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg146_1, view_549, permute_375);  arg146_1 = view_549 = permute_375 = None
        view_550: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_176, [8, 196, 320]);  addmm_176 = None
        view_551: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_550, [8, 196, 5, 64]);  view_550 = None
        permute_376: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_551, [0, 2, 1, 3]);  view_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_35 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_376, getitem_434, getitem_435, None, False);  permute_376 = getitem_434 = getitem_435 = None
        getitem_436: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_35[0];  _scaled_dot_product_efficient_attention_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_381: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_436, [0, 2, 1, 3]);  getitem_436 = None
        view_557: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_381, [8, 196, 320]);  permute_381 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_558: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_557, [1568, 320]);  view_557 = None
        permute_382: "f32[320, 320]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        
        # No stacktrace found for following nodes
        mm_default_62: "f32[1568, 320]" = torch.ops.aten.mm.default(view_558, permute_382);  view_558 = permute_382 = None
        add_tensor_62: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_62, arg154_1);  mm_default_62 = arg154_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_559: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_62, [8, 196, 320]);  add_tensor_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_335: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_330, view_559);  add_330 = view_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_112 = torch.ops.aten.var_mean.correction(add_335, [2], correction = 0, keepdim = True)
        getitem_440: "f32[8, 196, 1]" = var_mean_112[0]
        getitem_441: "f32[8, 196, 1]" = var_mean_112[1];  var_mean_112 = None
        sub_112: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_335, getitem_441);  getitem_441 = None
        add_336: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_440, 1e-06);  getitem_440 = None
        rsqrt_112: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_336);  add_336 = None
        mul_329: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_112);  sub_112 = rsqrt_112 = None
        mul_330: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_329, arg155_1);  mul_329 = arg155_1 = None
        add_337: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_330, arg156_1);  mul_330 = arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_560: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_337, [1568, 320]);  add_337 = None
        permute_383: "f32[320, 1280]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        
        # No stacktrace found for following nodes
        mm_default_61: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_560, permute_383);  view_560 = permute_383 = None
        add_tensor_61: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_61, arg158_1);  mm_default_61 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_561: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_61, [8, 196, 1280]);  add_tensor_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_331: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_561, 0.5)
        mul_332: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_561, 0.7071067811865476);  view_561 = None
        erf_35: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_332);  mul_332 = None
        add_338: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_333: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_331, add_338);  mul_331 = add_338 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_562: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_333, [1568, 1280]);  mul_333 = None
        permute_384: "f32[1280, 320]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        
        # No stacktrace found for following nodes
        mm_default_60: "f32[1568, 320]" = torch.ops.aten.mm.default(view_562, permute_384);  view_562 = permute_384 = None
        add_tensor_60: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_60, arg160_1);  mm_default_60 = arg160_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_563: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_60, [8, 196, 320]);  add_tensor_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_339: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_335, view_563);  add_335 = view_563 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:243 in forward, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
        permute_385: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_339, [0, 2, 1]);  add_339 = None
        view_564: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_385, [8, 320, 14, 14]);  permute_385 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:244 in forward, code: x = self.proj(cnn_feat_token)
        convolution_46: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(view_564, arg161_1, arg162_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 320);  arg161_1 = arg162_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:246 in forward, code: x += cnn_feat_token
        add_340: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(convolution_46, view_564);  convolution_46 = view_564 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        view_566: "f32[8, 320, 196]" = torch.ops.aten.reshape.default(add_340, [8, 320, 196]);  add_340 = None
        permute_387: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_566, [0, 2, 1]);  view_566 = None
        var_mean_113 = torch.ops.aten.var_mean.correction(permute_387, [2], correction = 0, keepdim = True)
        getitem_442: "f32[8, 196, 1]" = var_mean_113[0]
        getitem_443: "f32[8, 196, 1]" = var_mean_113[1];  var_mean_113 = None
        sub_113: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(permute_387, getitem_443);  getitem_443 = None
        add_341: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_442, 1e-06);  getitem_442 = None
        rsqrt_113: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        mul_334: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_113);  sub_113 = rsqrt_113 = None
        mul_335: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_334, arg163_1);  mul_334 = arg163_1 = None
        add_342: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_335, arg164_1);  mul_335 = arg164_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_390: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_342, [0, 2, 1])
        view_570: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_390, [8, 320, 14, 14]);  permute_390 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_47: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_570, arg167_1, arg168_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_570 = arg167_1 = arg168_1 = None
        view_571: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_47, [8, 320, 49]);  convolution_47 = None
        permute_391: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_571, [0, 2, 1]);  view_571 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_114 = torch.ops.aten.var_mean.correction(permute_391, [2], correction = 0, keepdim = True)
        getitem_444: "f32[8, 49, 1]" = var_mean_114[0]
        getitem_445: "f32[8, 49, 1]" = var_mean_114[1];  var_mean_114 = None
        sub_114: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_391, getitem_445);  permute_391 = getitem_445 = None
        add_343: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_444, 1e-05);  getitem_444 = None
        rsqrt_114: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_343);  add_343 = None
        mul_336: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_114);  sub_114 = rsqrt_114 = None
        mul_337: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_336, arg169_1);  mul_336 = arg169_1 = None
        add_344: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_337, arg170_1);  mul_337 = arg170_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_572: "f32[392, 320]" = torch.ops.aten.reshape.default(add_344, [392, 320]);  add_344 = None
        permute_392: "f32[320, 640]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_182: "f32[392, 640]" = torch.ops.aten.addmm.default(arg172_1, view_572, permute_392);  arg172_1 = view_572 = permute_392 = None
        view_573: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_182, [8, 49, 640]);  addmm_182 = None
        view_574: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_573, [8, -1, 2, 5, 64]);  view_573 = None
        permute_393: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_574, [2, 0, 3, 1, 4]);  view_574 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_36 = torch.ops.aten.unbind.int(permute_393);  permute_393 = None
        getitem_446: "f32[8, 5, 49, 64]" = unbind_36[0]
        getitem_447: "f32[8, 5, 49, 64]" = unbind_36[1];  unbind_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_567: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_342, [1568, 320]);  add_342 = None
        permute_388: "f32[320, 320]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_181: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg166_1, view_567, permute_388);  arg166_1 = view_567 = permute_388 = None
        view_568: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_181, [8, 196, 320]);  addmm_181 = None
        view_569: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_568, [8, 196, 5, 64]);  view_568 = None
        permute_389: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_569, [0, 2, 1, 3]);  view_569 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_36 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_389, getitem_446, getitem_447, None, False);  permute_389 = getitem_446 = getitem_447 = None
        getitem_448: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_36[0];  _scaled_dot_product_efficient_attention_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_394: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_448, [0, 2, 1, 3]);  getitem_448 = None
        view_575: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_394, [8, 196, 320]);  permute_394 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_576: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_575, [1568, 320]);  view_575 = None
        permute_395: "f32[320, 320]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        
        # No stacktrace found for following nodes
        mm_default_59: "f32[1568, 320]" = torch.ops.aten.mm.default(view_576, permute_395);  view_576 = permute_395 = None
        add_tensor_59: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_59, arg174_1);  mm_default_59 = arg174_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_577: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_59, [8, 196, 320]);  add_tensor_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_345: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_387, view_577);  permute_387 = view_577 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_115 = torch.ops.aten.var_mean.correction(add_345, [2], correction = 0, keepdim = True)
        getitem_452: "f32[8, 196, 1]" = var_mean_115[0]
        getitem_453: "f32[8, 196, 1]" = var_mean_115[1];  var_mean_115 = None
        sub_115: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_345, getitem_453);  getitem_453 = None
        add_346: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_452, 1e-06);  getitem_452 = None
        rsqrt_115: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_346);  add_346 = None
        mul_338: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_115);  sub_115 = rsqrt_115 = None
        mul_339: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_338, arg175_1);  mul_338 = arg175_1 = None
        add_347: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_339, arg176_1);  mul_339 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_578: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_347, [1568, 320]);  add_347 = None
        permute_396: "f32[320, 1280]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        
        # No stacktrace found for following nodes
        mm_default_58: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_578, permute_396);  view_578 = permute_396 = None
        add_tensor_58: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_58, arg178_1);  mm_default_58 = arg178_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_579: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_58, [8, 196, 1280]);  add_tensor_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_340: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_579, 0.5)
        mul_341: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_579, 0.7071067811865476);  view_579 = None
        erf_36: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_341);  mul_341 = None
        add_348: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_342: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_340, add_348);  mul_340 = add_348 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_580: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_342, [1568, 1280]);  mul_342 = None
        permute_397: "f32[1280, 320]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        
        # No stacktrace found for following nodes
        mm_default_57: "f32[1568, 320]" = torch.ops.aten.mm.default(view_580, permute_397);  view_580 = permute_397 = None
        add_tensor_57: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_57, arg180_1);  mm_default_57 = arg180_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_581: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_57, [8, 196, 320]);  add_tensor_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_349: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_345, view_581);  add_345 = view_581 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_116 = torch.ops.aten.var_mean.correction(add_349, [2], correction = 0, keepdim = True)
        getitem_454: "f32[8, 196, 1]" = var_mean_116[0]
        getitem_455: "f32[8, 196, 1]" = var_mean_116[1];  var_mean_116 = None
        sub_116: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_349, getitem_455);  getitem_455 = None
        add_350: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_454, 1e-06);  getitem_454 = None
        rsqrt_116: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_350);  add_350 = None
        mul_343: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_116);  sub_116 = rsqrt_116 = None
        mul_344: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_343, arg181_1);  mul_343 = arg181_1 = None
        add_351: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_344, arg182_1);  mul_344 = arg182_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_400: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_351, [0, 2, 1])
        view_585: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_400, [8, 320, 14, 14]);  permute_400 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_48: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_585, arg185_1, arg186_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_585 = arg185_1 = arg186_1 = None
        view_586: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_48, [8, 320, 49]);  convolution_48 = None
        permute_401: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_586, [0, 2, 1]);  view_586 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_117 = torch.ops.aten.var_mean.correction(permute_401, [2], correction = 0, keepdim = True)
        getitem_456: "f32[8, 49, 1]" = var_mean_117[0]
        getitem_457: "f32[8, 49, 1]" = var_mean_117[1];  var_mean_117 = None
        sub_117: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_401, getitem_457);  permute_401 = getitem_457 = None
        add_352: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_456, 1e-05);  getitem_456 = None
        rsqrt_117: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_352);  add_352 = None
        mul_345: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_117);  sub_117 = rsqrt_117 = None
        mul_346: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_345, arg187_1);  mul_345 = arg187_1 = None
        add_353: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_346, arg188_1);  mul_346 = arg188_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_587: "f32[392, 320]" = torch.ops.aten.reshape.default(add_353, [392, 320]);  add_353 = None
        permute_402: "f32[320, 640]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_187: "f32[392, 640]" = torch.ops.aten.addmm.default(arg190_1, view_587, permute_402);  arg190_1 = view_587 = permute_402 = None
        view_588: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_187, [8, 49, 640]);  addmm_187 = None
        view_589: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_588, [8, -1, 2, 5, 64]);  view_588 = None
        permute_403: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_589, [2, 0, 3, 1, 4]);  view_589 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_37 = torch.ops.aten.unbind.int(permute_403);  permute_403 = None
        getitem_458: "f32[8, 5, 49, 64]" = unbind_37[0]
        getitem_459: "f32[8, 5, 49, 64]" = unbind_37[1];  unbind_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_582: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_351, [1568, 320]);  add_351 = None
        permute_398: "f32[320, 320]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_186: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg184_1, view_582, permute_398);  arg184_1 = view_582 = permute_398 = None
        view_583: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_186, [8, 196, 320]);  addmm_186 = None
        view_584: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_583, [8, 196, 5, 64]);  view_583 = None
        permute_399: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_584, [0, 2, 1, 3]);  view_584 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_37 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_399, getitem_458, getitem_459, None, False);  permute_399 = getitem_458 = getitem_459 = None
        getitem_460: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_37[0];  _scaled_dot_product_efficient_attention_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_404: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_460, [0, 2, 1, 3]);  getitem_460 = None
        view_590: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_404, [8, 196, 320]);  permute_404 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_591: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_590, [1568, 320]);  view_590 = None
        permute_405: "f32[320, 320]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        
        # No stacktrace found for following nodes
        mm_default_56: "f32[1568, 320]" = torch.ops.aten.mm.default(view_591, permute_405);  view_591 = permute_405 = None
        add_tensor_56: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_56, arg192_1);  mm_default_56 = arg192_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_592: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_56, [8, 196, 320]);  add_tensor_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_354: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_349, view_592);  add_349 = view_592 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_118 = torch.ops.aten.var_mean.correction(add_354, [2], correction = 0, keepdim = True)
        getitem_464: "f32[8, 196, 1]" = var_mean_118[0]
        getitem_465: "f32[8, 196, 1]" = var_mean_118[1];  var_mean_118 = None
        sub_118: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_354, getitem_465);  getitem_465 = None
        add_355: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_464, 1e-06);  getitem_464 = None
        rsqrt_118: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_355);  add_355 = None
        mul_347: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_118);  sub_118 = rsqrt_118 = None
        mul_348: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_347, arg193_1);  mul_347 = arg193_1 = None
        add_356: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_348, arg194_1);  mul_348 = arg194_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_593: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_356, [1568, 320]);  add_356 = None
        permute_406: "f32[320, 1280]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        
        # No stacktrace found for following nodes
        mm_default_55: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_593, permute_406);  view_593 = permute_406 = None
        add_tensor_55: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_55, arg196_1);  mm_default_55 = arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_594: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_55, [8, 196, 1280]);  add_tensor_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_349: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_594, 0.5)
        mul_350: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_594, 0.7071067811865476);  view_594 = None
        erf_37: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_350);  mul_350 = None
        add_357: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_351: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_349, add_357);  mul_349 = add_357 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_595: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_351, [1568, 1280]);  mul_351 = None
        permute_407: "f32[1280, 320]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        
        # No stacktrace found for following nodes
        mm_default_54: "f32[1568, 320]" = torch.ops.aten.mm.default(view_595, permute_407);  view_595 = permute_407 = None
        add_tensor_54: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_54, arg198_1);  mm_default_54 = arg198_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_596: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_54, [8, 196, 320]);  add_tensor_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_358: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_354, view_596);  add_354 = view_596 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_119 = torch.ops.aten.var_mean.correction(add_358, [2], correction = 0, keepdim = True)
        getitem_466: "f32[8, 196, 1]" = var_mean_119[0]
        getitem_467: "f32[8, 196, 1]" = var_mean_119[1];  var_mean_119 = None
        sub_119: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_358, getitem_467);  getitem_467 = None
        add_359: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_466, 1e-06);  getitem_466 = None
        rsqrt_119: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_359);  add_359 = None
        mul_352: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_119);  sub_119 = rsqrt_119 = None
        mul_353: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_352, arg199_1);  mul_352 = arg199_1 = None
        add_360: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_353, arg200_1);  mul_353 = arg200_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_410: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_360, [0, 2, 1])
        view_600: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_410, [8, 320, 14, 14]);  permute_410 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_49: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_600, arg203_1, arg204_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_600 = arg203_1 = arg204_1 = None
        view_601: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_49, [8, 320, 49]);  convolution_49 = None
        permute_411: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_601, [0, 2, 1]);  view_601 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_120 = torch.ops.aten.var_mean.correction(permute_411, [2], correction = 0, keepdim = True)
        getitem_468: "f32[8, 49, 1]" = var_mean_120[0]
        getitem_469: "f32[8, 49, 1]" = var_mean_120[1];  var_mean_120 = None
        sub_120: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_411, getitem_469);  permute_411 = getitem_469 = None
        add_361: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_468, 1e-05);  getitem_468 = None
        rsqrt_120: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_361);  add_361 = None
        mul_354: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_120);  sub_120 = rsqrt_120 = None
        mul_355: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_354, arg205_1);  mul_354 = arg205_1 = None
        add_362: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_355, arg206_1);  mul_355 = arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_602: "f32[392, 320]" = torch.ops.aten.reshape.default(add_362, [392, 320]);  add_362 = None
        permute_412: "f32[320, 640]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_192: "f32[392, 640]" = torch.ops.aten.addmm.default(arg208_1, view_602, permute_412);  arg208_1 = view_602 = permute_412 = None
        view_603: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_192, [8, 49, 640]);  addmm_192 = None
        view_604: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_603, [8, -1, 2, 5, 64]);  view_603 = None
        permute_413: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_604, [2, 0, 3, 1, 4]);  view_604 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_38 = torch.ops.aten.unbind.int(permute_413);  permute_413 = None
        getitem_470: "f32[8, 5, 49, 64]" = unbind_38[0]
        getitem_471: "f32[8, 5, 49, 64]" = unbind_38[1];  unbind_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_597: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_360, [1568, 320]);  add_360 = None
        permute_408: "f32[320, 320]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_191: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg202_1, view_597, permute_408);  arg202_1 = view_597 = permute_408 = None
        view_598: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_191, [8, 196, 320]);  addmm_191 = None
        view_599: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_598, [8, 196, 5, 64]);  view_598 = None
        permute_409: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_599, [0, 2, 1, 3]);  view_599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_38 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_409, getitem_470, getitem_471, None, False);  permute_409 = getitem_470 = getitem_471 = None
        getitem_472: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_38[0];  _scaled_dot_product_efficient_attention_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_414: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_472, [0, 2, 1, 3]);  getitem_472 = None
        view_605: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_414, [8, 196, 320]);  permute_414 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_606: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_605, [1568, 320]);  view_605 = None
        permute_415: "f32[320, 320]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        
        # No stacktrace found for following nodes
        mm_default_53: "f32[1568, 320]" = torch.ops.aten.mm.default(view_606, permute_415);  view_606 = permute_415 = None
        add_tensor_53: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_53, arg210_1);  mm_default_53 = arg210_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_607: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_53, [8, 196, 320]);  add_tensor_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_363: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_358, view_607);  add_358 = view_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_121 = torch.ops.aten.var_mean.correction(add_363, [2], correction = 0, keepdim = True)
        getitem_476: "f32[8, 196, 1]" = var_mean_121[0]
        getitem_477: "f32[8, 196, 1]" = var_mean_121[1];  var_mean_121 = None
        sub_121: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_363, getitem_477);  getitem_477 = None
        add_364: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_476, 1e-06);  getitem_476 = None
        rsqrt_121: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_364);  add_364 = None
        mul_356: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_121);  sub_121 = rsqrt_121 = None
        mul_357: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_356, arg211_1);  mul_356 = arg211_1 = None
        add_365: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_357, arg212_1);  mul_357 = arg212_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_608: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_365, [1568, 320]);  add_365 = None
        permute_416: "f32[320, 1280]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        
        # No stacktrace found for following nodes
        mm_default_52: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_608, permute_416);  view_608 = permute_416 = None
        add_tensor_52: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_52, arg214_1);  mm_default_52 = arg214_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_609: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_52, [8, 196, 1280]);  add_tensor_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_358: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_609, 0.5)
        mul_359: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_609, 0.7071067811865476);  view_609 = None
        erf_38: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_359);  mul_359 = None
        add_366: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_360: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_358, add_366);  mul_358 = add_366 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_610: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_360, [1568, 1280]);  mul_360 = None
        permute_417: "f32[1280, 320]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        
        # No stacktrace found for following nodes
        mm_default_51: "f32[1568, 320]" = torch.ops.aten.mm.default(view_610, permute_417);  view_610 = permute_417 = None
        add_tensor_51: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_51, arg216_1);  mm_default_51 = arg216_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_611: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_51, [8, 196, 320]);  add_tensor_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_367: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_363, view_611);  add_363 = view_611 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_122 = torch.ops.aten.var_mean.correction(add_367, [2], correction = 0, keepdim = True)
        getitem_478: "f32[8, 196, 1]" = var_mean_122[0]
        getitem_479: "f32[8, 196, 1]" = var_mean_122[1];  var_mean_122 = None
        sub_122: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_367, getitem_479);  getitem_479 = None
        add_368: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_478, 1e-06);  getitem_478 = None
        rsqrt_122: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_368);  add_368 = None
        mul_361: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_122);  sub_122 = rsqrt_122 = None
        mul_362: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_361, arg217_1);  mul_361 = arg217_1 = None
        add_369: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_362, arg218_1);  mul_362 = arg218_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_420: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_369, [0, 2, 1])
        view_615: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_420, [8, 320, 14, 14]);  permute_420 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_50: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_615, arg221_1, arg222_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_615 = arg221_1 = arg222_1 = None
        view_616: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_50, [8, 320, 49]);  convolution_50 = None
        permute_421: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_616, [0, 2, 1]);  view_616 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_123 = torch.ops.aten.var_mean.correction(permute_421, [2], correction = 0, keepdim = True)
        getitem_480: "f32[8, 49, 1]" = var_mean_123[0]
        getitem_481: "f32[8, 49, 1]" = var_mean_123[1];  var_mean_123 = None
        sub_123: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_421, getitem_481);  permute_421 = getitem_481 = None
        add_370: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_480, 1e-05);  getitem_480 = None
        rsqrt_123: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_370);  add_370 = None
        mul_363: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_123, rsqrt_123);  sub_123 = rsqrt_123 = None
        mul_364: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_363, arg223_1);  mul_363 = arg223_1 = None
        add_371: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_364, arg224_1);  mul_364 = arg224_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_617: "f32[392, 320]" = torch.ops.aten.reshape.default(add_371, [392, 320]);  add_371 = None
        permute_422: "f32[320, 640]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_197: "f32[392, 640]" = torch.ops.aten.addmm.default(arg226_1, view_617, permute_422);  arg226_1 = view_617 = permute_422 = None
        view_618: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_197, [8, 49, 640]);  addmm_197 = None
        view_619: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_618, [8, -1, 2, 5, 64]);  view_618 = None
        permute_423: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_619, [2, 0, 3, 1, 4]);  view_619 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_39 = torch.ops.aten.unbind.int(permute_423);  permute_423 = None
        getitem_482: "f32[8, 5, 49, 64]" = unbind_39[0]
        getitem_483: "f32[8, 5, 49, 64]" = unbind_39[1];  unbind_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_612: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_369, [1568, 320]);  add_369 = None
        permute_418: "f32[320, 320]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_196: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg220_1, view_612, permute_418);  arg220_1 = view_612 = permute_418 = None
        view_613: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_196, [8, 196, 320]);  addmm_196 = None
        view_614: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_613, [8, 196, 5, 64]);  view_613 = None
        permute_419: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_39 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_419, getitem_482, getitem_483, None, False);  permute_419 = getitem_482 = getitem_483 = None
        getitem_484: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_39[0];  _scaled_dot_product_efficient_attention_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_424: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_484, [0, 2, 1, 3]);  getitem_484 = None
        view_620: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_424, [8, 196, 320]);  permute_424 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_621: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_620, [1568, 320]);  view_620 = None
        permute_425: "f32[320, 320]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        
        # No stacktrace found for following nodes
        mm_default_50: "f32[1568, 320]" = torch.ops.aten.mm.default(view_621, permute_425);  view_621 = permute_425 = None
        add_tensor_50: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_50, arg228_1);  mm_default_50 = arg228_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_622: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_50, [8, 196, 320]);  add_tensor_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_372: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_367, view_622);  add_367 = view_622 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_124 = torch.ops.aten.var_mean.correction(add_372, [2], correction = 0, keepdim = True)
        getitem_488: "f32[8, 196, 1]" = var_mean_124[0]
        getitem_489: "f32[8, 196, 1]" = var_mean_124[1];  var_mean_124 = None
        sub_124: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_372, getitem_489);  getitem_489 = None
        add_373: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_488, 1e-06);  getitem_488 = None
        rsqrt_124: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_373);  add_373 = None
        mul_365: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_124);  sub_124 = rsqrt_124 = None
        mul_366: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_365, arg229_1);  mul_365 = arg229_1 = None
        add_374: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_366, arg230_1);  mul_366 = arg230_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_623: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_374, [1568, 320]);  add_374 = None
        permute_426: "f32[320, 1280]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        
        # No stacktrace found for following nodes
        mm_default_49: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_623, permute_426);  view_623 = permute_426 = None
        add_tensor_49: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_49, arg232_1);  mm_default_49 = arg232_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_624: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_49, [8, 196, 1280]);  add_tensor_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_367: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_624, 0.5)
        mul_368: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_624, 0.7071067811865476);  view_624 = None
        erf_39: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_368);  mul_368 = None
        add_375: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_369: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_367, add_375);  mul_367 = add_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_625: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_369, [1568, 1280]);  mul_369 = None
        permute_427: "f32[1280, 320]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        
        # No stacktrace found for following nodes
        mm_default_48: "f32[1568, 320]" = torch.ops.aten.mm.default(view_625, permute_427);  view_625 = permute_427 = None
        add_tensor_48: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_48, arg234_1);  mm_default_48 = arg234_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_626: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_48, [8, 196, 320]);  add_tensor_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_376: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_372, view_626);  add_372 = view_626 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_125 = torch.ops.aten.var_mean.correction(add_376, [2], correction = 0, keepdim = True)
        getitem_490: "f32[8, 196, 1]" = var_mean_125[0]
        getitem_491: "f32[8, 196, 1]" = var_mean_125[1];  var_mean_125 = None
        sub_125: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_376, getitem_491);  getitem_491 = None
        add_377: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_490, 1e-06);  getitem_490 = None
        rsqrt_125: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_377);  add_377 = None
        mul_370: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_125);  sub_125 = rsqrt_125 = None
        mul_371: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_370, arg235_1);  mul_370 = arg235_1 = None
        add_378: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_371, arg236_1);  mul_371 = arg236_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_430: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_378, [0, 2, 1])
        view_630: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_430, [8, 320, 14, 14]);  permute_430 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_51: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_630, arg239_1, arg240_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_630 = arg239_1 = arg240_1 = None
        view_631: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_51, [8, 320, 49]);  convolution_51 = None
        permute_431: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_631, [0, 2, 1]);  view_631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_126 = torch.ops.aten.var_mean.correction(permute_431, [2], correction = 0, keepdim = True)
        getitem_492: "f32[8, 49, 1]" = var_mean_126[0]
        getitem_493: "f32[8, 49, 1]" = var_mean_126[1];  var_mean_126 = None
        sub_126: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_431, getitem_493);  permute_431 = getitem_493 = None
        add_379: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_492, 1e-05);  getitem_492 = None
        rsqrt_126: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_379);  add_379 = None
        mul_372: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_126);  sub_126 = rsqrt_126 = None
        mul_373: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_372, arg241_1);  mul_372 = arg241_1 = None
        add_380: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_373, arg242_1);  mul_373 = arg242_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_632: "f32[392, 320]" = torch.ops.aten.reshape.default(add_380, [392, 320]);  add_380 = None
        permute_432: "f32[320, 640]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        addmm_202: "f32[392, 640]" = torch.ops.aten.addmm.default(arg244_1, view_632, permute_432);  arg244_1 = view_632 = permute_432 = None
        view_633: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_202, [8, 49, 640]);  addmm_202 = None
        view_634: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_633, [8, -1, 2, 5, 64]);  view_633 = None
        permute_433: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_634, [2, 0, 3, 1, 4]);  view_634 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_40 = torch.ops.aten.unbind.int(permute_433);  permute_433 = None
        getitem_494: "f32[8, 5, 49, 64]" = unbind_40[0]
        getitem_495: "f32[8, 5, 49, 64]" = unbind_40[1];  unbind_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_627: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_378, [1568, 320]);  add_378 = None
        permute_428: "f32[320, 320]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_201: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg238_1, view_627, permute_428);  arg238_1 = view_627 = permute_428 = None
        view_628: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_201, [8, 196, 320]);  addmm_201 = None
        view_629: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_628, [8, 196, 5, 64]);  view_628 = None
        permute_429: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_629, [0, 2, 1, 3]);  view_629 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_40 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_429, getitem_494, getitem_495, None, False);  permute_429 = getitem_494 = getitem_495 = None
        getitem_496: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_40[0];  _scaled_dot_product_efficient_attention_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_434: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_496, [0, 2, 1, 3]);  getitem_496 = None
        view_635: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_434, [8, 196, 320]);  permute_434 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_636: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_635, [1568, 320]);  view_635 = None
        permute_435: "f32[320, 320]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        
        # No stacktrace found for following nodes
        mm_default_47: "f32[1568, 320]" = torch.ops.aten.mm.default(view_636, permute_435);  view_636 = permute_435 = None
        add_tensor_47: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_47, arg246_1);  mm_default_47 = arg246_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_637: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_47, [8, 196, 320]);  add_tensor_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_381: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_376, view_637);  add_376 = view_637 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_127 = torch.ops.aten.var_mean.correction(add_381, [2], correction = 0, keepdim = True)
        getitem_500: "f32[8, 196, 1]" = var_mean_127[0]
        getitem_501: "f32[8, 196, 1]" = var_mean_127[1];  var_mean_127 = None
        sub_127: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_381, getitem_501);  getitem_501 = None
        add_382: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_500, 1e-06);  getitem_500 = None
        rsqrt_127: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_382);  add_382 = None
        mul_374: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_127);  sub_127 = rsqrt_127 = None
        mul_375: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_374, arg247_1);  mul_374 = arg247_1 = None
        add_383: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_375, arg248_1);  mul_375 = arg248_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_638: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_383, [1568, 320]);  add_383 = None
        permute_436: "f32[320, 1280]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        
        # No stacktrace found for following nodes
        mm_default_46: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_638, permute_436);  view_638 = permute_436 = None
        add_tensor_46: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_46, arg250_1);  mm_default_46 = arg250_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_639: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_46, [8, 196, 1280]);  add_tensor_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_376: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_639, 0.5)
        mul_377: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_639, 0.7071067811865476);  view_639 = None
        erf_40: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_377);  mul_377 = None
        add_384: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_378: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_376, add_384);  mul_376 = add_384 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_640: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_378, [1568, 1280]);  mul_378 = None
        permute_437: "f32[1280, 320]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        
        # No stacktrace found for following nodes
        mm_default_45: "f32[1568, 320]" = torch.ops.aten.mm.default(view_640, permute_437);  view_640 = permute_437 = None
        add_tensor_45: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_45, arg252_1);  mm_default_45 = arg252_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_641: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_45, [8, 196, 320]);  add_tensor_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_385: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_381, view_641);  add_381 = view_641 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_128 = torch.ops.aten.var_mean.correction(add_385, [2], correction = 0, keepdim = True)
        getitem_502: "f32[8, 196, 1]" = var_mean_128[0]
        getitem_503: "f32[8, 196, 1]" = var_mean_128[1];  var_mean_128 = None
        sub_128: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_385, getitem_503);  getitem_503 = None
        add_386: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_502, 1e-06);  getitem_502 = None
        rsqrt_128: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_386);  add_386 = None
        mul_379: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_128);  sub_128 = rsqrt_128 = None
        mul_380: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_379, arg253_1);  mul_379 = arg253_1 = None
        add_387: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_380, arg254_1);  mul_380 = arg254_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_440: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_387, [0, 2, 1])
        view_645: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_440, [8, 320, 14, 14]);  permute_440 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_52: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_645, arg257_1, arg258_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_645 = arg257_1 = arg258_1 = None
        view_646: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_52, [8, 320, 49]);  convolution_52 = None
        permute_441: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_646, [0, 2, 1]);  view_646 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_129 = torch.ops.aten.var_mean.correction(permute_441, [2], correction = 0, keepdim = True)
        getitem_504: "f32[8, 49, 1]" = var_mean_129[0]
        getitem_505: "f32[8, 49, 1]" = var_mean_129[1];  var_mean_129 = None
        sub_129: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_441, getitem_505);  permute_441 = getitem_505 = None
        add_388: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_504, 1e-05);  getitem_504 = None
        rsqrt_129: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_388);  add_388 = None
        mul_381: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_129, rsqrt_129);  sub_129 = rsqrt_129 = None
        mul_382: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_381, arg259_1);  mul_381 = arg259_1 = None
        add_389: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_382, arg260_1);  mul_382 = arg260_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_647: "f32[392, 320]" = torch.ops.aten.reshape.default(add_389, [392, 320]);  add_389 = None
        permute_442: "f32[320, 640]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_207: "f32[392, 640]" = torch.ops.aten.addmm.default(arg262_1, view_647, permute_442);  arg262_1 = view_647 = permute_442 = None
        view_648: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_207, [8, 49, 640]);  addmm_207 = None
        view_649: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_648, [8, -1, 2, 5, 64]);  view_648 = None
        permute_443: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_649, [2, 0, 3, 1, 4]);  view_649 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_41 = torch.ops.aten.unbind.int(permute_443);  permute_443 = None
        getitem_506: "f32[8, 5, 49, 64]" = unbind_41[0]
        getitem_507: "f32[8, 5, 49, 64]" = unbind_41[1];  unbind_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_642: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_387, [1568, 320]);  add_387 = None
        permute_438: "f32[320, 320]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_206: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg256_1, view_642, permute_438);  arg256_1 = view_642 = permute_438 = None
        view_643: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_206, [8, 196, 320]);  addmm_206 = None
        view_644: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_643, [8, 196, 5, 64]);  view_643 = None
        permute_439: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_644, [0, 2, 1, 3]);  view_644 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_41 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_439, getitem_506, getitem_507, None, False);  permute_439 = getitem_506 = getitem_507 = None
        getitem_508: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_41[0];  _scaled_dot_product_efficient_attention_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_444: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_508, [0, 2, 1, 3]);  getitem_508 = None
        view_650: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_444, [8, 196, 320]);  permute_444 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_651: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_650, [1568, 320]);  view_650 = None
        permute_445: "f32[320, 320]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        
        # No stacktrace found for following nodes
        mm_default_44: "f32[1568, 320]" = torch.ops.aten.mm.default(view_651, permute_445);  view_651 = permute_445 = None
        add_tensor_44: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_44, arg264_1);  mm_default_44 = arg264_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_652: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_44, [8, 196, 320]);  add_tensor_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_390: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_385, view_652);  add_385 = view_652 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_130 = torch.ops.aten.var_mean.correction(add_390, [2], correction = 0, keepdim = True)
        getitem_512: "f32[8, 196, 1]" = var_mean_130[0]
        getitem_513: "f32[8, 196, 1]" = var_mean_130[1];  var_mean_130 = None
        sub_130: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_390, getitem_513);  getitem_513 = None
        add_391: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_512, 1e-06);  getitem_512 = None
        rsqrt_130: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_391);  add_391 = None
        mul_383: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_130);  sub_130 = rsqrt_130 = None
        mul_384: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_383, arg265_1);  mul_383 = arg265_1 = None
        add_392: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_384, arg266_1);  mul_384 = arg266_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_653: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_392, [1568, 320]);  add_392 = None
        permute_446: "f32[320, 1280]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        
        # No stacktrace found for following nodes
        mm_default_43: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_653, permute_446);  view_653 = permute_446 = None
        add_tensor_43: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_43, arg268_1);  mm_default_43 = arg268_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_654: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_43, [8, 196, 1280]);  add_tensor_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_385: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_654, 0.5)
        mul_386: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_654, 0.7071067811865476);  view_654 = None
        erf_41: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_386);  mul_386 = None
        add_393: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_387: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_385, add_393);  mul_385 = add_393 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_655: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_387, [1568, 1280]);  mul_387 = None
        permute_447: "f32[1280, 320]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        
        # No stacktrace found for following nodes
        mm_default_42: "f32[1568, 320]" = torch.ops.aten.mm.default(view_655, permute_447);  view_655 = permute_447 = None
        add_tensor_42: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_42, arg270_1);  mm_default_42 = arg270_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_656: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_42, [8, 196, 320]);  add_tensor_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_394: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_390, view_656);  add_390 = view_656 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_131 = torch.ops.aten.var_mean.correction(add_394, [2], correction = 0, keepdim = True)
        getitem_514: "f32[8, 196, 1]" = var_mean_131[0]
        getitem_515: "f32[8, 196, 1]" = var_mean_131[1];  var_mean_131 = None
        sub_131: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_394, getitem_515);  getitem_515 = None
        add_395: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_514, 1e-06);  getitem_514 = None
        rsqrt_131: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_395);  add_395 = None
        mul_388: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_131);  sub_131 = rsqrt_131 = None
        mul_389: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_388, arg271_1);  mul_388 = arg271_1 = None
        add_396: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_389, arg272_1);  mul_389 = arg272_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_450: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_396, [0, 2, 1])
        view_660: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_450, [8, 320, 14, 14]);  permute_450 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_53: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_660, arg275_1, arg276_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_660 = arg275_1 = arg276_1 = None
        view_661: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_53, [8, 320, 49]);  convolution_53 = None
        permute_451: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_661, [0, 2, 1]);  view_661 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_132 = torch.ops.aten.var_mean.correction(permute_451, [2], correction = 0, keepdim = True)
        getitem_516: "f32[8, 49, 1]" = var_mean_132[0]
        getitem_517: "f32[8, 49, 1]" = var_mean_132[1];  var_mean_132 = None
        sub_132: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_451, getitem_517);  permute_451 = getitem_517 = None
        add_397: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_516, 1e-05);  getitem_516 = None
        rsqrt_132: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_397);  add_397 = None
        mul_390: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_132, rsqrt_132);  sub_132 = rsqrt_132 = None
        mul_391: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_390, arg277_1);  mul_390 = arg277_1 = None
        add_398: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_391, arg278_1);  mul_391 = arg278_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_662: "f32[392, 320]" = torch.ops.aten.reshape.default(add_398, [392, 320]);  add_398 = None
        permute_452: "f32[320, 640]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_212: "f32[392, 640]" = torch.ops.aten.addmm.default(arg280_1, view_662, permute_452);  arg280_1 = view_662 = permute_452 = None
        view_663: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_212, [8, 49, 640]);  addmm_212 = None
        view_664: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_663, [8, -1, 2, 5, 64]);  view_663 = None
        permute_453: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_664, [2, 0, 3, 1, 4]);  view_664 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_42 = torch.ops.aten.unbind.int(permute_453);  permute_453 = None
        getitem_518: "f32[8, 5, 49, 64]" = unbind_42[0]
        getitem_519: "f32[8, 5, 49, 64]" = unbind_42[1];  unbind_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_657: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_396, [1568, 320]);  add_396 = None
        permute_448: "f32[320, 320]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_211: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg274_1, view_657, permute_448);  arg274_1 = view_657 = permute_448 = None
        view_658: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_211, [8, 196, 320]);  addmm_211 = None
        view_659: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_658, [8, 196, 5, 64]);  view_658 = None
        permute_449: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_659, [0, 2, 1, 3]);  view_659 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_42 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_449, getitem_518, getitem_519, None, False);  permute_449 = getitem_518 = getitem_519 = None
        getitem_520: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_42[0];  _scaled_dot_product_efficient_attention_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_454: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_520, [0, 2, 1, 3]);  getitem_520 = None
        view_665: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_454, [8, 196, 320]);  permute_454 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_666: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_665, [1568, 320]);  view_665 = None
        permute_455: "f32[320, 320]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        
        # No stacktrace found for following nodes
        mm_default_41: "f32[1568, 320]" = torch.ops.aten.mm.default(view_666, permute_455);  view_666 = permute_455 = None
        add_tensor_41: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_41, arg282_1);  mm_default_41 = arg282_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_667: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_41, [8, 196, 320]);  add_tensor_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_399: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_394, view_667);  add_394 = view_667 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_133 = torch.ops.aten.var_mean.correction(add_399, [2], correction = 0, keepdim = True)
        getitem_524: "f32[8, 196, 1]" = var_mean_133[0]
        getitem_525: "f32[8, 196, 1]" = var_mean_133[1];  var_mean_133 = None
        sub_133: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_399, getitem_525);  getitem_525 = None
        add_400: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_524, 1e-06);  getitem_524 = None
        rsqrt_133: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_400);  add_400 = None
        mul_392: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_133);  sub_133 = rsqrt_133 = None
        mul_393: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_392, arg283_1);  mul_392 = arg283_1 = None
        add_401: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_393, arg284_1);  mul_393 = arg284_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_668: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_401, [1568, 320]);  add_401 = None
        permute_456: "f32[320, 1280]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        
        # No stacktrace found for following nodes
        mm_default_40: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_668, permute_456);  view_668 = permute_456 = None
        add_tensor_40: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_40, arg286_1);  mm_default_40 = arg286_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_669: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_40, [8, 196, 1280]);  add_tensor_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_394: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_669, 0.5)
        mul_395: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_669, 0.7071067811865476);  view_669 = None
        erf_42: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_395);  mul_395 = None
        add_402: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_396: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_394, add_402);  mul_394 = add_402 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_670: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_396, [1568, 1280]);  mul_396 = None
        permute_457: "f32[1280, 320]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        
        # No stacktrace found for following nodes
        mm_default_39: "f32[1568, 320]" = torch.ops.aten.mm.default(view_670, permute_457);  view_670 = permute_457 = None
        add_tensor_39: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_39, arg288_1);  mm_default_39 = arg288_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_671: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_39, [8, 196, 320]);  add_tensor_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_403: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_399, view_671);  add_399 = view_671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_134 = torch.ops.aten.var_mean.correction(add_403, [2], correction = 0, keepdim = True)
        getitem_526: "f32[8, 196, 1]" = var_mean_134[0]
        getitem_527: "f32[8, 196, 1]" = var_mean_134[1];  var_mean_134 = None
        sub_134: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_403, getitem_527);  getitem_527 = None
        add_404: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_526, 1e-06);  getitem_526 = None
        rsqrt_134: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_404);  add_404 = None
        mul_397: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_134);  sub_134 = rsqrt_134 = None
        mul_398: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_397, arg289_1);  mul_397 = arg289_1 = None
        add_405: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_398, arg290_1);  mul_398 = arg290_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_460: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_405, [0, 2, 1])
        view_675: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_460, [8, 320, 14, 14]);  permute_460 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_54: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_675, arg293_1, arg294_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_675 = arg293_1 = arg294_1 = None
        view_676: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_54, [8, 320, 49]);  convolution_54 = None
        permute_461: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_676, [0, 2, 1]);  view_676 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_135 = torch.ops.aten.var_mean.correction(permute_461, [2], correction = 0, keepdim = True)
        getitem_528: "f32[8, 49, 1]" = var_mean_135[0]
        getitem_529: "f32[8, 49, 1]" = var_mean_135[1];  var_mean_135 = None
        sub_135: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_461, getitem_529);  permute_461 = getitem_529 = None
        add_406: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_528, 1e-05);  getitem_528 = None
        rsqrt_135: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_406);  add_406 = None
        mul_399: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_135, rsqrt_135);  sub_135 = rsqrt_135 = None
        mul_400: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_399, arg295_1);  mul_399 = arg295_1 = None
        add_407: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_400, arg296_1);  mul_400 = arg296_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_677: "f32[392, 320]" = torch.ops.aten.reshape.default(add_407, [392, 320]);  add_407 = None
        permute_462: "f32[320, 640]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_217: "f32[392, 640]" = torch.ops.aten.addmm.default(arg298_1, view_677, permute_462);  arg298_1 = view_677 = permute_462 = None
        view_678: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_217, [8, 49, 640]);  addmm_217 = None
        view_679: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_678, [8, -1, 2, 5, 64]);  view_678 = None
        permute_463: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_679, [2, 0, 3, 1, 4]);  view_679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_43 = torch.ops.aten.unbind.int(permute_463);  permute_463 = None
        getitem_530: "f32[8, 5, 49, 64]" = unbind_43[0]
        getitem_531: "f32[8, 5, 49, 64]" = unbind_43[1];  unbind_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_672: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_405, [1568, 320]);  add_405 = None
        permute_458: "f32[320, 320]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        addmm_216: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg292_1, view_672, permute_458);  arg292_1 = view_672 = permute_458 = None
        view_673: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_216, [8, 196, 320]);  addmm_216 = None
        view_674: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_673, [8, 196, 5, 64]);  view_673 = None
        permute_459: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_674, [0, 2, 1, 3]);  view_674 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_43 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_459, getitem_530, getitem_531, None, False);  permute_459 = getitem_530 = getitem_531 = None
        getitem_532: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_43[0];  _scaled_dot_product_efficient_attention_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_464: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_532, [0, 2, 1, 3]);  getitem_532 = None
        view_680: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_464, [8, 196, 320]);  permute_464 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_681: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_680, [1568, 320]);  view_680 = None
        permute_465: "f32[320, 320]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        
        # No stacktrace found for following nodes
        mm_default_38: "f32[1568, 320]" = torch.ops.aten.mm.default(view_681, permute_465);  view_681 = permute_465 = None
        add_tensor_38: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_38, arg300_1);  mm_default_38 = arg300_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_682: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_38, [8, 196, 320]);  add_tensor_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_408: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_403, view_682);  add_403 = view_682 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_136 = torch.ops.aten.var_mean.correction(add_408, [2], correction = 0, keepdim = True)
        getitem_536: "f32[8, 196, 1]" = var_mean_136[0]
        getitem_537: "f32[8, 196, 1]" = var_mean_136[1];  var_mean_136 = None
        sub_136: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_408, getitem_537);  getitem_537 = None
        add_409: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_536, 1e-06);  getitem_536 = None
        rsqrt_136: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_409);  add_409 = None
        mul_401: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_136, rsqrt_136);  sub_136 = rsqrt_136 = None
        mul_402: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_401, arg301_1);  mul_401 = arg301_1 = None
        add_410: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_402, arg302_1);  mul_402 = arg302_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_683: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_410, [1568, 320]);  add_410 = None
        permute_466: "f32[320, 1280]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        
        # No stacktrace found for following nodes
        mm_default_37: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_683, permute_466);  view_683 = permute_466 = None
        add_tensor_37: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_37, arg304_1);  mm_default_37 = arg304_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_684: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_37, [8, 196, 1280]);  add_tensor_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_403: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_684, 0.5)
        mul_404: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_684, 0.7071067811865476);  view_684 = None
        erf_43: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_404);  mul_404 = None
        add_411: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_405: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_403, add_411);  mul_403 = add_411 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_685: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_405, [1568, 1280]);  mul_405 = None
        permute_467: "f32[1280, 320]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        
        # No stacktrace found for following nodes
        mm_default_36: "f32[1568, 320]" = torch.ops.aten.mm.default(view_685, permute_467);  view_685 = permute_467 = None
        add_tensor_36: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_36, arg306_1);  mm_default_36 = arg306_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_686: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_36, [8, 196, 320]);  add_tensor_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_412: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_408, view_686);  add_408 = view_686 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_137 = torch.ops.aten.var_mean.correction(add_412, [2], correction = 0, keepdim = True)
        getitem_538: "f32[8, 196, 1]" = var_mean_137[0]
        getitem_539: "f32[8, 196, 1]" = var_mean_137[1];  var_mean_137 = None
        sub_137: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_412, getitem_539);  getitem_539 = None
        add_413: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_538, 1e-06);  getitem_538 = None
        rsqrt_137: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_413);  add_413 = None
        mul_406: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_137);  sub_137 = rsqrt_137 = None
        mul_407: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_406, arg307_1);  mul_406 = arg307_1 = None
        add_414: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_407, arg308_1);  mul_407 = arg308_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_470: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_414, [0, 2, 1])
        view_690: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_470, [8, 320, 14, 14]);  permute_470 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_55: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_690, arg311_1, arg312_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_690 = arg311_1 = arg312_1 = None
        view_691: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_55, [8, 320, 49]);  convolution_55 = None
        permute_471: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_691, [0, 2, 1]);  view_691 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_138 = torch.ops.aten.var_mean.correction(permute_471, [2], correction = 0, keepdim = True)
        getitem_540: "f32[8, 49, 1]" = var_mean_138[0]
        getitem_541: "f32[8, 49, 1]" = var_mean_138[1];  var_mean_138 = None
        sub_138: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_471, getitem_541);  permute_471 = getitem_541 = None
        add_415: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_540, 1e-05);  getitem_540 = None
        rsqrt_138: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_415);  add_415 = None
        mul_408: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_138, rsqrt_138);  sub_138 = rsqrt_138 = None
        mul_409: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_408, arg313_1);  mul_408 = arg313_1 = None
        add_416: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_409, arg314_1);  mul_409 = arg314_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_692: "f32[392, 320]" = torch.ops.aten.reshape.default(add_416, [392, 320]);  add_416 = None
        permute_472: "f32[320, 640]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_222: "f32[392, 640]" = torch.ops.aten.addmm.default(arg316_1, view_692, permute_472);  arg316_1 = view_692 = permute_472 = None
        view_693: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_222, [8, 49, 640]);  addmm_222 = None
        view_694: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_693, [8, -1, 2, 5, 64]);  view_693 = None
        permute_473: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_694, [2, 0, 3, 1, 4]);  view_694 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_44 = torch.ops.aten.unbind.int(permute_473);  permute_473 = None
        getitem_542: "f32[8, 5, 49, 64]" = unbind_44[0]
        getitem_543: "f32[8, 5, 49, 64]" = unbind_44[1];  unbind_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_687: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_414, [1568, 320]);  add_414 = None
        permute_468: "f32[320, 320]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_221: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg310_1, view_687, permute_468);  arg310_1 = view_687 = permute_468 = None
        view_688: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_221, [8, 196, 320]);  addmm_221 = None
        view_689: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_688, [8, 196, 5, 64]);  view_688 = None
        permute_469: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_689, [0, 2, 1, 3]);  view_689 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_44 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_469, getitem_542, getitem_543, None, False);  permute_469 = getitem_542 = getitem_543 = None
        getitem_544: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_44[0];  _scaled_dot_product_efficient_attention_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_474: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_544, [0, 2, 1, 3]);  getitem_544 = None
        view_695: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_474, [8, 196, 320]);  permute_474 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_696: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_695, [1568, 320]);  view_695 = None
        permute_475: "f32[320, 320]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        
        # No stacktrace found for following nodes
        mm_default_35: "f32[1568, 320]" = torch.ops.aten.mm.default(view_696, permute_475);  view_696 = permute_475 = None
        add_tensor_35: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_35, arg318_1);  mm_default_35 = arg318_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_697: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_35, [8, 196, 320]);  add_tensor_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_417: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_412, view_697);  add_412 = view_697 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_139 = torch.ops.aten.var_mean.correction(add_417, [2], correction = 0, keepdim = True)
        getitem_548: "f32[8, 196, 1]" = var_mean_139[0]
        getitem_549: "f32[8, 196, 1]" = var_mean_139[1];  var_mean_139 = None
        sub_139: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_417, getitem_549);  getitem_549 = None
        add_418: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_548, 1e-06);  getitem_548 = None
        rsqrt_139: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_418);  add_418 = None
        mul_410: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_139, rsqrt_139);  sub_139 = rsqrt_139 = None
        mul_411: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_410, arg319_1);  mul_410 = arg319_1 = None
        add_419: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_411, arg320_1);  mul_411 = arg320_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_698: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_419, [1568, 320]);  add_419 = None
        permute_476: "f32[320, 1280]" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        
        # No stacktrace found for following nodes
        mm_default_34: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_698, permute_476);  view_698 = permute_476 = None
        add_tensor_34: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_34, arg322_1);  mm_default_34 = arg322_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_699: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_34, [8, 196, 1280]);  add_tensor_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_412: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_699, 0.5)
        mul_413: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_699, 0.7071067811865476);  view_699 = None
        erf_44: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_413);  mul_413 = None
        add_420: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_414: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_412, add_420);  mul_412 = add_420 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_700: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_414, [1568, 1280]);  mul_414 = None
        permute_477: "f32[1280, 320]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        
        # No stacktrace found for following nodes
        mm_default_33: "f32[1568, 320]" = torch.ops.aten.mm.default(view_700, permute_477);  view_700 = permute_477 = None
        add_tensor_33: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_33, arg324_1);  mm_default_33 = arg324_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_701: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 196, 320]);  add_tensor_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_421: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_417, view_701);  add_417 = view_701 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_140 = torch.ops.aten.var_mean.correction(add_421, [2], correction = 0, keepdim = True)
        getitem_550: "f32[8, 196, 1]" = var_mean_140[0]
        getitem_551: "f32[8, 196, 1]" = var_mean_140[1];  var_mean_140 = None
        sub_140: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_421, getitem_551);  getitem_551 = None
        add_422: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_550, 1e-06);  getitem_550 = None
        rsqrt_140: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_422);  add_422 = None
        mul_415: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_140);  sub_140 = rsqrt_140 = None
        mul_416: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_415, arg325_1);  mul_415 = arg325_1 = None
        add_423: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_416, arg326_1);  mul_416 = arg326_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_480: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_423, [0, 2, 1])
        view_705: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_480, [8, 320, 14, 14]);  permute_480 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_56: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_705, arg329_1, arg330_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_705 = arg329_1 = arg330_1 = None
        view_706: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_56, [8, 320, 49]);  convolution_56 = None
        permute_481: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_706, [0, 2, 1]);  view_706 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_141 = torch.ops.aten.var_mean.correction(permute_481, [2], correction = 0, keepdim = True)
        getitem_552: "f32[8, 49, 1]" = var_mean_141[0]
        getitem_553: "f32[8, 49, 1]" = var_mean_141[1];  var_mean_141 = None
        sub_141: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_481, getitem_553);  permute_481 = getitem_553 = None
        add_424: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_552, 1e-05);  getitem_552 = None
        rsqrt_141: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_424);  add_424 = None
        mul_417: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_141, rsqrt_141);  sub_141 = rsqrt_141 = None
        mul_418: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_417, arg331_1);  mul_417 = arg331_1 = None
        add_425: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_418, arg332_1);  mul_418 = arg332_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_707: "f32[392, 320]" = torch.ops.aten.reshape.default(add_425, [392, 320]);  add_425 = None
        permute_482: "f32[320, 640]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        addmm_227: "f32[392, 640]" = torch.ops.aten.addmm.default(arg334_1, view_707, permute_482);  arg334_1 = view_707 = permute_482 = None
        view_708: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_227, [8, 49, 640]);  addmm_227 = None
        view_709: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_708, [8, -1, 2, 5, 64]);  view_708 = None
        permute_483: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_709, [2, 0, 3, 1, 4]);  view_709 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_45 = torch.ops.aten.unbind.int(permute_483);  permute_483 = None
        getitem_554: "f32[8, 5, 49, 64]" = unbind_45[0]
        getitem_555: "f32[8, 5, 49, 64]" = unbind_45[1];  unbind_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_702: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_423, [1568, 320]);  add_423 = None
        permute_478: "f32[320, 320]" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_226: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg328_1, view_702, permute_478);  arg328_1 = view_702 = permute_478 = None
        view_703: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_226, [8, 196, 320]);  addmm_226 = None
        view_704: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_703, [8, 196, 5, 64]);  view_703 = None
        permute_479: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_704, [0, 2, 1, 3]);  view_704 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_45 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_479, getitem_554, getitem_555, None, False);  permute_479 = getitem_554 = getitem_555 = None
        getitem_556: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_45[0];  _scaled_dot_product_efficient_attention_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_484: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_556, [0, 2, 1, 3]);  getitem_556 = None
        view_710: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_484, [8, 196, 320]);  permute_484 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_711: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_710, [1568, 320]);  view_710 = None
        permute_485: "f32[320, 320]" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        
        # No stacktrace found for following nodes
        mm_default_32: "f32[1568, 320]" = torch.ops.aten.mm.default(view_711, permute_485);  view_711 = permute_485 = None
        add_tensor_32: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_32, arg336_1);  mm_default_32 = arg336_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_712: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 196, 320]);  add_tensor_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_426: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_421, view_712);  add_421 = view_712 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_142 = torch.ops.aten.var_mean.correction(add_426, [2], correction = 0, keepdim = True)
        getitem_560: "f32[8, 196, 1]" = var_mean_142[0]
        getitem_561: "f32[8, 196, 1]" = var_mean_142[1];  var_mean_142 = None
        sub_142: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_426, getitem_561);  getitem_561 = None
        add_427: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_560, 1e-06);  getitem_560 = None
        rsqrt_142: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_427);  add_427 = None
        mul_419: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_142, rsqrt_142);  sub_142 = rsqrt_142 = None
        mul_420: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_419, arg337_1);  mul_419 = arg337_1 = None
        add_428: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_420, arg338_1);  mul_420 = arg338_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_713: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_428, [1568, 320]);  add_428 = None
        permute_486: "f32[320, 1280]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        
        # No stacktrace found for following nodes
        mm_default_31: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_713, permute_486);  view_713 = permute_486 = None
        add_tensor_31: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_31, arg340_1);  mm_default_31 = arg340_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_714: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 196, 1280]);  add_tensor_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_421: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_714, 0.5)
        mul_422: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_714, 0.7071067811865476);  view_714 = None
        erf_45: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_422);  mul_422 = None
        add_429: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_423: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_421, add_429);  mul_421 = add_429 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_715: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_423, [1568, 1280]);  mul_423 = None
        permute_487: "f32[1280, 320]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        
        # No stacktrace found for following nodes
        mm_default_30: "f32[1568, 320]" = torch.ops.aten.mm.default(view_715, permute_487);  view_715 = permute_487 = None
        add_tensor_30: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_30, arg342_1);  mm_default_30 = arg342_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_716: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 196, 320]);  add_tensor_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_430: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_426, view_716);  add_426 = view_716 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_143 = torch.ops.aten.var_mean.correction(add_430, [2], correction = 0, keepdim = True)
        getitem_562: "f32[8, 196, 1]" = var_mean_143[0]
        getitem_563: "f32[8, 196, 1]" = var_mean_143[1];  var_mean_143 = None
        sub_143: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_430, getitem_563);  getitem_563 = None
        add_431: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_562, 1e-06);  getitem_562 = None
        rsqrt_143: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_431);  add_431 = None
        mul_424: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_143, rsqrt_143);  sub_143 = rsqrt_143 = None
        mul_425: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_424, arg343_1);  mul_424 = arg343_1 = None
        add_432: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_425, arg344_1);  mul_425 = arg344_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_490: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_432, [0, 2, 1])
        view_720: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_490, [8, 320, 14, 14]);  permute_490 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_57: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_720, arg347_1, arg348_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_720 = arg347_1 = arg348_1 = None
        view_721: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_57, [8, 320, 49]);  convolution_57 = None
        permute_491: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_721, [0, 2, 1]);  view_721 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_144 = torch.ops.aten.var_mean.correction(permute_491, [2], correction = 0, keepdim = True)
        getitem_564: "f32[8, 49, 1]" = var_mean_144[0]
        getitem_565: "f32[8, 49, 1]" = var_mean_144[1];  var_mean_144 = None
        sub_144: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_491, getitem_565);  permute_491 = getitem_565 = None
        add_433: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_564, 1e-05);  getitem_564 = None
        rsqrt_144: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_433);  add_433 = None
        mul_426: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_144, rsqrt_144);  sub_144 = rsqrt_144 = None
        mul_427: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_426, arg349_1);  mul_426 = arg349_1 = None
        add_434: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_427, arg350_1);  mul_427 = arg350_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_722: "f32[392, 320]" = torch.ops.aten.reshape.default(add_434, [392, 320]);  add_434 = None
        permute_492: "f32[320, 640]" = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        addmm_232: "f32[392, 640]" = torch.ops.aten.addmm.default(arg352_1, view_722, permute_492);  arg352_1 = view_722 = permute_492 = None
        view_723: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_232, [8, 49, 640]);  addmm_232 = None
        view_724: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_723, [8, -1, 2, 5, 64]);  view_723 = None
        permute_493: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_724, [2, 0, 3, 1, 4]);  view_724 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_46 = torch.ops.aten.unbind.int(permute_493);  permute_493 = None
        getitem_566: "f32[8, 5, 49, 64]" = unbind_46[0]
        getitem_567: "f32[8, 5, 49, 64]" = unbind_46[1];  unbind_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_717: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_432, [1568, 320]);  add_432 = None
        permute_488: "f32[320, 320]" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_231: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg346_1, view_717, permute_488);  arg346_1 = view_717 = permute_488 = None
        view_718: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_231, [8, 196, 320]);  addmm_231 = None
        view_719: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_718, [8, 196, 5, 64]);  view_718 = None
        permute_489: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_719, [0, 2, 1, 3]);  view_719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_46 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_489, getitem_566, getitem_567, None, False);  permute_489 = getitem_566 = getitem_567 = None
        getitem_568: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_46[0];  _scaled_dot_product_efficient_attention_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_494: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_568, [0, 2, 1, 3]);  getitem_568 = None
        view_725: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_494, [8, 196, 320]);  permute_494 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_726: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_725, [1568, 320]);  view_725 = None
        permute_495: "f32[320, 320]" = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        
        # No stacktrace found for following nodes
        mm_default_29: "f32[1568, 320]" = torch.ops.aten.mm.default(view_726, permute_495);  view_726 = permute_495 = None
        add_tensor_29: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_29, arg354_1);  mm_default_29 = arg354_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_727: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_29, [8, 196, 320]);  add_tensor_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_435: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_430, view_727);  add_430 = view_727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_145 = torch.ops.aten.var_mean.correction(add_435, [2], correction = 0, keepdim = True)
        getitem_572: "f32[8, 196, 1]" = var_mean_145[0]
        getitem_573: "f32[8, 196, 1]" = var_mean_145[1];  var_mean_145 = None
        sub_145: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_435, getitem_573);  getitem_573 = None
        add_436: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_572, 1e-06);  getitem_572 = None
        rsqrt_145: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_436);  add_436 = None
        mul_428: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_145);  sub_145 = rsqrt_145 = None
        mul_429: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_428, arg355_1);  mul_428 = arg355_1 = None
        add_437: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_429, arg356_1);  mul_429 = arg356_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_728: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_437, [1568, 320]);  add_437 = None
        permute_496: "f32[320, 1280]" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        
        # No stacktrace found for following nodes
        mm_default_28: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_728, permute_496);  view_728 = permute_496 = None
        add_tensor_28: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_28, arg358_1);  mm_default_28 = arg358_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_729: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_28, [8, 196, 1280]);  add_tensor_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_430: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_729, 0.5)
        mul_431: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_729, 0.7071067811865476);  view_729 = None
        erf_46: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_431);  mul_431 = None
        add_438: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_432: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_430, add_438);  mul_430 = add_438 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_730: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_432, [1568, 1280]);  mul_432 = None
        permute_497: "f32[1280, 320]" = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        
        # No stacktrace found for following nodes
        mm_default_27: "f32[1568, 320]" = torch.ops.aten.mm.default(view_730, permute_497);  view_730 = permute_497 = None
        add_tensor_27: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_27, arg360_1);  mm_default_27 = arg360_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_731: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_27, [8, 196, 320]);  add_tensor_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_439: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_435, view_731);  add_435 = view_731 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_146 = torch.ops.aten.var_mean.correction(add_439, [2], correction = 0, keepdim = True)
        getitem_574: "f32[8, 196, 1]" = var_mean_146[0]
        getitem_575: "f32[8, 196, 1]" = var_mean_146[1];  var_mean_146 = None
        sub_146: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_439, getitem_575);  getitem_575 = None
        add_440: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_574, 1e-06);  getitem_574 = None
        rsqrt_146: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_440);  add_440 = None
        mul_433: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_146, rsqrt_146);  sub_146 = rsqrt_146 = None
        mul_434: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_433, arg361_1);  mul_433 = arg361_1 = None
        add_441: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_434, arg362_1);  mul_434 = arg362_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_500: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_441, [0, 2, 1])
        view_735: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_500, [8, 320, 14, 14]);  permute_500 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_58: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_735, arg365_1, arg366_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_735 = arg365_1 = arg366_1 = None
        view_736: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_58, [8, 320, 49]);  convolution_58 = None
        permute_501: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_736, [0, 2, 1]);  view_736 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_147 = torch.ops.aten.var_mean.correction(permute_501, [2], correction = 0, keepdim = True)
        getitem_576: "f32[8, 49, 1]" = var_mean_147[0]
        getitem_577: "f32[8, 49, 1]" = var_mean_147[1];  var_mean_147 = None
        sub_147: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_501, getitem_577);  permute_501 = getitem_577 = None
        add_442: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_576, 1e-05);  getitem_576 = None
        rsqrt_147: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_442);  add_442 = None
        mul_435: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_147, rsqrt_147);  sub_147 = rsqrt_147 = None
        mul_436: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_435, arg367_1);  mul_435 = arg367_1 = None
        add_443: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_436, arg368_1);  mul_436 = arg368_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_737: "f32[392, 320]" = torch.ops.aten.reshape.default(add_443, [392, 320]);  add_443 = None
        permute_502: "f32[320, 640]" = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_237: "f32[392, 640]" = torch.ops.aten.addmm.default(arg370_1, view_737, permute_502);  arg370_1 = view_737 = permute_502 = None
        view_738: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_237, [8, 49, 640]);  addmm_237 = None
        view_739: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_738, [8, -1, 2, 5, 64]);  view_738 = None
        permute_503: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_739, [2, 0, 3, 1, 4]);  view_739 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_47 = torch.ops.aten.unbind.int(permute_503);  permute_503 = None
        getitem_578: "f32[8, 5, 49, 64]" = unbind_47[0]
        getitem_579: "f32[8, 5, 49, 64]" = unbind_47[1];  unbind_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_732: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_441, [1568, 320]);  add_441 = None
        permute_498: "f32[320, 320]" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_236: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg364_1, view_732, permute_498);  arg364_1 = view_732 = permute_498 = None
        view_733: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_236, [8, 196, 320]);  addmm_236 = None
        view_734: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_733, [8, 196, 5, 64]);  view_733 = None
        permute_499: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_47 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_499, getitem_578, getitem_579, None, False);  permute_499 = getitem_578 = getitem_579 = None
        getitem_580: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_47[0];  _scaled_dot_product_efficient_attention_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_504: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_580, [0, 2, 1, 3]);  getitem_580 = None
        view_740: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_504, [8, 196, 320]);  permute_504 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_741: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_740, [1568, 320]);  view_740 = None
        permute_505: "f32[320, 320]" = torch.ops.aten.permute.default(arg371_1, [1, 0]);  arg371_1 = None
        
        # No stacktrace found for following nodes
        mm_default_26: "f32[1568, 320]" = torch.ops.aten.mm.default(view_741, permute_505);  view_741 = permute_505 = None
        add_tensor_26: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_26, arg372_1);  mm_default_26 = arg372_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_742: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 196, 320]);  add_tensor_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_444: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_439, view_742);  add_439 = view_742 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_148 = torch.ops.aten.var_mean.correction(add_444, [2], correction = 0, keepdim = True)
        getitem_584: "f32[8, 196, 1]" = var_mean_148[0]
        getitem_585: "f32[8, 196, 1]" = var_mean_148[1];  var_mean_148 = None
        sub_148: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_444, getitem_585);  getitem_585 = None
        add_445: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_584, 1e-06);  getitem_584 = None
        rsqrt_148: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_445);  add_445 = None
        mul_437: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_148, rsqrt_148);  sub_148 = rsqrt_148 = None
        mul_438: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_437, arg373_1);  mul_437 = arg373_1 = None
        add_446: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_438, arg374_1);  mul_438 = arg374_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_743: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_446, [1568, 320]);  add_446 = None
        permute_506: "f32[320, 1280]" = torch.ops.aten.permute.default(arg375_1, [1, 0]);  arg375_1 = None
        
        # No stacktrace found for following nodes
        mm_default_25: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_743, permute_506);  view_743 = permute_506 = None
        add_tensor_25: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_25, arg376_1);  mm_default_25 = arg376_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_744: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 196, 1280]);  add_tensor_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_439: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_744, 0.5)
        mul_440: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_744, 0.7071067811865476);  view_744 = None
        erf_47: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_440);  mul_440 = None
        add_447: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_441: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_439, add_447);  mul_439 = add_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_745: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_441, [1568, 1280]);  mul_441 = None
        permute_507: "f32[1280, 320]" = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
        
        # No stacktrace found for following nodes
        mm_default_24: "f32[1568, 320]" = torch.ops.aten.mm.default(view_745, permute_507);  view_745 = permute_507 = None
        add_tensor_24: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_24, arg378_1);  mm_default_24 = arg378_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_746: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 196, 320]);  add_tensor_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_448: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_444, view_746);  add_444 = view_746 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_149 = torch.ops.aten.var_mean.correction(add_448, [2], correction = 0, keepdim = True)
        getitem_586: "f32[8, 196, 1]" = var_mean_149[0]
        getitem_587: "f32[8, 196, 1]" = var_mean_149[1];  var_mean_149 = None
        sub_149: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_448, getitem_587);  getitem_587 = None
        add_449: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_586, 1e-06);  getitem_586 = None
        rsqrt_149: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_449);  add_449 = None
        mul_442: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_149, rsqrt_149);  sub_149 = rsqrt_149 = None
        mul_443: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_442, arg379_1);  mul_442 = arg379_1 = None
        add_450: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_443, arg380_1);  mul_443 = arg380_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_510: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_450, [0, 2, 1])
        view_750: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_510, [8, 320, 14, 14]);  permute_510 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_59: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_750, arg383_1, arg384_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_750 = arg383_1 = arg384_1 = None
        view_751: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_59, [8, 320, 49]);  convolution_59 = None
        permute_511: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_751, [0, 2, 1]);  view_751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_150 = torch.ops.aten.var_mean.correction(permute_511, [2], correction = 0, keepdim = True)
        getitem_588: "f32[8, 49, 1]" = var_mean_150[0]
        getitem_589: "f32[8, 49, 1]" = var_mean_150[1];  var_mean_150 = None
        sub_150: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_511, getitem_589);  permute_511 = getitem_589 = None
        add_451: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_588, 1e-05);  getitem_588 = None
        rsqrt_150: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_451);  add_451 = None
        mul_444: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_150, rsqrt_150);  sub_150 = rsqrt_150 = None
        mul_445: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_444, arg385_1);  mul_444 = arg385_1 = None
        add_452: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_445, arg386_1);  mul_445 = arg386_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_752: "f32[392, 320]" = torch.ops.aten.reshape.default(add_452, [392, 320]);  add_452 = None
        permute_512: "f32[320, 640]" = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
        addmm_242: "f32[392, 640]" = torch.ops.aten.addmm.default(arg388_1, view_752, permute_512);  arg388_1 = view_752 = permute_512 = None
        view_753: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_242, [8, 49, 640]);  addmm_242 = None
        view_754: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_753, [8, -1, 2, 5, 64]);  view_753 = None
        permute_513: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_754, [2, 0, 3, 1, 4]);  view_754 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_48 = torch.ops.aten.unbind.int(permute_513);  permute_513 = None
        getitem_590: "f32[8, 5, 49, 64]" = unbind_48[0]
        getitem_591: "f32[8, 5, 49, 64]" = unbind_48[1];  unbind_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_747: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_450, [1568, 320]);  add_450 = None
        permute_508: "f32[320, 320]" = torch.ops.aten.permute.default(arg381_1, [1, 0]);  arg381_1 = None
        addmm_241: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg382_1, view_747, permute_508);  arg382_1 = view_747 = permute_508 = None
        view_748: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_241, [8, 196, 320]);  addmm_241 = None
        view_749: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_748, [8, 196, 5, 64]);  view_748 = None
        permute_509: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_749, [0, 2, 1, 3]);  view_749 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_48 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_509, getitem_590, getitem_591, None, False);  permute_509 = getitem_590 = getitem_591 = None
        getitem_592: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_48[0];  _scaled_dot_product_efficient_attention_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_514: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_592, [0, 2, 1, 3]);  getitem_592 = None
        view_755: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_514, [8, 196, 320]);  permute_514 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_756: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_755, [1568, 320]);  view_755 = None
        permute_515: "f32[320, 320]" = torch.ops.aten.permute.default(arg389_1, [1, 0]);  arg389_1 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[1568, 320]" = torch.ops.aten.mm.default(view_756, permute_515);  view_756 = permute_515 = None
        add_tensor_23: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_23, arg390_1);  mm_default_23 = arg390_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_757: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 196, 320]);  add_tensor_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_453: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_448, view_757);  add_448 = view_757 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_151 = torch.ops.aten.var_mean.correction(add_453, [2], correction = 0, keepdim = True)
        getitem_596: "f32[8, 196, 1]" = var_mean_151[0]
        getitem_597: "f32[8, 196, 1]" = var_mean_151[1];  var_mean_151 = None
        sub_151: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_453, getitem_597);  getitem_597 = None
        add_454: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_596, 1e-06);  getitem_596 = None
        rsqrt_151: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_454);  add_454 = None
        mul_446: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_151, rsqrt_151);  sub_151 = rsqrt_151 = None
        mul_447: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_446, arg391_1);  mul_446 = arg391_1 = None
        add_455: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_447, arg392_1);  mul_447 = arg392_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_758: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_455, [1568, 320]);  add_455 = None
        permute_516: "f32[320, 1280]" = torch.ops.aten.permute.default(arg393_1, [1, 0]);  arg393_1 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_758, permute_516);  view_758 = permute_516 = None
        add_tensor_22: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_22, arg394_1);  mm_default_22 = arg394_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_759: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 196, 1280]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_448: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_759, 0.5)
        mul_449: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_759, 0.7071067811865476);  view_759 = None
        erf_48: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_449);  mul_449 = None
        add_456: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
        mul_450: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_448, add_456);  mul_448 = add_456 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_760: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_450, [1568, 1280]);  mul_450 = None
        permute_517: "f32[1280, 320]" = torch.ops.aten.permute.default(arg395_1, [1, 0]);  arg395_1 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[1568, 320]" = torch.ops.aten.mm.default(view_760, permute_517);  view_760 = permute_517 = None
        add_tensor_21: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_21, arg396_1);  mm_default_21 = arg396_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_761: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 196, 320]);  add_tensor_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_457: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_453, view_761);  add_453 = view_761 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_152 = torch.ops.aten.var_mean.correction(add_457, [2], correction = 0, keepdim = True)
        getitem_598: "f32[8, 196, 1]" = var_mean_152[0]
        getitem_599: "f32[8, 196, 1]" = var_mean_152[1];  var_mean_152 = None
        sub_152: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_457, getitem_599);  getitem_599 = None
        add_458: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_598, 1e-06);  getitem_598 = None
        rsqrt_152: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_458);  add_458 = None
        mul_451: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_152, rsqrt_152);  sub_152 = rsqrt_152 = None
        mul_452: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_451, arg397_1);  mul_451 = arg397_1 = None
        add_459: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_452, arg398_1);  mul_452 = arg398_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_520: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_459, [0, 2, 1])
        view_765: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_520, [8, 320, 14, 14]);  permute_520 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_60: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_765, arg401_1, arg402_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_765 = arg401_1 = arg402_1 = None
        view_766: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_60, [8, 320, 49]);  convolution_60 = None
        permute_521: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_766, [0, 2, 1]);  view_766 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_153 = torch.ops.aten.var_mean.correction(permute_521, [2], correction = 0, keepdim = True)
        getitem_600: "f32[8, 49, 1]" = var_mean_153[0]
        getitem_601: "f32[8, 49, 1]" = var_mean_153[1];  var_mean_153 = None
        sub_153: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_521, getitem_601);  permute_521 = getitem_601 = None
        add_460: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_600, 1e-05);  getitem_600 = None
        rsqrt_153: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_460);  add_460 = None
        mul_453: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_153, rsqrt_153);  sub_153 = rsqrt_153 = None
        mul_454: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_453, arg403_1);  mul_453 = arg403_1 = None
        add_461: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_454, arg404_1);  mul_454 = arg404_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_767: "f32[392, 320]" = torch.ops.aten.reshape.default(add_461, [392, 320]);  add_461 = None
        permute_522: "f32[320, 640]" = torch.ops.aten.permute.default(arg405_1, [1, 0]);  arg405_1 = None
        addmm_247: "f32[392, 640]" = torch.ops.aten.addmm.default(arg406_1, view_767, permute_522);  arg406_1 = view_767 = permute_522 = None
        view_768: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_247, [8, 49, 640]);  addmm_247 = None
        view_769: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_768, [8, -1, 2, 5, 64]);  view_768 = None
        permute_523: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_769, [2, 0, 3, 1, 4]);  view_769 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_49 = torch.ops.aten.unbind.int(permute_523);  permute_523 = None
        getitem_602: "f32[8, 5, 49, 64]" = unbind_49[0]
        getitem_603: "f32[8, 5, 49, 64]" = unbind_49[1];  unbind_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_762: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_459, [1568, 320]);  add_459 = None
        permute_518: "f32[320, 320]" = torch.ops.aten.permute.default(arg399_1, [1, 0]);  arg399_1 = None
        addmm_246: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg400_1, view_762, permute_518);  arg400_1 = view_762 = permute_518 = None
        view_763: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_246, [8, 196, 320]);  addmm_246 = None
        view_764: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_763, [8, 196, 5, 64]);  view_763 = None
        permute_519: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_764, [0, 2, 1, 3]);  view_764 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_49 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_519, getitem_602, getitem_603, None, False);  permute_519 = getitem_602 = getitem_603 = None
        getitem_604: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_49[0];  _scaled_dot_product_efficient_attention_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_524: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_604, [0, 2, 1, 3]);  getitem_604 = None
        view_770: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_524, [8, 196, 320]);  permute_524 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_771: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_770, [1568, 320]);  view_770 = None
        permute_525: "f32[320, 320]" = torch.ops.aten.permute.default(arg407_1, [1, 0]);  arg407_1 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[1568, 320]" = torch.ops.aten.mm.default(view_771, permute_525);  view_771 = permute_525 = None
        add_tensor_20: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_20, arg408_1);  mm_default_20 = arg408_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_772: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 196, 320]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_462: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_457, view_772);  add_457 = view_772 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_154 = torch.ops.aten.var_mean.correction(add_462, [2], correction = 0, keepdim = True)
        getitem_608: "f32[8, 196, 1]" = var_mean_154[0]
        getitem_609: "f32[8, 196, 1]" = var_mean_154[1];  var_mean_154 = None
        sub_154: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_462, getitem_609);  getitem_609 = None
        add_463: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_608, 1e-06);  getitem_608 = None
        rsqrt_154: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_463);  add_463 = None
        mul_455: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_154, rsqrt_154);  sub_154 = rsqrt_154 = None
        mul_456: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_455, arg409_1);  mul_455 = arg409_1 = None
        add_464: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_456, arg410_1);  mul_456 = arg410_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_773: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_464, [1568, 320]);  add_464 = None
        permute_526: "f32[320, 1280]" = torch.ops.aten.permute.default(arg411_1, [1, 0]);  arg411_1 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_773, permute_526);  view_773 = permute_526 = None
        add_tensor_19: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_19, arg412_1);  mm_default_19 = arg412_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_774: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 196, 1280]);  add_tensor_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_457: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_774, 0.5)
        mul_458: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_774, 0.7071067811865476);  view_774 = None
        erf_49: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_458);  mul_458 = None
        add_465: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
        mul_459: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_457, add_465);  mul_457 = add_465 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_775: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_459, [1568, 1280]);  mul_459 = None
        permute_527: "f32[1280, 320]" = torch.ops.aten.permute.default(arg413_1, [1, 0]);  arg413_1 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[1568, 320]" = torch.ops.aten.mm.default(view_775, permute_527);  view_775 = permute_527 = None
        add_tensor_18: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_18, arg414_1);  mm_default_18 = arg414_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_776: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 196, 320]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_466: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_462, view_776);  add_462 = view_776 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_155 = torch.ops.aten.var_mean.correction(add_466, [2], correction = 0, keepdim = True)
        getitem_610: "f32[8, 196, 1]" = var_mean_155[0]
        getitem_611: "f32[8, 196, 1]" = var_mean_155[1];  var_mean_155 = None
        sub_155: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_466, getitem_611);  getitem_611 = None
        add_467: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_610, 1e-06);  getitem_610 = None
        rsqrt_155: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_467);  add_467 = None
        mul_460: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_155, rsqrt_155);  sub_155 = rsqrt_155 = None
        mul_461: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_460, arg415_1);  mul_460 = arg415_1 = None
        add_468: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_461, arg416_1);  mul_461 = arg416_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_530: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_468, [0, 2, 1])
        view_780: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_530, [8, 320, 14, 14]);  permute_530 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_61: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_780, arg419_1, arg420_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_780 = arg419_1 = arg420_1 = None
        view_781: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_61, [8, 320, 49]);  convolution_61 = None
        permute_531: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_781, [0, 2, 1]);  view_781 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_156 = torch.ops.aten.var_mean.correction(permute_531, [2], correction = 0, keepdim = True)
        getitem_612: "f32[8, 49, 1]" = var_mean_156[0]
        getitem_613: "f32[8, 49, 1]" = var_mean_156[1];  var_mean_156 = None
        sub_156: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_531, getitem_613);  permute_531 = getitem_613 = None
        add_469: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_612, 1e-05);  getitem_612 = None
        rsqrt_156: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_469);  add_469 = None
        mul_462: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_156, rsqrt_156);  sub_156 = rsqrt_156 = None
        mul_463: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_462, arg421_1);  mul_462 = arg421_1 = None
        add_470: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_463, arg422_1);  mul_463 = arg422_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_782: "f32[392, 320]" = torch.ops.aten.reshape.default(add_470, [392, 320]);  add_470 = None
        permute_532: "f32[320, 640]" = torch.ops.aten.permute.default(arg423_1, [1, 0]);  arg423_1 = None
        addmm_252: "f32[392, 640]" = torch.ops.aten.addmm.default(arg424_1, view_782, permute_532);  arg424_1 = view_782 = permute_532 = None
        view_783: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_252, [8, 49, 640]);  addmm_252 = None
        view_784: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_783, [8, -1, 2, 5, 64]);  view_783 = None
        permute_533: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_784, [2, 0, 3, 1, 4]);  view_784 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_50 = torch.ops.aten.unbind.int(permute_533);  permute_533 = None
        getitem_614: "f32[8, 5, 49, 64]" = unbind_50[0]
        getitem_615: "f32[8, 5, 49, 64]" = unbind_50[1];  unbind_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_777: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_468, [1568, 320]);  add_468 = None
        permute_528: "f32[320, 320]" = torch.ops.aten.permute.default(arg417_1, [1, 0]);  arg417_1 = None
        addmm_251: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg418_1, view_777, permute_528);  arg418_1 = view_777 = permute_528 = None
        view_778: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_251, [8, 196, 320]);  addmm_251 = None
        view_779: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_778, [8, 196, 5, 64]);  view_778 = None
        permute_529: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_779, [0, 2, 1, 3]);  view_779 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_50 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_529, getitem_614, getitem_615, None, False);  permute_529 = getitem_614 = getitem_615 = None
        getitem_616: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_50[0];  _scaled_dot_product_efficient_attention_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_534: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_616, [0, 2, 1, 3]);  getitem_616 = None
        view_785: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_534, [8, 196, 320]);  permute_534 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_786: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_785, [1568, 320]);  view_785 = None
        permute_535: "f32[320, 320]" = torch.ops.aten.permute.default(arg425_1, [1, 0]);  arg425_1 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[1568, 320]" = torch.ops.aten.mm.default(view_786, permute_535);  view_786 = permute_535 = None
        add_tensor_17: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_17, arg426_1);  mm_default_17 = arg426_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_787: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 196, 320]);  add_tensor_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_471: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_466, view_787);  add_466 = view_787 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_157 = torch.ops.aten.var_mean.correction(add_471, [2], correction = 0, keepdim = True)
        getitem_620: "f32[8, 196, 1]" = var_mean_157[0]
        getitem_621: "f32[8, 196, 1]" = var_mean_157[1];  var_mean_157 = None
        sub_157: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_471, getitem_621);  getitem_621 = None
        add_472: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_620, 1e-06);  getitem_620 = None
        rsqrt_157: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_472);  add_472 = None
        mul_464: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_157, rsqrt_157);  sub_157 = rsqrt_157 = None
        mul_465: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_464, arg427_1);  mul_464 = arg427_1 = None
        add_473: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_465, arg428_1);  mul_465 = arg428_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_788: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_473, [1568, 320]);  add_473 = None
        permute_536: "f32[320, 1280]" = torch.ops.aten.permute.default(arg429_1, [1, 0]);  arg429_1 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_788, permute_536);  view_788 = permute_536 = None
        add_tensor_16: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_16, arg430_1);  mm_default_16 = arg430_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_789: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 196, 1280]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_466: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_789, 0.5)
        mul_467: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_789, 0.7071067811865476);  view_789 = None
        erf_50: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_467);  mul_467 = None
        add_474: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
        mul_468: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_466, add_474);  mul_466 = add_474 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_790: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_468, [1568, 1280]);  mul_468 = None
        permute_537: "f32[1280, 320]" = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[1568, 320]" = torch.ops.aten.mm.default(view_790, permute_537);  view_790 = permute_537 = None
        add_tensor_15: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_15, arg432_1);  mm_default_15 = arg432_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_791: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 196, 320]);  add_tensor_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_475: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_471, view_791);  add_471 = view_791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_158 = torch.ops.aten.var_mean.correction(add_475, [2], correction = 0, keepdim = True)
        getitem_622: "f32[8, 196, 1]" = var_mean_158[0]
        getitem_623: "f32[8, 196, 1]" = var_mean_158[1];  var_mean_158 = None
        sub_158: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_475, getitem_623);  getitem_623 = None
        add_476: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_622, 1e-06);  getitem_622 = None
        rsqrt_158: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_476);  add_476 = None
        mul_469: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_158, rsqrt_158);  sub_158 = rsqrt_158 = None
        mul_470: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_469, arg433_1);  mul_469 = arg433_1 = None
        add_477: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_470, arg434_1);  mul_470 = arg434_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_540: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_477, [0, 2, 1])
        view_795: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_540, [8, 320, 14, 14]);  permute_540 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_62: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_795, arg437_1, arg438_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_795 = arg437_1 = arg438_1 = None
        view_796: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_62, [8, 320, 49]);  convolution_62 = None
        permute_541: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_796, [0, 2, 1]);  view_796 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_159 = torch.ops.aten.var_mean.correction(permute_541, [2], correction = 0, keepdim = True)
        getitem_624: "f32[8, 49, 1]" = var_mean_159[0]
        getitem_625: "f32[8, 49, 1]" = var_mean_159[1];  var_mean_159 = None
        sub_159: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_541, getitem_625);  permute_541 = getitem_625 = None
        add_478: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_624, 1e-05);  getitem_624 = None
        rsqrt_159: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_478);  add_478 = None
        mul_471: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_159, rsqrt_159);  sub_159 = rsqrt_159 = None
        mul_472: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_471, arg439_1);  mul_471 = arg439_1 = None
        add_479: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_472, arg440_1);  mul_472 = arg440_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_797: "f32[392, 320]" = torch.ops.aten.reshape.default(add_479, [392, 320]);  add_479 = None
        permute_542: "f32[320, 640]" = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
        addmm_257: "f32[392, 640]" = torch.ops.aten.addmm.default(arg442_1, view_797, permute_542);  arg442_1 = view_797 = permute_542 = None
        view_798: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_257, [8, 49, 640]);  addmm_257 = None
        view_799: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_798, [8, -1, 2, 5, 64]);  view_798 = None
        permute_543: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_799, [2, 0, 3, 1, 4]);  view_799 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_51 = torch.ops.aten.unbind.int(permute_543);  permute_543 = None
        getitem_626: "f32[8, 5, 49, 64]" = unbind_51[0]
        getitem_627: "f32[8, 5, 49, 64]" = unbind_51[1];  unbind_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_792: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_477, [1568, 320]);  add_477 = None
        permute_538: "f32[320, 320]" = torch.ops.aten.permute.default(arg435_1, [1, 0]);  arg435_1 = None
        addmm_256: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg436_1, view_792, permute_538);  arg436_1 = view_792 = permute_538 = None
        view_793: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_256, [8, 196, 320]);  addmm_256 = None
        view_794: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_793, [8, 196, 5, 64]);  view_793 = None
        permute_539: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_794, [0, 2, 1, 3]);  view_794 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_51 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_539, getitem_626, getitem_627, None, False);  permute_539 = getitem_626 = getitem_627 = None
        getitem_628: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_51[0];  _scaled_dot_product_efficient_attention_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_544: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_628, [0, 2, 1, 3]);  getitem_628 = None
        view_800: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_544, [8, 196, 320]);  permute_544 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_801: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_800, [1568, 320]);  view_800 = None
        permute_545: "f32[320, 320]" = torch.ops.aten.permute.default(arg443_1, [1, 0]);  arg443_1 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[1568, 320]" = torch.ops.aten.mm.default(view_801, permute_545);  view_801 = permute_545 = None
        add_tensor_14: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_14, arg444_1);  mm_default_14 = arg444_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_802: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 196, 320]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_480: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_475, view_802);  add_475 = view_802 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_160 = torch.ops.aten.var_mean.correction(add_480, [2], correction = 0, keepdim = True)
        getitem_632: "f32[8, 196, 1]" = var_mean_160[0]
        getitem_633: "f32[8, 196, 1]" = var_mean_160[1];  var_mean_160 = None
        sub_160: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_480, getitem_633);  getitem_633 = None
        add_481: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_632, 1e-06);  getitem_632 = None
        rsqrt_160: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_481);  add_481 = None
        mul_473: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_160, rsqrt_160);  sub_160 = rsqrt_160 = None
        mul_474: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_473, arg445_1);  mul_473 = arg445_1 = None
        add_482: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_474, arg446_1);  mul_474 = arg446_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_803: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_482, [1568, 320]);  add_482 = None
        permute_546: "f32[320, 1280]" = torch.ops.aten.permute.default(arg447_1, [1, 0]);  arg447_1 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_803, permute_546);  view_803 = permute_546 = None
        add_tensor_13: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_13, arg448_1);  mm_default_13 = arg448_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_804: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 196, 1280]);  add_tensor_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_475: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_804, 0.5)
        mul_476: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_804, 0.7071067811865476);  view_804 = None
        erf_51: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_476);  mul_476 = None
        add_483: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
        mul_477: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_475, add_483);  mul_475 = add_483 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_805: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_477, [1568, 1280]);  mul_477 = None
        permute_547: "f32[1280, 320]" = torch.ops.aten.permute.default(arg449_1, [1, 0]);  arg449_1 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[1568, 320]" = torch.ops.aten.mm.default(view_805, permute_547);  view_805 = permute_547 = None
        add_tensor_12: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_12, arg450_1);  mm_default_12 = arg450_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_806: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 196, 320]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_484: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_480, view_806);  add_480 = view_806 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_161 = torch.ops.aten.var_mean.correction(add_484, [2], correction = 0, keepdim = True)
        getitem_634: "f32[8, 196, 1]" = var_mean_161[0]
        getitem_635: "f32[8, 196, 1]" = var_mean_161[1];  var_mean_161 = None
        sub_161: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_484, getitem_635);  getitem_635 = None
        add_485: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_634, 1e-06);  getitem_634 = None
        rsqrt_161: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_485);  add_485 = None
        mul_478: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_161, rsqrt_161);  sub_161 = rsqrt_161 = None
        mul_479: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_478, arg451_1);  mul_478 = arg451_1 = None
        add_486: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_479, arg452_1);  mul_479 = arg452_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:167 in forward, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
        permute_550: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_486, [0, 2, 1])
        view_810: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_550, [8, 320, 14, 14]);  permute_550 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:168 in forward, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
        convolution_63: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_810, arg455_1, arg456_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  view_810 = arg455_1 = arg456_1 = None
        view_811: "f32[8, 320, 49]" = torch.ops.aten.reshape.default(convolution_63, [8, 320, 49]);  convolution_63 = None
        permute_551: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_811, [0, 2, 1]);  view_811 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:169 in forward, code: x = self.norm(x)
        var_mean_162 = torch.ops.aten.var_mean.correction(permute_551, [2], correction = 0, keepdim = True)
        getitem_636: "f32[8, 49, 1]" = var_mean_162[0]
        getitem_637: "f32[8, 49, 1]" = var_mean_162[1];  var_mean_162 = None
        sub_162: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(permute_551, getitem_637);  permute_551 = getitem_637 = None
        add_487: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_636, 1e-05);  getitem_636 = None
        rsqrt_162: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_487);  add_487 = None
        mul_480: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_162, rsqrt_162);  sub_162 = rsqrt_162 = None
        mul_481: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_480, arg457_1);  mul_480 = arg457_1 = None
        add_488: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_481, arg458_1);  mul_481 = arg458_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_812: "f32[392, 320]" = torch.ops.aten.reshape.default(add_488, [392, 320]);  add_488 = None
        permute_552: "f32[320, 640]" = torch.ops.aten.permute.default(arg459_1, [1, 0]);  arg459_1 = None
        addmm_262: "f32[392, 640]" = torch.ops.aten.addmm.default(arg460_1, view_812, permute_552);  arg460_1 = view_812 = permute_552 = None
        view_813: "f32[8, 49, 640]" = torch.ops.aten.reshape.default(addmm_262, [8, 49, 640]);  addmm_262 = None
        view_814: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.reshape.default(view_813, [8, -1, 2, 5, 64]);  view_813 = None
        permute_553: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_814, [2, 0, 3, 1, 4]);  view_814 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_52 = torch.ops.aten.unbind.int(permute_553);  permute_553 = None
        getitem_638: "f32[8, 5, 49, 64]" = unbind_52[0]
        getitem_639: "f32[8, 5, 49, 64]" = unbind_52[1];  unbind_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_807: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_486, [1568, 320]);  add_486 = None
        permute_548: "f32[320, 320]" = torch.ops.aten.permute.default(arg453_1, [1, 0]);  arg453_1 = None
        addmm_261: "f32[1568, 320]" = torch.ops.aten.addmm.default(arg454_1, view_807, permute_548);  arg454_1 = view_807 = permute_548 = None
        view_808: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(addmm_261, [8, 196, 320]);  addmm_261 = None
        view_809: "f32[8, 196, 5, 64]" = torch.ops.aten.reshape.default(view_808, [8, 196, 5, 64]);  view_808 = None
        permute_549: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_809, [0, 2, 1, 3]);  view_809 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_52 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_549, getitem_638, getitem_639, None, False);  permute_549 = getitem_638 = getitem_639 = None
        getitem_640: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_52[0];  _scaled_dot_product_efficient_attention_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_554: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_640, [0, 2, 1, 3]);  getitem_640 = None
        view_815: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(permute_554, [8, 196, 320]);  permute_554 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_816: "f32[1568, 320]" = torch.ops.aten.reshape.default(view_815, [1568, 320]);  view_815 = None
        permute_555: "f32[320, 320]" = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[1568, 320]" = torch.ops.aten.mm.default(view_816, permute_555);  view_816 = permute_555 = None
        add_tensor_11: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_11, arg462_1);  mm_default_11 = arg462_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_817: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 196, 320]);  add_tensor_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_489: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_484, view_817);  add_484 = view_817 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_163 = torch.ops.aten.var_mean.correction(add_489, [2], correction = 0, keepdim = True)
        getitem_644: "f32[8, 196, 1]" = var_mean_163[0]
        getitem_645: "f32[8, 196, 1]" = var_mean_163[1];  var_mean_163 = None
        sub_163: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_489, getitem_645);  getitem_645 = None
        add_490: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_644, 1e-06);  getitem_644 = None
        rsqrt_163: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_490);  add_490 = None
        mul_482: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_163, rsqrt_163);  sub_163 = rsqrt_163 = None
        mul_483: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_482, arg463_1);  mul_482 = arg463_1 = None
        add_491: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_483, arg464_1);  mul_483 = arg464_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_818: "f32[1568, 320]" = torch.ops.aten.reshape.default(add_491, [1568, 320]);  add_491 = None
        permute_556: "f32[320, 1280]" = torch.ops.aten.permute.default(arg465_1, [1, 0]);  arg465_1 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_818, permute_556);  view_818 = permute_556 = None
        add_tensor_10: "f32[1568, 1280]" = torch.ops.aten.add.Tensor(mm_default_10, arg466_1);  mm_default_10 = arg466_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_819: "f32[8, 196, 1280]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 196, 1280]);  add_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_484: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_819, 0.5)
        mul_485: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_819, 0.7071067811865476);  view_819 = None
        erf_52: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_485);  mul_485 = None
        add_492: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
        mul_486: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_484, add_492);  mul_484 = add_492 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_820: "f32[1568, 1280]" = torch.ops.aten.reshape.default(mul_486, [1568, 1280]);  mul_486 = None
        permute_557: "f32[1280, 320]" = torch.ops.aten.permute.default(arg467_1, [1, 0]);  arg467_1 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[1568, 320]" = torch.ops.aten.mm.default(view_820, permute_557);  view_820 = permute_557 = None
        add_tensor_9: "f32[1568, 320]" = torch.ops.aten.add.Tensor(mm_default_9, arg468_1);  mm_default_9 = arg468_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_821: "f32[8, 196, 320]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 196, 320]);  add_tensor_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_493: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_489, view_821);  add_489 = view_821 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:489 in forward_features, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
        view_822: "f32[8, 14, 14, 320]" = torch.ops.aten.reshape.default(add_493, [8, 14, 14, -1]);  add_493 = None
        permute_558: "f32[8, 320, 14, 14]" = torch.ops.aten.permute.default(view_822, [0, 3, 1, 2]);  view_822 = None
        clone_179: "f32[8, 320, 14, 14]" = torch.ops.aten.clone.default(permute_558, memory_format = torch.contiguous_format);  permute_558 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:275 in forward, code: x = self.proj(x).flatten(2).transpose(1, 2)
        convolution_64: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(clone_179, arg469_1, arg470_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_179 = arg469_1 = arg470_1 = None
        view_823: "f32[8, 512, 49]" = torch.ops.aten.reshape.default(convolution_64, [8, 512, 49]);  convolution_64 = None
        permute_559: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_823, [0, 2, 1]);  view_823 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:276 in forward, code: x = self.norm(x)
        clone_180: "f32[8, 49, 512]" = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
        var_mean_164 = torch.ops.aten.var_mean.correction(clone_180, [2], correction = 0, keepdim = True)
        getitem_646: "f32[8, 49, 1]" = var_mean_164[0]
        getitem_647: "f32[8, 49, 1]" = var_mean_164[1];  var_mean_164 = None
        sub_164: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_180, getitem_647);  clone_180 = getitem_647 = None
        add_494: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_646, 1e-05);  getitem_646 = None
        rsqrt_164: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_494);  add_494 = None
        mul_487: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_164, rsqrt_164);  sub_164 = rsqrt_164 = None
        mul_488: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_487, arg471_1);  mul_487 = arg471_1 = None
        add_495: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_488, arg472_1);  mul_488 = arg472_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_165 = torch.ops.aten.var_mean.correction(add_495, [2], correction = 0, keepdim = True)
        getitem_648: "f32[8, 49, 1]" = var_mean_165[0]
        getitem_649: "f32[8, 49, 1]" = var_mean_165[1];  var_mean_165 = None
        sub_165: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(add_495, getitem_649);  getitem_649 = None
        add_496: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_648, 1e-06);  getitem_648 = None
        rsqrt_165: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_496);  add_496 = None
        mul_489: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_165, rsqrt_165);  sub_165 = rsqrt_165 = None
        mul_490: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_489, arg473_1);  mul_489 = arg473_1 = None
        add_497: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_490, arg474_1);  mul_490 = arg474_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_827: "f32[392, 512]" = torch.ops.aten.reshape.default(add_497, [392, 512])
        permute_562: "f32[512, 1024]" = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
        addmm_267: "f32[392, 1024]" = torch.ops.aten.addmm.default(arg478_1, view_827, permute_562);  arg478_1 = view_827 = permute_562 = None
        view_828: "f32[8, 49, 1024]" = torch.ops.aten.reshape.default(addmm_267, [8, 49, 1024]);  addmm_267 = None
        view_829: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.reshape.default(view_828, [8, -1, 2, 8, 64]);  view_828 = None
        permute_563: "f32[2, 8, 8, 49, 64]" = torch.ops.aten.permute.default(view_829, [2, 0, 3, 1, 4]);  view_829 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_53 = torch.ops.aten.unbind.int(permute_563);  permute_563 = None
        getitem_650: "f32[8, 8, 49, 64]" = unbind_53[0]
        getitem_651: "f32[8, 8, 49, 64]" = unbind_53[1];  unbind_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_824: "f32[392, 512]" = torch.ops.aten.reshape.default(add_497, [392, 512]);  add_497 = None
        permute_560: "f32[512, 512]" = torch.ops.aten.permute.default(arg475_1, [1, 0]);  arg475_1 = None
        addmm_266: "f32[392, 512]" = torch.ops.aten.addmm.default(arg476_1, view_824, permute_560);  arg476_1 = view_824 = permute_560 = None
        view_825: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(addmm_266, [8, 49, 512]);  addmm_266 = None
        view_826: "f32[8, 49, 8, 64]" = torch.ops.aten.reshape.default(view_825, [8, 49, 8, 64]);  view_825 = None
        permute_561: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_826, [0, 2, 1, 3]);  view_826 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_53 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_561, getitem_650, getitem_651, None, False);  permute_561 = getitem_650 = getitem_651 = None
        getitem_652: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_53[0];  _scaled_dot_product_efficient_attention_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_564: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_652, [0, 2, 1, 3]);  getitem_652 = None
        view_830: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(permute_564, [8, 49, 512]);  permute_564 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_831: "f32[392, 512]" = torch.ops.aten.reshape.default(view_830, [392, 512]);  view_830 = None
        permute_565: "f32[512, 512]" = torch.ops.aten.permute.default(arg479_1, [1, 0]);  arg479_1 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[392, 512]" = torch.ops.aten.mm.default(view_831, permute_565);  view_831 = permute_565 = None
        add_tensor_8: "f32[392, 512]" = torch.ops.aten.add.Tensor(mm_default_8, arg480_1);  mm_default_8 = arg480_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_832: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 49, 512]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_498: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_495, view_832);  add_495 = view_832 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_166 = torch.ops.aten.var_mean.correction(add_498, [2], correction = 0, keepdim = True)
        getitem_656: "f32[8, 49, 1]" = var_mean_166[0]
        getitem_657: "f32[8, 49, 1]" = var_mean_166[1];  var_mean_166 = None
        sub_166: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(add_498, getitem_657);  getitem_657 = None
        add_499: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_656, 1e-06);  getitem_656 = None
        rsqrt_166: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_499);  add_499 = None
        mul_491: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_166, rsqrt_166);  sub_166 = rsqrt_166 = None
        mul_492: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_491, arg481_1);  mul_491 = arg481_1 = None
        add_500: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_492, arg482_1);  mul_492 = arg482_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_833: "f32[392, 512]" = torch.ops.aten.reshape.default(add_500, [392, 512]);  add_500 = None
        permute_566: "f32[512, 2048]" = torch.ops.aten.permute.default(arg483_1, [1, 0]);  arg483_1 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[392, 2048]" = torch.ops.aten.mm.default(view_833, permute_566);  view_833 = permute_566 = None
        add_tensor_7: "f32[392, 2048]" = torch.ops.aten.add.Tensor(mm_default_7, arg484_1);  mm_default_7 = arg484_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_834: "f32[8, 49, 2048]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 49, 2048]);  add_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_493: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_834, 0.5)
        mul_494: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_834, 0.7071067811865476);  view_834 = None
        erf_53: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_494);  mul_494 = None
        add_501: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
        mul_495: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_493, add_501);  mul_493 = add_501 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_835: "f32[392, 2048]" = torch.ops.aten.reshape.default(mul_495, [392, 2048]);  mul_495 = None
        permute_567: "f32[2048, 512]" = torch.ops.aten.permute.default(arg485_1, [1, 0]);  arg485_1 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[392, 512]" = torch.ops.aten.mm.default(view_835, permute_567);  view_835 = permute_567 = None
        add_tensor_6: "f32[392, 512]" = torch.ops.aten.add.Tensor(mm_default_6, arg486_1);  mm_default_6 = arg486_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_836: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 49, 512]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_502: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_498, view_836);  add_498 = view_836 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:243 in forward, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
        permute_568: "f32[8, 512, 49]" = torch.ops.aten.permute.default(add_502, [0, 2, 1]);  add_502 = None
        view_837: "f32[8, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_568, [8, 512, 7, 7]);  permute_568 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:244 in forward, code: x = self.proj(cnn_feat_token)
        convolution_65: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(view_837, arg487_1, arg488_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 512);  arg487_1 = arg488_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:246 in forward, code: x += cnn_feat_token
        add_503: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(convolution_65, view_837);  convolution_65 = view_837 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        view_839: "f32[8, 512, 49]" = torch.ops.aten.reshape.default(add_503, [8, 512, 49]);  add_503 = None
        permute_570: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_839, [0, 2, 1]);  view_839 = None
        var_mean_167 = torch.ops.aten.var_mean.correction(permute_570, [2], correction = 0, keepdim = True)
        getitem_658: "f32[8, 49, 1]" = var_mean_167[0]
        getitem_659: "f32[8, 49, 1]" = var_mean_167[1];  var_mean_167 = None
        sub_167: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(permute_570, getitem_659);  getitem_659 = None
        add_504: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_658, 1e-06);  getitem_658 = None
        rsqrt_167: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_504);  add_504 = None
        mul_496: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_167, rsqrt_167);  sub_167 = rsqrt_167 = None
        mul_497: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_496, arg489_1);  mul_496 = arg489_1 = None
        add_505: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_497, arg490_1);  mul_497 = arg490_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_843: "f32[392, 512]" = torch.ops.aten.reshape.default(add_505, [392, 512])
        permute_573: "f32[512, 1024]" = torch.ops.aten.permute.default(arg493_1, [1, 0]);  arg493_1 = None
        addmm_272: "f32[392, 1024]" = torch.ops.aten.addmm.default(arg494_1, view_843, permute_573);  arg494_1 = view_843 = permute_573 = None
        view_844: "f32[8, 49, 1024]" = torch.ops.aten.reshape.default(addmm_272, [8, 49, 1024]);  addmm_272 = None
        view_845: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.reshape.default(view_844, [8, -1, 2, 8, 64]);  view_844 = None
        permute_574: "f32[2, 8, 8, 49, 64]" = torch.ops.aten.permute.default(view_845, [2, 0, 3, 1, 4]);  view_845 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_54 = torch.ops.aten.unbind.int(permute_574);  permute_574 = None
        getitem_660: "f32[8, 8, 49, 64]" = unbind_54[0]
        getitem_661: "f32[8, 8, 49, 64]" = unbind_54[1];  unbind_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_840: "f32[392, 512]" = torch.ops.aten.reshape.default(add_505, [392, 512]);  add_505 = None
        permute_571: "f32[512, 512]" = torch.ops.aten.permute.default(arg491_1, [1, 0]);  arg491_1 = None
        addmm_271: "f32[392, 512]" = torch.ops.aten.addmm.default(arg492_1, view_840, permute_571);  arg492_1 = view_840 = permute_571 = None
        view_841: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(addmm_271, [8, 49, 512]);  addmm_271 = None
        view_842: "f32[8, 49, 8, 64]" = torch.ops.aten.reshape.default(view_841, [8, 49, 8, 64]);  view_841 = None
        permute_572: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_842, [0, 2, 1, 3]);  view_842 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_54 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_572, getitem_660, getitem_661, None, False);  permute_572 = getitem_660 = getitem_661 = None
        getitem_662: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_54[0];  _scaled_dot_product_efficient_attention_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_575: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_662, [0, 2, 1, 3]);  getitem_662 = None
        view_846: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(permute_575, [8, 49, 512]);  permute_575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_847: "f32[392, 512]" = torch.ops.aten.reshape.default(view_846, [392, 512]);  view_846 = None
        permute_576: "f32[512, 512]" = torch.ops.aten.permute.default(arg495_1, [1, 0]);  arg495_1 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[392, 512]" = torch.ops.aten.mm.default(view_847, permute_576);  view_847 = permute_576 = None
        add_tensor_5: "f32[392, 512]" = torch.ops.aten.add.Tensor(mm_default_5, arg496_1);  mm_default_5 = arg496_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_848: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 49, 512]);  add_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_506: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(permute_570, view_848);  permute_570 = view_848 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_168 = torch.ops.aten.var_mean.correction(add_506, [2], correction = 0, keepdim = True)
        getitem_666: "f32[8, 49, 1]" = var_mean_168[0]
        getitem_667: "f32[8, 49, 1]" = var_mean_168[1];  var_mean_168 = None
        sub_168: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(add_506, getitem_667);  getitem_667 = None
        add_507: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_666, 1e-06);  getitem_666 = None
        rsqrt_168: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_507);  add_507 = None
        mul_498: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_168, rsqrt_168);  sub_168 = rsqrt_168 = None
        mul_499: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_498, arg497_1);  mul_498 = arg497_1 = None
        add_508: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_499, arg498_1);  mul_499 = arg498_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_849: "f32[392, 512]" = torch.ops.aten.reshape.default(add_508, [392, 512]);  add_508 = None
        permute_577: "f32[512, 2048]" = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[392, 2048]" = torch.ops.aten.mm.default(view_849, permute_577);  view_849 = permute_577 = None
        add_tensor_4: "f32[392, 2048]" = torch.ops.aten.add.Tensor(mm_default_4, arg500_1);  mm_default_4 = arg500_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_850: "f32[8, 49, 2048]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 49, 2048]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_500: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_850, 0.5)
        mul_501: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_850, 0.7071067811865476);  view_850 = None
        erf_54: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_501);  mul_501 = None
        add_509: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
        mul_502: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_500, add_509);  mul_500 = add_509 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_851: "f32[392, 2048]" = torch.ops.aten.reshape.default(mul_502, [392, 2048]);  mul_502 = None
        permute_578: "f32[2048, 512]" = torch.ops.aten.permute.default(arg501_1, [1, 0]);  arg501_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[392, 512]" = torch.ops.aten.mm.default(view_851, permute_578);  view_851 = permute_578 = None
        add_tensor_3: "f32[392, 512]" = torch.ops.aten.add.Tensor(mm_default_3, arg502_1);  mm_default_3 = arg502_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_852: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 49, 512]);  add_tensor_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_510: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_506, view_852);  add_506 = view_852 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        var_mean_169 = torch.ops.aten.var_mean.correction(add_510, [2], correction = 0, keepdim = True)
        getitem_668: "f32[8, 49, 1]" = var_mean_169[0]
        getitem_669: "f32[8, 49, 1]" = var_mean_169[1];  var_mean_169 = None
        sub_169: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(add_510, getitem_669);  getitem_669 = None
        add_511: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_668, 1e-06);  getitem_668 = None
        rsqrt_169: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_511);  add_511 = None
        mul_503: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_169, rsqrt_169);  sub_169 = rsqrt_169 = None
        mul_504: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_503, arg503_1);  mul_503 = arg503_1 = None
        add_512: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_504, arg504_1);  mul_504 = arg504_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:170 in forward, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_856: "f32[392, 512]" = torch.ops.aten.reshape.default(add_512, [392, 512])
        permute_581: "f32[512, 1024]" = torch.ops.aten.permute.default(arg507_1, [1, 0]);  arg507_1 = None
        addmm_277: "f32[392, 1024]" = torch.ops.aten.addmm.default(arg508_1, view_856, permute_581);  arg508_1 = view_856 = permute_581 = None
        view_857: "f32[8, 49, 1024]" = torch.ops.aten.reshape.default(addmm_277, [8, 49, 1024]);  addmm_277 = None
        view_858: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.reshape.default(view_857, [8, -1, 2, 8, 64]);  view_857 = None
        permute_582: "f32[2, 8, 8, 49, 64]" = torch.ops.aten.permute.default(view_858, [2, 0, 3, 1, 4]);  view_858 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:171 in forward, code: k, v = kv.unbind(0)
        unbind_55 = torch.ops.aten.unbind.int(permute_582);  permute_582 = None
        getitem_670: "f32[8, 8, 49, 64]" = unbind_55[0]
        getitem_671: "f32[8, 8, 49, 64]" = unbind_55[1];  unbind_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:164 in forward, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_853: "f32[392, 512]" = torch.ops.aten.reshape.default(add_512, [392, 512]);  add_512 = None
        permute_579: "f32[512, 512]" = torch.ops.aten.permute.default(arg505_1, [1, 0]);  arg505_1 = None
        addmm_276: "f32[392, 512]" = torch.ops.aten.addmm.default(arg506_1, view_853, permute_579);  arg506_1 = view_853 = permute_579 = None
        view_854: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(addmm_276, [8, 49, 512]);  addmm_276 = None
        view_855: "f32[8, 49, 8, 64]" = torch.ops.aten.reshape.default(view_854, [8, 49, 8, 64]);  view_854 = None
        permute_580: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_855, [0, 2, 1, 3]);  view_855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:174 in forward, code: x = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_55 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_580, getitem_670, getitem_671, None, False);  permute_580 = getitem_670 = getitem_671 = None
        getitem_672: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_55[0];  _scaled_dot_product_efficient_attention_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:185 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_583: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_672, [0, 2, 1, 3]);  getitem_672 = None
        view_859: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(permute_583, [8, 49, 512]);  permute_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_860: "f32[392, 512]" = torch.ops.aten.reshape.default(view_859, [392, 512]);  view_859 = None
        permute_584: "f32[512, 512]" = torch.ops.aten.permute.default(arg509_1, [1, 0]);  arg509_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[392, 512]" = torch.ops.aten.mm.default(view_860, permute_584);  view_860 = permute_584 = None
        add_tensor_2: "f32[392, 512]" = torch.ops.aten.add.Tensor(mm_default_2, arg510_1);  mm_default_2 = arg510_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:186 in forward, code: x = self.proj(x)
        view_861: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 49, 512]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:227 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
        add_513: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_510, view_861);  add_510 = view_861 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_170 = torch.ops.aten.var_mean.correction(add_513, [2], correction = 0, keepdim = True)
        getitem_676: "f32[8, 49, 1]" = var_mean_170[0]
        getitem_677: "f32[8, 49, 1]" = var_mean_170[1];  var_mean_170 = None
        sub_170: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(add_513, getitem_677);  getitem_677 = None
        add_514: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_676, 1e-06);  getitem_676 = None
        rsqrt_170: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_514);  add_514 = None
        mul_505: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_170, rsqrt_170);  sub_170 = rsqrt_170 = None
        mul_506: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_505, arg511_1);  mul_505 = arg511_1 = None
        add_515: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_506, arg512_1);  mul_506 = arg512_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_862: "f32[392, 512]" = torch.ops.aten.reshape.default(add_515, [392, 512]);  add_515 = None
        permute_585: "f32[512, 2048]" = torch.ops.aten.permute.default(arg513_1, [1, 0]);  arg513_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[392, 2048]" = torch.ops.aten.mm.default(view_862, permute_585);  view_862 = permute_585 = None
        add_tensor_1: "f32[392, 2048]" = torch.ops.aten.add.Tensor(mm_default_1, arg514_1);  mm_default_1 = arg514_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_863: "f32[8, 49, 2048]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 49, 2048]);  add_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_507: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_863, 0.5)
        mul_508: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_863, 0.7071067811865476);  view_863 = None
        erf_55: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_508);  mul_508 = None
        add_516: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
        mul_509: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_507, add_516);  mul_507 = add_516 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_864: "f32[392, 2048]" = torch.ops.aten.reshape.default(mul_509, [392, 2048]);  mul_509 = None
        permute_586: "f32[2048, 512]" = torch.ops.aten.permute.default(arg515_1, [1, 0]);  arg515_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[392, 512]" = torch.ops.aten.mm.default(view_864, permute_586);  view_864 = permute_586 = None
        add_tensor: "f32[392, 512]" = torch.ops.aten.add.Tensor(mm_default, arg516_1);  mm_default = arg516_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_865: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_tensor, [8, 49, 512]);  add_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:228 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_517: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_513, view_865);  add_513 = view_865 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:490 in forward_features, code: x = self.norm(x)
        var_mean_171 = torch.ops.aten.var_mean.correction(add_517, [2], correction = 0, keepdim = True)
        getitem_678: "f32[8, 49, 1]" = var_mean_171[0]
        getitem_679: "f32[8, 49, 1]" = var_mean_171[1];  var_mean_171 = None
        sub_171: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(add_517, getitem_679);  add_517 = getitem_679 = None
        add_518: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_678, 1e-06);  getitem_678 = None
        rsqrt_171: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_518);  add_518 = None
        mul_510: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_171, rsqrt_171);  sub_171 = rsqrt_171 = None
        mul_511: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_510, arg517_1);  mul_510 = arg517_1 = None
        add_519: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_511, arg518_1);  mul_511 = arg518_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:495 in forward_head, code: x = x.mean(dim=1)
        mean_1: "f32[8, 512]" = torch.ops.aten.mean.dim(add_519, [1]);  add_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/twins.py:497 in forward_head, code: return x if pre_logits else self.head(x)
        permute_587: "f32[512, 1000]" = torch.ops.aten.permute.default(arg519_1, [1, 0]);  arg519_1 = None
        addmm_281: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg520_1, mean_1, permute_587);  arg520_1 = mean_1 = permute_587 = None
        return (addmm_281,)
        