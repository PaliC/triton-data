class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[128, 3, 7, 7]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "f32[128]", arg6_1: "f32[128]", arg7_1: "f32[128]", arg8_1: "f32[128]", arg9_1: "f32[128]", arg10_1: "f32[296, 128, 1, 1]", arg11_1: "f32[128]", arg12_1: "f32[128]", arg13_1: "f32[128]", arg14_1: "f32[128]", arg15_1: "f32[200, 128, 1, 1]", arg16_1: "f32[200]", arg17_1: "f32[200]", arg18_1: "f32[200]", arg19_1: "f32[200]", arg20_1: "f32[200, 4, 3, 3]", arg21_1: "f32[200]", arg22_1: "f32[200]", arg23_1: "f32[200]", arg24_1: "f32[200]", arg25_1: "f32[276, 200, 1, 1]", arg26_1: "f32[316]", arg27_1: "f32[316]", arg28_1: "f32[316]", arg29_1: "f32[316]", arg30_1: "f32[200, 316, 1, 1]", arg31_1: "f32[200]", arg32_1: "f32[200]", arg33_1: "f32[200]", arg34_1: "f32[200]", arg35_1: "f32[200, 4, 3, 3]", arg36_1: "f32[200]", arg37_1: "f32[200]", arg38_1: "f32[200]", arg39_1: "f32[200]", arg40_1: "f32[276, 200, 1, 1]", arg41_1: "f32[336]", arg42_1: "f32[336]", arg43_1: "f32[336]", arg44_1: "f32[336]", arg45_1: "f32[200, 336, 1, 1]", arg46_1: "f32[200]", arg47_1: "f32[200]", arg48_1: "f32[200]", arg49_1: "f32[200]", arg50_1: "f32[200, 4, 3, 3]", arg51_1: "f32[200]", arg52_1: "f32[200]", arg53_1: "f32[200]", arg54_1: "f32[200]", arg55_1: "f32[276, 200, 1, 1]", arg56_1: "f32[356]", arg57_1: "f32[356]", arg58_1: "f32[356]", arg59_1: "f32[356]", arg60_1: "f32[200, 356, 1, 1]", arg61_1: "f32[200]", arg62_1: "f32[200]", arg63_1: "f32[200]", arg64_1: "f32[200]", arg65_1: "f32[200, 4, 3, 3]", arg66_1: "f32[200]", arg67_1: "f32[200]", arg68_1: "f32[200]", arg69_1: "f32[200]", arg70_1: "f32[276, 200, 1, 1]", arg71_1: "f32[376]", arg72_1: "f32[376]", arg73_1: "f32[376]", arg74_1: "f32[376]", arg75_1: "f32[640, 376, 1, 1]", arg76_1: "f32[376]", arg77_1: "f32[376]", arg78_1: "f32[376]", arg79_1: "f32[376]", arg80_1: "f32[400, 376, 1, 1]", arg81_1: "f32[400]", arg82_1: "f32[400]", arg83_1: "f32[400]", arg84_1: "f32[400]", arg85_1: "f32[400, 8, 3, 3]", arg86_1: "f32[400]", arg87_1: "f32[400]", arg88_1: "f32[400]", arg89_1: "f32[400]", arg90_1: "f32[576, 400, 1, 1]", arg91_1: "f32[704]", arg92_1: "f32[704]", arg93_1: "f32[704]", arg94_1: "f32[704]", arg95_1: "f32[400, 704, 1, 1]", arg96_1: "f32[400]", arg97_1: "f32[400]", arg98_1: "f32[400]", arg99_1: "f32[400]", arg100_1: "f32[400, 8, 3, 3]", arg101_1: "f32[400]", arg102_1: "f32[400]", arg103_1: "f32[400]", arg104_1: "f32[400]", arg105_1: "f32[576, 400, 1, 1]", arg106_1: "f32[768]", arg107_1: "f32[768]", arg108_1: "f32[768]", arg109_1: "f32[768]", arg110_1: "f32[400, 768, 1, 1]", arg111_1: "f32[400]", arg112_1: "f32[400]", arg113_1: "f32[400]", arg114_1: "f32[400]", arg115_1: "f32[400, 8, 3, 3]", arg116_1: "f32[400]", arg117_1: "f32[400]", arg118_1: "f32[400]", arg119_1: "f32[400]", arg120_1: "f32[576, 400, 1, 1]", arg121_1: "f32[832]", arg122_1: "f32[832]", arg123_1: "f32[832]", arg124_1: "f32[832]", arg125_1: "f32[400, 832, 1, 1]", arg126_1: "f32[400]", arg127_1: "f32[400]", arg128_1: "f32[400]", arg129_1: "f32[400]", arg130_1: "f32[400, 8, 3, 3]", arg131_1: "f32[400]", arg132_1: "f32[400]", arg133_1: "f32[400]", arg134_1: "f32[400]", arg135_1: "f32[576, 400, 1, 1]", arg136_1: "f32[896]", arg137_1: "f32[896]", arg138_1: "f32[896]", arg139_1: "f32[896]", arg140_1: "f32[400, 896, 1, 1]", arg141_1: "f32[400]", arg142_1: "f32[400]", arg143_1: "f32[400]", arg144_1: "f32[400]", arg145_1: "f32[400, 8, 3, 3]", arg146_1: "f32[400]", arg147_1: "f32[400]", arg148_1: "f32[400]", arg149_1: "f32[400]", arg150_1: "f32[576, 400, 1, 1]", arg151_1: "f32[960]", arg152_1: "f32[960]", arg153_1: "f32[960]", arg154_1: "f32[960]", arg155_1: "f32[400, 960, 1, 1]", arg156_1: "f32[400]", arg157_1: "f32[400]", arg158_1: "f32[400]", arg159_1: "f32[400]", arg160_1: "f32[400, 8, 3, 3]", arg161_1: "f32[400]", arg162_1: "f32[400]", arg163_1: "f32[400]", arg164_1: "f32[400]", arg165_1: "f32[576, 400, 1, 1]", arg166_1: "f32[1024]", arg167_1: "f32[1024]", arg168_1: "f32[1024]", arg169_1: "f32[1024]", arg170_1: "f32[400, 1024, 1, 1]", arg171_1: "f32[400]", arg172_1: "f32[400]", arg173_1: "f32[400]", arg174_1: "f32[400]", arg175_1: "f32[400, 8, 3, 3]", arg176_1: "f32[400]", arg177_1: "f32[400]", arg178_1: "f32[400]", arg179_1: "f32[400]", arg180_1: "f32[576, 400, 1, 1]", arg181_1: "f32[1088]", arg182_1: "f32[1088]", arg183_1: "f32[1088]", arg184_1: "f32[1088]", arg185_1: "f32[400, 1088, 1, 1]", arg186_1: "f32[400]", arg187_1: "f32[400]", arg188_1: "f32[400]", arg189_1: "f32[400]", arg190_1: "f32[400, 8, 3, 3]", arg191_1: "f32[400]", arg192_1: "f32[400]", arg193_1: "f32[400]", arg194_1: "f32[400]", arg195_1: "f32[576, 400, 1, 1]", arg196_1: "f32[1152]", arg197_1: "f32[1152]", arg198_1: "f32[1152]", arg199_1: "f32[1152]", arg200_1: "f32[1152, 1152, 1, 1]", arg201_1: "f32[1152]", arg202_1: "f32[1152]", arg203_1: "f32[1152]", arg204_1: "f32[1152]", arg205_1: "f32[800, 1152, 1, 1]", arg206_1: "f32[800]", arg207_1: "f32[800]", arg208_1: "f32[800]", arg209_1: "f32[800]", arg210_1: "f32[800, 16, 3, 3]", arg211_1: "f32[800]", arg212_1: "f32[800]", arg213_1: "f32[800]", arg214_1: "f32[800]", arg215_1: "f32[1088, 800, 1, 1]", arg216_1: "f32[1216]", arg217_1: "f32[1216]", arg218_1: "f32[1216]", arg219_1: "f32[1216]", arg220_1: "f32[800, 1216, 1, 1]", arg221_1: "f32[800]", arg222_1: "f32[800]", arg223_1: "f32[800]", arg224_1: "f32[800]", arg225_1: "f32[800, 16, 3, 3]", arg226_1: "f32[800]", arg227_1: "f32[800]", arg228_1: "f32[800]", arg229_1: "f32[800]", arg230_1: "f32[1088, 800, 1, 1]", arg231_1: "f32[1280]", arg232_1: "f32[1280]", arg233_1: "f32[1280]", arg234_1: "f32[1280]", arg235_1: "f32[800, 1280, 1, 1]", arg236_1: "f32[800]", arg237_1: "f32[800]", arg238_1: "f32[800]", arg239_1: "f32[800]", arg240_1: "f32[800, 16, 3, 3]", arg241_1: "f32[800]", arg242_1: "f32[800]", arg243_1: "f32[800]", arg244_1: "f32[800]", arg245_1: "f32[1088, 800, 1, 1]", arg246_1: "f32[1344]", arg247_1: "f32[1344]", arg248_1: "f32[1344]", arg249_1: "f32[1344]", arg250_1: "f32[800, 1344, 1, 1]", arg251_1: "f32[800]", arg252_1: "f32[800]", arg253_1: "f32[800]", arg254_1: "f32[800]", arg255_1: "f32[800, 16, 3, 3]", arg256_1: "f32[800]", arg257_1: "f32[800]", arg258_1: "f32[800]", arg259_1: "f32[800]", arg260_1: "f32[1088, 800, 1, 1]", arg261_1: "f32[1408]", arg262_1: "f32[1408]", arg263_1: "f32[1408]", arg264_1: "f32[1408]", arg265_1: "f32[800, 1408, 1, 1]", arg266_1: "f32[800]", arg267_1: "f32[800]", arg268_1: "f32[800]", arg269_1: "f32[800]", arg270_1: "f32[800, 16, 3, 3]", arg271_1: "f32[800]", arg272_1: "f32[800]", arg273_1: "f32[800]", arg274_1: "f32[800]", arg275_1: "f32[1088, 800, 1, 1]", arg276_1: "f32[1472]", arg277_1: "f32[1472]", arg278_1: "f32[1472]", arg279_1: "f32[1472]", arg280_1: "f32[800, 1472, 1, 1]", arg281_1: "f32[800]", arg282_1: "f32[800]", arg283_1: "f32[800]", arg284_1: "f32[800]", arg285_1: "f32[800, 16, 3, 3]", arg286_1: "f32[800]", arg287_1: "f32[800]", arg288_1: "f32[800]", arg289_1: "f32[800]", arg290_1: "f32[1088, 800, 1, 1]", arg291_1: "f32[1536]", arg292_1: "f32[1536]", arg293_1: "f32[1536]", arg294_1: "f32[1536]", arg295_1: "f32[800, 1536, 1, 1]", arg296_1: "f32[800]", arg297_1: "f32[800]", arg298_1: "f32[800]", arg299_1: "f32[800]", arg300_1: "f32[800, 16, 3, 3]", arg301_1: "f32[800]", arg302_1: "f32[800]", arg303_1: "f32[800]", arg304_1: "f32[800]", arg305_1: "f32[1088, 800, 1, 1]", arg306_1: "f32[1600]", arg307_1: "f32[1600]", arg308_1: "f32[1600]", arg309_1: "f32[1600]", arg310_1: "f32[800, 1600, 1, 1]", arg311_1: "f32[800]", arg312_1: "f32[800]", arg313_1: "f32[800]", arg314_1: "f32[800]", arg315_1: "f32[800, 16, 3, 3]", arg316_1: "f32[800]", arg317_1: "f32[800]", arg318_1: "f32[800]", arg319_1: "f32[800]", arg320_1: "f32[1088, 800, 1, 1]", arg321_1: "f32[1664]", arg322_1: "f32[1664]", arg323_1: "f32[1664]", arg324_1: "f32[1664]", arg325_1: "f32[800, 1664, 1, 1]", arg326_1: "f32[800]", arg327_1: "f32[800]", arg328_1: "f32[800]", arg329_1: "f32[800]", arg330_1: "f32[800, 16, 3, 3]", arg331_1: "f32[800]", arg332_1: "f32[800]", arg333_1: "f32[800]", arg334_1: "f32[800]", arg335_1: "f32[1088, 800, 1, 1]", arg336_1: "f32[1728]", arg337_1: "f32[1728]", arg338_1: "f32[1728]", arg339_1: "f32[1728]", arg340_1: "f32[800, 1728, 1, 1]", arg341_1: "f32[800]", arg342_1: "f32[800]", arg343_1: "f32[800]", arg344_1: "f32[800]", arg345_1: "f32[800, 16, 3, 3]", arg346_1: "f32[800]", arg347_1: "f32[800]", arg348_1: "f32[800]", arg349_1: "f32[800]", arg350_1: "f32[1088, 800, 1, 1]", arg351_1: "f32[1792]", arg352_1: "f32[1792]", arg353_1: "f32[1792]", arg354_1: "f32[1792]", arg355_1: "f32[800, 1792, 1, 1]", arg356_1: "f32[800]", arg357_1: "f32[800]", arg358_1: "f32[800]", arg359_1: "f32[800]", arg360_1: "f32[800, 16, 3, 3]", arg361_1: "f32[800]", arg362_1: "f32[800]", arg363_1: "f32[800]", arg364_1: "f32[800]", arg365_1: "f32[1088, 800, 1, 1]", arg366_1: "f32[1856]", arg367_1: "f32[1856]", arg368_1: "f32[1856]", arg369_1: "f32[1856]", arg370_1: "f32[800, 1856, 1, 1]", arg371_1: "f32[800]", arg372_1: "f32[800]", arg373_1: "f32[800]", arg374_1: "f32[800]", arg375_1: "f32[800, 16, 3, 3]", arg376_1: "f32[800]", arg377_1: "f32[800]", arg378_1: "f32[800]", arg379_1: "f32[800]", arg380_1: "f32[1088, 800, 1, 1]", arg381_1: "f32[1920]", arg382_1: "f32[1920]", arg383_1: "f32[1920]", arg384_1: "f32[1920]", arg385_1: "f32[800, 1920, 1, 1]", arg386_1: "f32[800]", arg387_1: "f32[800]", arg388_1: "f32[800]", arg389_1: "f32[800]", arg390_1: "f32[800, 16, 3, 3]", arg391_1: "f32[800]", arg392_1: "f32[800]", arg393_1: "f32[800]", arg394_1: "f32[800]", arg395_1: "f32[1088, 800, 1, 1]", arg396_1: "f32[1984]", arg397_1: "f32[1984]", arg398_1: "f32[1984]", arg399_1: "f32[1984]", arg400_1: "f32[800, 1984, 1, 1]", arg401_1: "f32[800]", arg402_1: "f32[800]", arg403_1: "f32[800]", arg404_1: "f32[800]", arg405_1: "f32[800, 16, 3, 3]", arg406_1: "f32[800]", arg407_1: "f32[800]", arg408_1: "f32[800]", arg409_1: "f32[800]", arg410_1: "f32[1088, 800, 1, 1]", arg411_1: "f32[2048]", arg412_1: "f32[2048]", arg413_1: "f32[2048]", arg414_1: "f32[2048]", arg415_1: "f32[800, 2048, 1, 1]", arg416_1: "f32[800]", arg417_1: "f32[800]", arg418_1: "f32[800]", arg419_1: "f32[800]", arg420_1: "f32[800, 16, 3, 3]", arg421_1: "f32[800]", arg422_1: "f32[800]", arg423_1: "f32[800]", arg424_1: "f32[800]", arg425_1: "f32[1088, 800, 1, 1]", arg426_1: "f32[2112]", arg427_1: "f32[2112]", arg428_1: "f32[2112]", arg429_1: "f32[2112]", arg430_1: "f32[800, 2112, 1, 1]", arg431_1: "f32[800]", arg432_1: "f32[800]", arg433_1: "f32[800]", arg434_1: "f32[800]", arg435_1: "f32[800, 16, 3, 3]", arg436_1: "f32[800]", arg437_1: "f32[800]", arg438_1: "f32[800]", arg439_1: "f32[800]", arg440_1: "f32[1088, 800, 1, 1]", arg441_1: "f32[2176]", arg442_1: "f32[2176]", arg443_1: "f32[2176]", arg444_1: "f32[2176]", arg445_1: "f32[800, 2176, 1, 1]", arg446_1: "f32[800]", arg447_1: "f32[800]", arg448_1: "f32[800]", arg449_1: "f32[800]", arg450_1: "f32[800, 16, 3, 3]", arg451_1: "f32[800]", arg452_1: "f32[800]", arg453_1: "f32[800]", arg454_1: "f32[800]", arg455_1: "f32[1088, 800, 1, 1]", arg456_1: "f32[2240]", arg457_1: "f32[2240]", arg458_1: "f32[2240]", arg459_1: "f32[2240]", arg460_1: "f32[800, 2240, 1, 1]", arg461_1: "f32[800]", arg462_1: "f32[800]", arg463_1: "f32[800]", arg464_1: "f32[800]", arg465_1: "f32[800, 16, 3, 3]", arg466_1: "f32[800]", arg467_1: "f32[800]", arg468_1: "f32[800]", arg469_1: "f32[800]", arg470_1: "f32[1088, 800, 1, 1]", arg471_1: "f32[2304]", arg472_1: "f32[2304]", arg473_1: "f32[2304]", arg474_1: "f32[2304]", arg475_1: "f32[800, 2304, 1, 1]", arg476_1: "f32[800]", arg477_1: "f32[800]", arg478_1: "f32[800]", arg479_1: "f32[800]", arg480_1: "f32[800, 16, 3, 3]", arg481_1: "f32[800]", arg482_1: "f32[800]", arg483_1: "f32[800]", arg484_1: "f32[800]", arg485_1: "f32[1088, 800, 1, 1]", arg486_1: "f32[2368]", arg487_1: "f32[2368]", arg488_1: "f32[2368]", arg489_1: "f32[2368]", arg490_1: "f32[800, 2368, 1, 1]", arg491_1: "f32[800]", arg492_1: "f32[800]", arg493_1: "f32[800]", arg494_1: "f32[800]", arg495_1: "f32[800, 16, 3, 3]", arg496_1: "f32[800]", arg497_1: "f32[800]", arg498_1: "f32[800]", arg499_1: "f32[800]", arg500_1: "f32[1088, 800, 1, 1]", arg501_1: "f32[2432]", arg502_1: "f32[2432]", arg503_1: "f32[2432]", arg504_1: "f32[2432]", arg505_1: "f32[2304, 2432, 1, 1]", arg506_1: "f32[2432]", arg507_1: "f32[2432]", arg508_1: "f32[2432]", arg509_1: "f32[2432]", arg510_1: "f32[1600, 2432, 1, 1]", arg511_1: "f32[1600]", arg512_1: "f32[1600]", arg513_1: "f32[1600]", arg514_1: "f32[1600]", arg515_1: "f32[1600, 32, 3, 3]", arg516_1: "f32[1600]", arg517_1: "f32[1600]", arg518_1: "f32[1600]", arg519_1: "f32[1600]", arg520_1: "f32[2176, 1600, 1, 1]", arg521_1: "f32[2432]", arg522_1: "f32[2432]", arg523_1: "f32[2432]", arg524_1: "f32[2432]", arg525_1: "f32[1600, 2432, 1, 1]", arg526_1: "f32[1600]", arg527_1: "f32[1600]", arg528_1: "f32[1600]", arg529_1: "f32[1600]", arg530_1: "f32[1600, 32, 3, 3]", arg531_1: "f32[1600]", arg532_1: "f32[1600]", arg533_1: "f32[1600]", arg534_1: "f32[1600]", arg535_1: "f32[2176, 1600, 1, 1]", arg536_1: "f32[2560]", arg537_1: "f32[2560]", arg538_1: "f32[2560]", arg539_1: "f32[2560]", arg540_1: "f32[1600, 2560, 1, 1]", arg541_1: "f32[1600]", arg542_1: "f32[1600]", arg543_1: "f32[1600]", arg544_1: "f32[1600]", arg545_1: "f32[1600, 32, 3, 3]", arg546_1: "f32[1600]", arg547_1: "f32[1600]", arg548_1: "f32[1600]", arg549_1: "f32[1600]", arg550_1: "f32[2176, 1600, 1, 1]", arg551_1: "f32[2688]", arg552_1: "f32[2688]", arg553_1: "f32[2688]", arg554_1: "f32[2688]", arg555_1: "f32[1000, 2688, 1, 1]", arg556_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_111: "f32[8, 128, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_888: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_889: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        sub_111: "f32[8, 128, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_889);  convolution_111 = unsqueeze_889 = None
        add_257: "f32[128]" = torch.ops.aten.add.Tensor(arg3_1, 0.001);  arg3_1 = None
        sqrt_111: "f32[128]" = torch.ops.aten.sqrt.default(add_257);  add_257 = None
        reciprocal_111: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_333: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_890: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
        unsqueeze_891: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        mul_334: "f32[8, 128, 112, 112]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_893: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_335: "f32[8, 128, 112, 112]" = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_893);  mul_334 = unsqueeze_893 = None
        unsqueeze_894: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_895: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_258: "f32[8, 128, 112, 112]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_895);  mul_335 = unsqueeze_895 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_111: "f32[8, 128, 112, 112]" = torch.ops.aten.relu.default(add_258);  add_258 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:266 in forward_features, code: return self.features(x)
        _low_memory_max_pool2d_with_offsets_1 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_111, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu_111 = None
        getitem_2: "f32[8, 128, 56, 56]" = _low_memory_max_pool2d_with_offsets_1[0];  _low_memory_max_pool2d_with_offsets_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_896: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_897: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        sub_112: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_2, unsqueeze_897);  unsqueeze_897 = None
        add_259: "f32[128]" = torch.ops.aten.add.Tensor(arg7_1, 0.001);  arg7_1 = None
        sqrt_112: "f32[128]" = torch.ops.aten.sqrt.default(add_259);  add_259 = None
        reciprocal_112: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_336: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_898: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
        unsqueeze_899: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        mul_337: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_901: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_338: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_901);  mul_337 = unsqueeze_901 = None
        unsqueeze_902: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_903: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_260: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_338, unsqueeze_903);  mul_338 = unsqueeze_903 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_112: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_260);  add_260 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_112: "f32[8, 296, 56, 56]" = torch.ops.aten.convolution.default(relu_112, arg10_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_112 = arg10_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:133 in forward, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
        slice_314: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_112, 1, 0, 256)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_904: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_905: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        sub_113: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_2, unsqueeze_905);  getitem_2 = unsqueeze_905 = None
        add_261: "f32[128]" = torch.ops.aten.add.Tensor(arg12_1, 0.001);  arg12_1 = None
        sqrt_113: "f32[128]" = torch.ops.aten.sqrt.default(add_261);  add_261 = None
        reciprocal_113: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_339: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_906: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
        unsqueeze_907: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        mul_340: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
        unsqueeze_909: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_341: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_909);  mul_340 = unsqueeze_909 = None
        unsqueeze_910: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_911: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_262: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_341, unsqueeze_911);  mul_341 = unsqueeze_911 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_113: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_262);  add_262 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_113: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_113, arg15_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_113 = arg15_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_912: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_913: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        sub_114: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_913);  convolution_113 = unsqueeze_913 = None
        add_263: "f32[200]" = torch.ops.aten.add.Tensor(arg17_1, 0.001);  arg17_1 = None
        sqrt_114: "f32[200]" = torch.ops.aten.sqrt.default(add_263);  add_263 = None
        reciprocal_114: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_342: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_914: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
        unsqueeze_915: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        mul_343: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_917: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_344: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_917);  mul_343 = unsqueeze_917 = None
        unsqueeze_918: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_919: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_264: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_344, unsqueeze_919);  mul_344 = unsqueeze_919 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_114: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_264);  add_264 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_114: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_114, arg20_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_114 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_920: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_921: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        sub_115: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_921);  convolution_114 = unsqueeze_921 = None
        add_265: "f32[200]" = torch.ops.aten.add.Tensor(arg22_1, 0.001);  arg22_1 = None
        sqrt_115: "f32[200]" = torch.ops.aten.sqrt.default(add_265);  add_265 = None
        reciprocal_115: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_345: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_922: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_345, -1);  mul_345 = None
        unsqueeze_923: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        mul_346: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_925: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_347: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_925);  mul_346 = unsqueeze_925 = None
        unsqueeze_926: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_927: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_266: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_347, unsqueeze_927);  mul_347 = unsqueeze_927 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_115: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_266);  add_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_115: "f32[8, 276, 56, 56]" = torch.ops.aten.convolution.default(relu_115, arg25_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_115 = arg25_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_322: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_115, 1, 0, 256)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_267: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(slice_314, slice_322);  slice_314 = slice_322 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:134 in forward, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
        slice_318: "f32[8, 40, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_112, 1, 256, 9223372036854775807);  convolution_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_326: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_115, 1, 256, 9223372036854775807);  convolution_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_70: "f32[8, 60, 56, 56]" = torch.ops.aten.cat.default([slice_318, slice_326], 1);  slice_318 = slice_326 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_71: "f32[8, 316, 56, 56]" = torch.ops.aten.cat.default([add_267, cat_70], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_928: "f32[316, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_929: "f32[316, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
        sub_116: "f32[8, 316, 56, 56]" = torch.ops.aten.sub.Tensor(cat_71, unsqueeze_929);  cat_71 = unsqueeze_929 = None
        add_268: "f32[316]" = torch.ops.aten.add.Tensor(arg27_1, 0.001);  arg27_1 = None
        sqrt_116: "f32[316]" = torch.ops.aten.sqrt.default(add_268);  add_268 = None
        reciprocal_116: "f32[316]" = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_348: "f32[316]" = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        unsqueeze_930: "f32[316, 1]" = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
        unsqueeze_931: "f32[316, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
        mul_349: "f32[8, 316, 56, 56]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
        unsqueeze_932: "f32[316, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_933: "f32[316, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
        mul_350: "f32[8, 316, 56, 56]" = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_933);  mul_349 = unsqueeze_933 = None
        unsqueeze_934: "f32[316, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_935: "f32[316, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
        add_269: "f32[8, 316, 56, 56]" = torch.ops.aten.add.Tensor(mul_350, unsqueeze_935);  mul_350 = unsqueeze_935 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_116: "f32[8, 316, 56, 56]" = torch.ops.aten.relu.default(add_269);  add_269 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_116: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_116, arg30_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_116 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_936: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_937: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
        sub_117: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_937);  convolution_116 = unsqueeze_937 = None
        add_270: "f32[200]" = torch.ops.aten.add.Tensor(arg32_1, 0.001);  arg32_1 = None
        sqrt_117: "f32[200]" = torch.ops.aten.sqrt.default(add_270);  add_270 = None
        reciprocal_117: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_351: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        unsqueeze_938: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
        unsqueeze_939: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
        mul_352: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
        unsqueeze_940: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_941: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
        mul_353: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_941);  mul_352 = unsqueeze_941 = None
        unsqueeze_942: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_943: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
        add_271: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_353, unsqueeze_943);  mul_353 = unsqueeze_943 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_117: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_271);  add_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_117: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_117, arg35_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_117 = arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_944: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_945: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
        sub_118: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_945);  convolution_117 = unsqueeze_945 = None
        add_272: "f32[200]" = torch.ops.aten.add.Tensor(arg37_1, 0.001);  arg37_1 = None
        sqrt_118: "f32[200]" = torch.ops.aten.sqrt.default(add_272);  add_272 = None
        reciprocal_118: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_354: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        unsqueeze_946: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_354, -1);  mul_354 = None
        unsqueeze_947: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
        mul_355: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
        unsqueeze_948: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_949: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
        mul_356: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_355, unsqueeze_949);  mul_355 = unsqueeze_949 = None
        unsqueeze_950: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_951: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
        add_273: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_951);  mul_356 = unsqueeze_951 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_118: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_273);  add_273 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_118: "f32[8, 276, 56, 56]" = torch.ops.aten.convolution.default(relu_118, arg40_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_118 = arg40_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_330: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_118, 1, 0, 256)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_274: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_267, slice_330);  add_267 = slice_330 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_334: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_118, 1, 256, 9223372036854775807);  convolution_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_72: "f32[8, 80, 56, 56]" = torch.ops.aten.cat.default([cat_70, slice_334], 1);  cat_70 = slice_334 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_73: "f32[8, 336, 56, 56]" = torch.ops.aten.cat.default([add_274, cat_72], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_952: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_953: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
        sub_119: "f32[8, 336, 56, 56]" = torch.ops.aten.sub.Tensor(cat_73, unsqueeze_953);  cat_73 = unsqueeze_953 = None
        add_275: "f32[336]" = torch.ops.aten.add.Tensor(arg42_1, 0.001);  arg42_1 = None
        sqrt_119: "f32[336]" = torch.ops.aten.sqrt.default(add_275);  add_275 = None
        reciprocal_119: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_357: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        unsqueeze_954: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_357, -1);  mul_357 = None
        unsqueeze_955: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
        mul_358: "f32[8, 336, 56, 56]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
        unsqueeze_956: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_957: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
        mul_359: "f32[8, 336, 56, 56]" = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_957);  mul_358 = unsqueeze_957 = None
        unsqueeze_958: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_959: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
        add_276: "f32[8, 336, 56, 56]" = torch.ops.aten.add.Tensor(mul_359, unsqueeze_959);  mul_359 = unsqueeze_959 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_119: "f32[8, 336, 56, 56]" = torch.ops.aten.relu.default(add_276);  add_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_119: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_119, arg45_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_119 = arg45_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_960: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
        unsqueeze_961: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
        sub_120: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_961);  convolution_119 = unsqueeze_961 = None
        add_277: "f32[200]" = torch.ops.aten.add.Tensor(arg47_1, 0.001);  arg47_1 = None
        sqrt_120: "f32[200]" = torch.ops.aten.sqrt.default(add_277);  add_277 = None
        reciprocal_120: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_360: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        unsqueeze_962: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
        unsqueeze_963: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
        mul_361: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
        unsqueeze_964: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_965: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
        mul_362: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_965);  mul_361 = unsqueeze_965 = None
        unsqueeze_966: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_967: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
        add_278: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_362, unsqueeze_967);  mul_362 = unsqueeze_967 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_120: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_278);  add_278 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_120: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_120, arg50_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_120 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_968: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_969: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
        sub_121: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_969);  convolution_120 = unsqueeze_969 = None
        add_279: "f32[200]" = torch.ops.aten.add.Tensor(arg52_1, 0.001);  arg52_1 = None
        sqrt_121: "f32[200]" = torch.ops.aten.sqrt.default(add_279);  add_279 = None
        reciprocal_121: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_363: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        unsqueeze_970: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_363, -1);  mul_363 = None
        unsqueeze_971: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
        mul_364: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
        unsqueeze_972: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_973: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
        mul_365: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_973);  mul_364 = unsqueeze_973 = None
        unsqueeze_974: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_975: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
        add_280: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_365, unsqueeze_975);  mul_365 = unsqueeze_975 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_121: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_280);  add_280 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_121: "f32[8, 276, 56, 56]" = torch.ops.aten.convolution.default(relu_121, arg55_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_121 = arg55_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_338: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_121, 1, 0, 256)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_281: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_274, slice_338);  add_274 = slice_338 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_342: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_121, 1, 256, 9223372036854775807);  convolution_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_74: "f32[8, 100, 56, 56]" = torch.ops.aten.cat.default([cat_72, slice_342], 1);  cat_72 = slice_342 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_75: "f32[8, 356, 56, 56]" = torch.ops.aten.cat.default([add_281, cat_74], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_976: "f32[356, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_977: "f32[356, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
        sub_122: "f32[8, 356, 56, 56]" = torch.ops.aten.sub.Tensor(cat_75, unsqueeze_977);  cat_75 = unsqueeze_977 = None
        add_282: "f32[356]" = torch.ops.aten.add.Tensor(arg57_1, 0.001);  arg57_1 = None
        sqrt_122: "f32[356]" = torch.ops.aten.sqrt.default(add_282);  add_282 = None
        reciprocal_122: "f32[356]" = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_366: "f32[356]" = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        unsqueeze_978: "f32[356, 1]" = torch.ops.aten.unsqueeze.default(mul_366, -1);  mul_366 = None
        unsqueeze_979: "f32[356, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
        mul_367: "f32[8, 356, 56, 56]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
        unsqueeze_980: "f32[356, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_981: "f32[356, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
        mul_368: "f32[8, 356, 56, 56]" = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_981);  mul_367 = unsqueeze_981 = None
        unsqueeze_982: "f32[356, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_983: "f32[356, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
        add_283: "f32[8, 356, 56, 56]" = torch.ops.aten.add.Tensor(mul_368, unsqueeze_983);  mul_368 = unsqueeze_983 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_122: "f32[8, 356, 56, 56]" = torch.ops.aten.relu.default(add_283);  add_283 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_122: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_122, arg60_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_122 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_984: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_985: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
        sub_123: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_985);  convolution_122 = unsqueeze_985 = None
        add_284: "f32[200]" = torch.ops.aten.add.Tensor(arg62_1, 0.001);  arg62_1 = None
        sqrt_123: "f32[200]" = torch.ops.aten.sqrt.default(add_284);  add_284 = None
        reciprocal_123: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_369: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        unsqueeze_986: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
        unsqueeze_987: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
        mul_370: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
        unsqueeze_988: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_989: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
        mul_371: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_989);  mul_370 = unsqueeze_989 = None
        unsqueeze_990: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_991: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
        add_285: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_371, unsqueeze_991);  mul_371 = unsqueeze_991 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_123: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_285);  add_285 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_123: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_123, arg65_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_123 = arg65_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_992: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_993: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, -1);  unsqueeze_992 = None
        sub_124: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_993);  convolution_123 = unsqueeze_993 = None
        add_286: "f32[200]" = torch.ops.aten.add.Tensor(arg67_1, 0.001);  arg67_1 = None
        sqrt_124: "f32[200]" = torch.ops.aten.sqrt.default(add_286);  add_286 = None
        reciprocal_124: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
        mul_372: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
        unsqueeze_994: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
        unsqueeze_995: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, -1);  unsqueeze_994 = None
        mul_373: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_995);  sub_124 = unsqueeze_995 = None
        unsqueeze_996: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_997: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, -1);  unsqueeze_996 = None
        mul_374: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_997);  mul_373 = unsqueeze_997 = None
        unsqueeze_998: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_999: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, -1);  unsqueeze_998 = None
        add_287: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_999);  mul_374 = unsqueeze_999 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_124: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_287);  add_287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_124: "f32[8, 276, 56, 56]" = torch.ops.aten.convolution.default(relu_124, arg70_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_124 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_346: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_124, 1, 0, 256)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_288: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_281, slice_346);  add_281 = slice_346 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_350: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_124, 1, 256, 9223372036854775807);  convolution_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_76: "f32[8, 120, 56, 56]" = torch.ops.aten.cat.default([cat_74, slice_350], 1);  cat_74 = slice_350 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_77: "f32[8, 376, 56, 56]" = torch.ops.aten.cat.default([add_288, cat_76], 1);  add_288 = cat_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1000: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_1001: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, -1);  unsqueeze_1000 = None
        sub_125: "f32[8, 376, 56, 56]" = torch.ops.aten.sub.Tensor(cat_77, unsqueeze_1001);  unsqueeze_1001 = None
        add_289: "f32[376]" = torch.ops.aten.add.Tensor(arg72_1, 0.001);  arg72_1 = None
        sqrt_125: "f32[376]" = torch.ops.aten.sqrt.default(add_289);  add_289 = None
        reciprocal_125: "f32[376]" = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
        mul_375: "f32[376]" = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
        unsqueeze_1002: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(mul_375, -1);  mul_375 = None
        unsqueeze_1003: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, -1);  unsqueeze_1002 = None
        mul_376: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_1003);  sub_125 = unsqueeze_1003 = None
        unsqueeze_1004: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_1005: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, -1);  unsqueeze_1004 = None
        mul_377: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(mul_376, unsqueeze_1005);  mul_376 = unsqueeze_1005 = None
        unsqueeze_1006: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_1007: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, -1);  unsqueeze_1006 = None
        add_290: "f32[8, 376, 56, 56]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_1007);  mul_377 = unsqueeze_1007 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_125: "f32[8, 376, 56, 56]" = torch.ops.aten.relu.default(add_290);  add_290 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_125: "f32[8, 640, 28, 28]" = torch.ops.aten.convolution.default(relu_125, arg75_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_125 = arg75_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:133 in forward, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
        slice_354: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_125, 1, 0, 512)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1008: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_1009: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, -1);  unsqueeze_1008 = None
        sub_126: "f32[8, 376, 56, 56]" = torch.ops.aten.sub.Tensor(cat_77, unsqueeze_1009);  cat_77 = unsqueeze_1009 = None
        add_291: "f32[376]" = torch.ops.aten.add.Tensor(arg77_1, 0.001);  arg77_1 = None
        sqrt_126: "f32[376]" = torch.ops.aten.sqrt.default(add_291);  add_291 = None
        reciprocal_126: "f32[376]" = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
        mul_378: "f32[376]" = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
        unsqueeze_1010: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
        unsqueeze_1011: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, -1);  unsqueeze_1010 = None
        mul_379: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_1011);  sub_126 = unsqueeze_1011 = None
        unsqueeze_1012: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_1013: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, -1);  unsqueeze_1012 = None
        mul_380: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_1013);  mul_379 = unsqueeze_1013 = None
        unsqueeze_1014: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_1015: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, -1);  unsqueeze_1014 = None
        add_292: "f32[8, 376, 56, 56]" = torch.ops.aten.add.Tensor(mul_380, unsqueeze_1015);  mul_380 = unsqueeze_1015 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_126: "f32[8, 376, 56, 56]" = torch.ops.aten.relu.default(add_292);  add_292 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_126: "f32[8, 400, 56, 56]" = torch.ops.aten.convolution.default(relu_126, arg80_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_126 = arg80_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1016: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_1017: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, -1);  unsqueeze_1016 = None
        sub_127: "f32[8, 400, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_1017);  convolution_126 = unsqueeze_1017 = None
        add_293: "f32[400]" = torch.ops.aten.add.Tensor(arg82_1, 0.001);  arg82_1 = None
        sqrt_127: "f32[400]" = torch.ops.aten.sqrt.default(add_293);  add_293 = None
        reciprocal_127: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
        mul_381: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
        unsqueeze_1018: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_381, -1);  mul_381 = None
        unsqueeze_1019: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, -1);  unsqueeze_1018 = None
        mul_382: "f32[8, 400, 56, 56]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_1019);  sub_127 = unsqueeze_1019 = None
        unsqueeze_1020: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_1021: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, -1);  unsqueeze_1020 = None
        mul_383: "f32[8, 400, 56, 56]" = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_1021);  mul_382 = unsqueeze_1021 = None
        unsqueeze_1022: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_1023: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, -1);  unsqueeze_1022 = None
        add_294: "f32[8, 400, 56, 56]" = torch.ops.aten.add.Tensor(mul_383, unsqueeze_1023);  mul_383 = unsqueeze_1023 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_127: "f32[8, 400, 56, 56]" = torch.ops.aten.relu.default(add_294);  add_294 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_127: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_127, arg85_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 50);  relu_127 = arg85_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1024: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_1025: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, -1);  unsqueeze_1024 = None
        sub_128: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_1025);  convolution_127 = unsqueeze_1025 = None
        add_295: "f32[400]" = torch.ops.aten.add.Tensor(arg87_1, 0.001);  arg87_1 = None
        sqrt_128: "f32[400]" = torch.ops.aten.sqrt.default(add_295);  add_295 = None
        reciprocal_128: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_128);  sqrt_128 = None
        mul_384: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_128, 1);  reciprocal_128 = None
        unsqueeze_1026: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_384, -1);  mul_384 = None
        unsqueeze_1027: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, -1);  unsqueeze_1026 = None
        mul_385: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_1027);  sub_128 = unsqueeze_1027 = None
        unsqueeze_1028: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_1029: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, -1);  unsqueeze_1028 = None
        mul_386: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_1029);  mul_385 = unsqueeze_1029 = None
        unsqueeze_1030: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_1031: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, -1);  unsqueeze_1030 = None
        add_296: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_386, unsqueeze_1031);  mul_386 = unsqueeze_1031 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_128: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_296);  add_296 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_128: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_128, arg90_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_128 = arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_362: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_128, 1, 0, 512)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_297: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(slice_354, slice_362);  slice_354 = slice_362 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:134 in forward, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
        slice_358: "f32[8, 128, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_125, 1, 512, 9223372036854775807);  convolution_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_366: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_128, 1, 512, 9223372036854775807);  convolution_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_78: "f32[8, 192, 28, 28]" = torch.ops.aten.cat.default([slice_358, slice_366], 1);  slice_358 = slice_366 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_79: "f32[8, 704, 28, 28]" = torch.ops.aten.cat.default([add_297, cat_78], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1032: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_1033: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, -1);  unsqueeze_1032 = None
        sub_129: "f32[8, 704, 28, 28]" = torch.ops.aten.sub.Tensor(cat_79, unsqueeze_1033);  cat_79 = unsqueeze_1033 = None
        add_298: "f32[704]" = torch.ops.aten.add.Tensor(arg92_1, 0.001);  arg92_1 = None
        sqrt_129: "f32[704]" = torch.ops.aten.sqrt.default(add_298);  add_298 = None
        reciprocal_129: "f32[704]" = torch.ops.aten.reciprocal.default(sqrt_129);  sqrt_129 = None
        mul_387: "f32[704]" = torch.ops.aten.mul.Tensor(reciprocal_129, 1);  reciprocal_129 = None
        unsqueeze_1034: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(mul_387, -1);  mul_387 = None
        unsqueeze_1035: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, -1);  unsqueeze_1034 = None
        mul_388: "f32[8, 704, 28, 28]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_1035);  sub_129 = unsqueeze_1035 = None
        unsqueeze_1036: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_1037: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, -1);  unsqueeze_1036 = None
        mul_389: "f32[8, 704, 28, 28]" = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_1037);  mul_388 = unsqueeze_1037 = None
        unsqueeze_1038: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_1039: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, -1);  unsqueeze_1038 = None
        add_299: "f32[8, 704, 28, 28]" = torch.ops.aten.add.Tensor(mul_389, unsqueeze_1039);  mul_389 = unsqueeze_1039 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_129: "f32[8, 704, 28, 28]" = torch.ops.aten.relu.default(add_299);  add_299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_129: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_129, arg95_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_129 = arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1040: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_1041: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, -1);  unsqueeze_1040 = None
        sub_130: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_1041);  convolution_129 = unsqueeze_1041 = None
        add_300: "f32[400]" = torch.ops.aten.add.Tensor(arg97_1, 0.001);  arg97_1 = None
        sqrt_130: "f32[400]" = torch.ops.aten.sqrt.default(add_300);  add_300 = None
        reciprocal_130: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_130);  sqrt_130 = None
        mul_390: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_130, 1);  reciprocal_130 = None
        unsqueeze_1042: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_390, -1);  mul_390 = None
        unsqueeze_1043: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, -1);  unsqueeze_1042 = None
        mul_391: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_1043);  sub_130 = unsqueeze_1043 = None
        unsqueeze_1044: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_1045: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, -1);  unsqueeze_1044 = None
        mul_392: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_391, unsqueeze_1045);  mul_391 = unsqueeze_1045 = None
        unsqueeze_1046: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_1047: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, -1);  unsqueeze_1046 = None
        add_301: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_392, unsqueeze_1047);  mul_392 = unsqueeze_1047 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_130: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_301);  add_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_130: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_130, arg100_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_130 = arg100_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1048: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_1049: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, -1);  unsqueeze_1048 = None
        sub_131: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_1049);  convolution_130 = unsqueeze_1049 = None
        add_302: "f32[400]" = torch.ops.aten.add.Tensor(arg102_1, 0.001);  arg102_1 = None
        sqrt_131: "f32[400]" = torch.ops.aten.sqrt.default(add_302);  add_302 = None
        reciprocal_131: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_131);  sqrt_131 = None
        mul_393: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_131, 1);  reciprocal_131 = None
        unsqueeze_1050: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_393, -1);  mul_393 = None
        unsqueeze_1051: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, -1);  unsqueeze_1050 = None
        mul_394: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_1051);  sub_131 = unsqueeze_1051 = None
        unsqueeze_1052: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_1053: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, -1);  unsqueeze_1052 = None
        mul_395: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_394, unsqueeze_1053);  mul_394 = unsqueeze_1053 = None
        unsqueeze_1054: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_1055: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, -1);  unsqueeze_1054 = None
        add_303: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_395, unsqueeze_1055);  mul_395 = unsqueeze_1055 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_131: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_303);  add_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_131: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_131, arg105_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_131 = arg105_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_370: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_131, 1, 0, 512)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_304: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_297, slice_370);  add_297 = slice_370 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_374: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_131, 1, 512, 9223372036854775807);  convolution_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_80: "f32[8, 256, 28, 28]" = torch.ops.aten.cat.default([cat_78, slice_374], 1);  cat_78 = slice_374 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_81: "f32[8, 768, 28, 28]" = torch.ops.aten.cat.default([add_304, cat_80], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1056: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_1057: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, -1);  unsqueeze_1056 = None
        sub_132: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(cat_81, unsqueeze_1057);  cat_81 = unsqueeze_1057 = None
        add_305: "f32[768]" = torch.ops.aten.add.Tensor(arg107_1, 0.001);  arg107_1 = None
        sqrt_132: "f32[768]" = torch.ops.aten.sqrt.default(add_305);  add_305 = None
        reciprocal_132: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_132);  sqrt_132 = None
        mul_396: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_132, 1);  reciprocal_132 = None
        unsqueeze_1058: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_396, -1);  mul_396 = None
        unsqueeze_1059: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, -1);  unsqueeze_1058 = None
        mul_397: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_1059);  sub_132 = unsqueeze_1059 = None
        unsqueeze_1060: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_1061: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, -1);  unsqueeze_1060 = None
        mul_398: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_397, unsqueeze_1061);  mul_397 = unsqueeze_1061 = None
        unsqueeze_1062: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_1063: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, -1);  unsqueeze_1062 = None
        add_306: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_1063);  mul_398 = unsqueeze_1063 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_132: "f32[8, 768, 28, 28]" = torch.ops.aten.relu.default(add_306);  add_306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_132: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_132, arg110_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_132 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1064: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_1065: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, -1);  unsqueeze_1064 = None
        sub_133: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_1065);  convolution_132 = unsqueeze_1065 = None
        add_307: "f32[400]" = torch.ops.aten.add.Tensor(arg112_1, 0.001);  arg112_1 = None
        sqrt_133: "f32[400]" = torch.ops.aten.sqrt.default(add_307);  add_307 = None
        reciprocal_133: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_133);  sqrt_133 = None
        mul_399: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_133, 1);  reciprocal_133 = None
        unsqueeze_1066: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_399, -1);  mul_399 = None
        unsqueeze_1067: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, -1);  unsqueeze_1066 = None
        mul_400: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_1067);  sub_133 = unsqueeze_1067 = None
        unsqueeze_1068: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_1069: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, -1);  unsqueeze_1068 = None
        mul_401: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_400, unsqueeze_1069);  mul_400 = unsqueeze_1069 = None
        unsqueeze_1070: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_1071: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, -1);  unsqueeze_1070 = None
        add_308: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_401, unsqueeze_1071);  mul_401 = unsqueeze_1071 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_133: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_308);  add_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_133: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_133, arg115_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_133 = arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1072: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_1073: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, -1);  unsqueeze_1072 = None
        sub_134: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_1073);  convolution_133 = unsqueeze_1073 = None
        add_309: "f32[400]" = torch.ops.aten.add.Tensor(arg117_1, 0.001);  arg117_1 = None
        sqrt_134: "f32[400]" = torch.ops.aten.sqrt.default(add_309);  add_309 = None
        reciprocal_134: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_134);  sqrt_134 = None
        mul_402: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_134, 1);  reciprocal_134 = None
        unsqueeze_1074: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_402, -1);  mul_402 = None
        unsqueeze_1075: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, -1);  unsqueeze_1074 = None
        mul_403: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_1075);  sub_134 = unsqueeze_1075 = None
        unsqueeze_1076: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_1077: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, -1);  unsqueeze_1076 = None
        mul_404: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_403, unsqueeze_1077);  mul_403 = unsqueeze_1077 = None
        unsqueeze_1078: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_1079: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, -1);  unsqueeze_1078 = None
        add_310: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_404, unsqueeze_1079);  mul_404 = unsqueeze_1079 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_134: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_310);  add_310 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_134: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_134, arg120_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_134 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_378: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_134, 1, 0, 512)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_311: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_304, slice_378);  add_304 = slice_378 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_382: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_134, 1, 512, 9223372036854775807);  convolution_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_82: "f32[8, 320, 28, 28]" = torch.ops.aten.cat.default([cat_80, slice_382], 1);  cat_80 = slice_382 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_83: "f32[8, 832, 28, 28]" = torch.ops.aten.cat.default([add_311, cat_82], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1080: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_1081: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, -1);  unsqueeze_1080 = None
        sub_135: "f32[8, 832, 28, 28]" = torch.ops.aten.sub.Tensor(cat_83, unsqueeze_1081);  cat_83 = unsqueeze_1081 = None
        add_312: "f32[832]" = torch.ops.aten.add.Tensor(arg122_1, 0.001);  arg122_1 = None
        sqrt_135: "f32[832]" = torch.ops.aten.sqrt.default(add_312);  add_312 = None
        reciprocal_135: "f32[832]" = torch.ops.aten.reciprocal.default(sqrt_135);  sqrt_135 = None
        mul_405: "f32[832]" = torch.ops.aten.mul.Tensor(reciprocal_135, 1);  reciprocal_135 = None
        unsqueeze_1082: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(mul_405, -1);  mul_405 = None
        unsqueeze_1083: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, -1);  unsqueeze_1082 = None
        mul_406: "f32[8, 832, 28, 28]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_1083);  sub_135 = unsqueeze_1083 = None
        unsqueeze_1084: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_1085: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, -1);  unsqueeze_1084 = None
        mul_407: "f32[8, 832, 28, 28]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_1085);  mul_406 = unsqueeze_1085 = None
        unsqueeze_1086: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_1087: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, -1);  unsqueeze_1086 = None
        add_313: "f32[8, 832, 28, 28]" = torch.ops.aten.add.Tensor(mul_407, unsqueeze_1087);  mul_407 = unsqueeze_1087 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_135: "f32[8, 832, 28, 28]" = torch.ops.aten.relu.default(add_313);  add_313 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_135: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_135, arg125_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_135 = arg125_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1088: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_1089: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, -1);  unsqueeze_1088 = None
        sub_136: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_1089);  convolution_135 = unsqueeze_1089 = None
        add_314: "f32[400]" = torch.ops.aten.add.Tensor(arg127_1, 0.001);  arg127_1 = None
        sqrt_136: "f32[400]" = torch.ops.aten.sqrt.default(add_314);  add_314 = None
        reciprocal_136: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_136);  sqrt_136 = None
        mul_408: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_136, 1);  reciprocal_136 = None
        unsqueeze_1090: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_408, -1);  mul_408 = None
        unsqueeze_1091: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, -1);  unsqueeze_1090 = None
        mul_409: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_1091);  sub_136 = unsqueeze_1091 = None
        unsqueeze_1092: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_1093: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, -1);  unsqueeze_1092 = None
        mul_410: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_409, unsqueeze_1093);  mul_409 = unsqueeze_1093 = None
        unsqueeze_1094: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_1095: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, -1);  unsqueeze_1094 = None
        add_315: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_410, unsqueeze_1095);  mul_410 = unsqueeze_1095 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_136: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_315);  add_315 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_136: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_136, arg130_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_136 = arg130_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1096: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_1097: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, -1);  unsqueeze_1096 = None
        sub_137: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_1097);  convolution_136 = unsqueeze_1097 = None
        add_316: "f32[400]" = torch.ops.aten.add.Tensor(arg132_1, 0.001);  arg132_1 = None
        sqrt_137: "f32[400]" = torch.ops.aten.sqrt.default(add_316);  add_316 = None
        reciprocal_137: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_137);  sqrt_137 = None
        mul_411: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_137, 1);  reciprocal_137 = None
        unsqueeze_1098: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
        unsqueeze_1099: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, -1);  unsqueeze_1098 = None
        mul_412: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_1099);  sub_137 = unsqueeze_1099 = None
        unsqueeze_1100: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_1101: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, -1);  unsqueeze_1100 = None
        mul_413: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_1101);  mul_412 = unsqueeze_1101 = None
        unsqueeze_1102: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_1103: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, -1);  unsqueeze_1102 = None
        add_317: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_413, unsqueeze_1103);  mul_413 = unsqueeze_1103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_137: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_317);  add_317 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_137: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_137, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_137 = arg135_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_386: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_137, 1, 0, 512)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_318: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_311, slice_386);  add_311 = slice_386 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_390: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_137, 1, 512, 9223372036854775807);  convolution_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_84: "f32[8, 384, 28, 28]" = torch.ops.aten.cat.default([cat_82, slice_390], 1);  cat_82 = slice_390 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_85: "f32[8, 896, 28, 28]" = torch.ops.aten.cat.default([add_318, cat_84], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1104: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_1105: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, -1);  unsqueeze_1104 = None
        sub_138: "f32[8, 896, 28, 28]" = torch.ops.aten.sub.Tensor(cat_85, unsqueeze_1105);  cat_85 = unsqueeze_1105 = None
        add_319: "f32[896]" = torch.ops.aten.add.Tensor(arg137_1, 0.001);  arg137_1 = None
        sqrt_138: "f32[896]" = torch.ops.aten.sqrt.default(add_319);  add_319 = None
        reciprocal_138: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_138);  sqrt_138 = None
        mul_414: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_138, 1);  reciprocal_138 = None
        unsqueeze_1106: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_414, -1);  mul_414 = None
        unsqueeze_1107: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, -1);  unsqueeze_1106 = None
        mul_415: "f32[8, 896, 28, 28]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_1107);  sub_138 = unsqueeze_1107 = None
        unsqueeze_1108: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_1109: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, -1);  unsqueeze_1108 = None
        mul_416: "f32[8, 896, 28, 28]" = torch.ops.aten.mul.Tensor(mul_415, unsqueeze_1109);  mul_415 = unsqueeze_1109 = None
        unsqueeze_1110: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_1111: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, -1);  unsqueeze_1110 = None
        add_320: "f32[8, 896, 28, 28]" = torch.ops.aten.add.Tensor(mul_416, unsqueeze_1111);  mul_416 = unsqueeze_1111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_138: "f32[8, 896, 28, 28]" = torch.ops.aten.relu.default(add_320);  add_320 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_138: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_138, arg140_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_138 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1112: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_1113: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, -1);  unsqueeze_1112 = None
        sub_139: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_1113);  convolution_138 = unsqueeze_1113 = None
        add_321: "f32[400]" = torch.ops.aten.add.Tensor(arg142_1, 0.001);  arg142_1 = None
        sqrt_139: "f32[400]" = torch.ops.aten.sqrt.default(add_321);  add_321 = None
        reciprocal_139: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_139);  sqrt_139 = None
        mul_417: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_139, 1);  reciprocal_139 = None
        unsqueeze_1114: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_417, -1);  mul_417 = None
        unsqueeze_1115: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, -1);  unsqueeze_1114 = None
        mul_418: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_1115);  sub_139 = unsqueeze_1115 = None
        unsqueeze_1116: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_1117: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, -1);  unsqueeze_1116 = None
        mul_419: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_418, unsqueeze_1117);  mul_418 = unsqueeze_1117 = None
        unsqueeze_1118: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_1119: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, -1);  unsqueeze_1118 = None
        add_322: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_1119);  mul_419 = unsqueeze_1119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_139: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_322);  add_322 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_139: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_139, arg145_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_139 = arg145_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1120: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_1121: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, -1);  unsqueeze_1120 = None
        sub_140: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_1121);  convolution_139 = unsqueeze_1121 = None
        add_323: "f32[400]" = torch.ops.aten.add.Tensor(arg147_1, 0.001);  arg147_1 = None
        sqrt_140: "f32[400]" = torch.ops.aten.sqrt.default(add_323);  add_323 = None
        reciprocal_140: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_140);  sqrt_140 = None
        mul_420: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_140, 1);  reciprocal_140 = None
        unsqueeze_1122: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_420, -1);  mul_420 = None
        unsqueeze_1123: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, -1);  unsqueeze_1122 = None
        mul_421: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_1123);  sub_140 = unsqueeze_1123 = None
        unsqueeze_1124: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_1125: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, -1);  unsqueeze_1124 = None
        mul_422: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_421, unsqueeze_1125);  mul_421 = unsqueeze_1125 = None
        unsqueeze_1126: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_1127: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, -1);  unsqueeze_1126 = None
        add_324: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_422, unsqueeze_1127);  mul_422 = unsqueeze_1127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_140: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_324);  add_324 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_140: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_140, arg150_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_140 = arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_394: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_140, 1, 0, 512)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_325: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_318, slice_394);  add_318 = slice_394 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_398: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_140, 1, 512, 9223372036854775807);  convolution_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_86: "f32[8, 448, 28, 28]" = torch.ops.aten.cat.default([cat_84, slice_398], 1);  cat_84 = slice_398 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_87: "f32[8, 960, 28, 28]" = torch.ops.aten.cat.default([add_325, cat_86], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1128: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_1129: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, -1);  unsqueeze_1128 = None
        sub_141: "f32[8, 960, 28, 28]" = torch.ops.aten.sub.Tensor(cat_87, unsqueeze_1129);  cat_87 = unsqueeze_1129 = None
        add_326: "f32[960]" = torch.ops.aten.add.Tensor(arg152_1, 0.001);  arg152_1 = None
        sqrt_141: "f32[960]" = torch.ops.aten.sqrt.default(add_326);  add_326 = None
        reciprocal_141: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_141);  sqrt_141 = None
        mul_423: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_141, 1);  reciprocal_141 = None
        unsqueeze_1130: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_423, -1);  mul_423 = None
        unsqueeze_1131: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, -1);  unsqueeze_1130 = None
        mul_424: "f32[8, 960, 28, 28]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_1131);  sub_141 = unsqueeze_1131 = None
        unsqueeze_1132: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_1133: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, -1);  unsqueeze_1132 = None
        mul_425: "f32[8, 960, 28, 28]" = torch.ops.aten.mul.Tensor(mul_424, unsqueeze_1133);  mul_424 = unsqueeze_1133 = None
        unsqueeze_1134: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_1135: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, -1);  unsqueeze_1134 = None
        add_327: "f32[8, 960, 28, 28]" = torch.ops.aten.add.Tensor(mul_425, unsqueeze_1135);  mul_425 = unsqueeze_1135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_141: "f32[8, 960, 28, 28]" = torch.ops.aten.relu.default(add_327);  add_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_141: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_141, arg155_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_141 = arg155_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1136: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_1137: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, -1);  unsqueeze_1136 = None
        sub_142: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_1137);  convolution_141 = unsqueeze_1137 = None
        add_328: "f32[400]" = torch.ops.aten.add.Tensor(arg157_1, 0.001);  arg157_1 = None
        sqrt_142: "f32[400]" = torch.ops.aten.sqrt.default(add_328);  add_328 = None
        reciprocal_142: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_142);  sqrt_142 = None
        mul_426: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_142, 1);  reciprocal_142 = None
        unsqueeze_1138: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_426, -1);  mul_426 = None
        unsqueeze_1139: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, -1);  unsqueeze_1138 = None
        mul_427: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_1139);  sub_142 = unsqueeze_1139 = None
        unsqueeze_1140: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_1141: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, -1);  unsqueeze_1140 = None
        mul_428: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_1141);  mul_427 = unsqueeze_1141 = None
        unsqueeze_1142: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_1143: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, -1);  unsqueeze_1142 = None
        add_329: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_428, unsqueeze_1143);  mul_428 = unsqueeze_1143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_142: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_329);  add_329 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_142: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_142, arg160_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_142 = arg160_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1144: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_1145: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, -1);  unsqueeze_1144 = None
        sub_143: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_1145);  convolution_142 = unsqueeze_1145 = None
        add_330: "f32[400]" = torch.ops.aten.add.Tensor(arg162_1, 0.001);  arg162_1 = None
        sqrt_143: "f32[400]" = torch.ops.aten.sqrt.default(add_330);  add_330 = None
        reciprocal_143: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_143);  sqrt_143 = None
        mul_429: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_143, 1);  reciprocal_143 = None
        unsqueeze_1146: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_429, -1);  mul_429 = None
        unsqueeze_1147: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, -1);  unsqueeze_1146 = None
        mul_430: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_1147);  sub_143 = unsqueeze_1147 = None
        unsqueeze_1148: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_1149: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, -1);  unsqueeze_1148 = None
        mul_431: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_430, unsqueeze_1149);  mul_430 = unsqueeze_1149 = None
        unsqueeze_1150: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_1151: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, -1);  unsqueeze_1150 = None
        add_331: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_431, unsqueeze_1151);  mul_431 = unsqueeze_1151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_143: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_331);  add_331 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_143: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_143, arg165_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_143 = arg165_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_402: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_143, 1, 0, 512)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_332: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_325, slice_402);  add_325 = slice_402 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_406: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_143, 1, 512, 9223372036854775807);  convolution_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_88: "f32[8, 512, 28, 28]" = torch.ops.aten.cat.default([cat_86, slice_406], 1);  cat_86 = slice_406 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_89: "f32[8, 1024, 28, 28]" = torch.ops.aten.cat.default([add_332, cat_88], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1152: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_1153: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, -1);  unsqueeze_1152 = None
        sub_144: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(cat_89, unsqueeze_1153);  cat_89 = unsqueeze_1153 = None
        add_333: "f32[1024]" = torch.ops.aten.add.Tensor(arg167_1, 0.001);  arg167_1 = None
        sqrt_144: "f32[1024]" = torch.ops.aten.sqrt.default(add_333);  add_333 = None
        reciprocal_144: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_144);  sqrt_144 = None
        mul_432: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_144, 1);  reciprocal_144 = None
        unsqueeze_1154: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_432, -1);  mul_432 = None
        unsqueeze_1155: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, -1);  unsqueeze_1154 = None
        mul_433: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_1155);  sub_144 = unsqueeze_1155 = None
        unsqueeze_1156: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_1157: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, -1);  unsqueeze_1156 = None
        mul_434: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_1157);  mul_433 = unsqueeze_1157 = None
        unsqueeze_1158: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_1159: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, -1);  unsqueeze_1158 = None
        add_334: "f32[8, 1024, 28, 28]" = torch.ops.aten.add.Tensor(mul_434, unsqueeze_1159);  mul_434 = unsqueeze_1159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_144: "f32[8, 1024, 28, 28]" = torch.ops.aten.relu.default(add_334);  add_334 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_144: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_144, arg170_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_144 = arg170_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1160: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_1161: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, -1);  unsqueeze_1160 = None
        sub_145: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_144, unsqueeze_1161);  convolution_144 = unsqueeze_1161 = None
        add_335: "f32[400]" = torch.ops.aten.add.Tensor(arg172_1, 0.001);  arg172_1 = None
        sqrt_145: "f32[400]" = torch.ops.aten.sqrt.default(add_335);  add_335 = None
        reciprocal_145: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_145);  sqrt_145 = None
        mul_435: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_145, 1);  reciprocal_145 = None
        unsqueeze_1162: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_435, -1);  mul_435 = None
        unsqueeze_1163: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, -1);  unsqueeze_1162 = None
        mul_436: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_1163);  sub_145 = unsqueeze_1163 = None
        unsqueeze_1164: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_1165: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, -1);  unsqueeze_1164 = None
        mul_437: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_436, unsqueeze_1165);  mul_436 = unsqueeze_1165 = None
        unsqueeze_1166: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_1167: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, -1);  unsqueeze_1166 = None
        add_336: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_437, unsqueeze_1167);  mul_437 = unsqueeze_1167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_145: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_336);  add_336 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_145: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_145, arg175_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_145 = arg175_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1168: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
        unsqueeze_1169: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, -1);  unsqueeze_1168 = None
        sub_146: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_1169);  convolution_145 = unsqueeze_1169 = None
        add_337: "f32[400]" = torch.ops.aten.add.Tensor(arg177_1, 0.001);  arg177_1 = None
        sqrt_146: "f32[400]" = torch.ops.aten.sqrt.default(add_337);  add_337 = None
        reciprocal_146: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_146);  sqrt_146 = None
        mul_438: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_146, 1);  reciprocal_146 = None
        unsqueeze_1170: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_438, -1);  mul_438 = None
        unsqueeze_1171: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, -1);  unsqueeze_1170 = None
        mul_439: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_1171);  sub_146 = unsqueeze_1171 = None
        unsqueeze_1172: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_1173: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, -1);  unsqueeze_1172 = None
        mul_440: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_439, unsqueeze_1173);  mul_439 = unsqueeze_1173 = None
        unsqueeze_1174: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_1175: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, -1);  unsqueeze_1174 = None
        add_338: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_1175);  mul_440 = unsqueeze_1175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_146: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_338);  add_338 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_146: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_146, arg180_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_146 = arg180_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_410: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_146, 1, 0, 512)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_339: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_332, slice_410);  add_332 = slice_410 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_414: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_146, 1, 512, 9223372036854775807);  convolution_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_90: "f32[8, 576, 28, 28]" = torch.ops.aten.cat.default([cat_88, slice_414], 1);  cat_88 = slice_414 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_91: "f32[8, 1088, 28, 28]" = torch.ops.aten.cat.default([add_339, cat_90], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1176: "f32[1088, 1]" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_1177: "f32[1088, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, -1);  unsqueeze_1176 = None
        sub_147: "f32[8, 1088, 28, 28]" = torch.ops.aten.sub.Tensor(cat_91, unsqueeze_1177);  cat_91 = unsqueeze_1177 = None
        add_340: "f32[1088]" = torch.ops.aten.add.Tensor(arg182_1, 0.001);  arg182_1 = None
        sqrt_147: "f32[1088]" = torch.ops.aten.sqrt.default(add_340);  add_340 = None
        reciprocal_147: "f32[1088]" = torch.ops.aten.reciprocal.default(sqrt_147);  sqrt_147 = None
        mul_441: "f32[1088]" = torch.ops.aten.mul.Tensor(reciprocal_147, 1);  reciprocal_147 = None
        unsqueeze_1178: "f32[1088, 1]" = torch.ops.aten.unsqueeze.default(mul_441, -1);  mul_441 = None
        unsqueeze_1179: "f32[1088, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, -1);  unsqueeze_1178 = None
        mul_442: "f32[8, 1088, 28, 28]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_1179);  sub_147 = unsqueeze_1179 = None
        unsqueeze_1180: "f32[1088, 1]" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_1181: "f32[1088, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, -1);  unsqueeze_1180 = None
        mul_443: "f32[8, 1088, 28, 28]" = torch.ops.aten.mul.Tensor(mul_442, unsqueeze_1181);  mul_442 = unsqueeze_1181 = None
        unsqueeze_1182: "f32[1088, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_1183: "f32[1088, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, -1);  unsqueeze_1182 = None
        add_341: "f32[8, 1088, 28, 28]" = torch.ops.aten.add.Tensor(mul_443, unsqueeze_1183);  mul_443 = unsqueeze_1183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_147: "f32[8, 1088, 28, 28]" = torch.ops.aten.relu.default(add_341);  add_341 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_147: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_147, arg185_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_147 = arg185_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1184: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
        unsqueeze_1185: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, -1);  unsqueeze_1184 = None
        sub_148: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_1185);  convolution_147 = unsqueeze_1185 = None
        add_342: "f32[400]" = torch.ops.aten.add.Tensor(arg187_1, 0.001);  arg187_1 = None
        sqrt_148: "f32[400]" = torch.ops.aten.sqrt.default(add_342);  add_342 = None
        reciprocal_148: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_148);  sqrt_148 = None
        mul_444: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_148, 1);  reciprocal_148 = None
        unsqueeze_1186: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_444, -1);  mul_444 = None
        unsqueeze_1187: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, -1);  unsqueeze_1186 = None
        mul_445: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_1187);  sub_148 = unsqueeze_1187 = None
        unsqueeze_1188: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_1189: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, -1);  unsqueeze_1188 = None
        mul_446: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_1189);  mul_445 = unsqueeze_1189 = None
        unsqueeze_1190: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_1191: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, -1);  unsqueeze_1190 = None
        add_343: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_446, unsqueeze_1191);  mul_446 = unsqueeze_1191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_148: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_343);  add_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_148: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_148, arg190_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_148 = arg190_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1192: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
        unsqueeze_1193: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, -1);  unsqueeze_1192 = None
        sub_149: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_148, unsqueeze_1193);  convolution_148 = unsqueeze_1193 = None
        add_344: "f32[400]" = torch.ops.aten.add.Tensor(arg192_1, 0.001);  arg192_1 = None
        sqrt_149: "f32[400]" = torch.ops.aten.sqrt.default(add_344);  add_344 = None
        reciprocal_149: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_149);  sqrt_149 = None
        mul_447: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_149, 1);  reciprocal_149 = None
        unsqueeze_1194: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_447, -1);  mul_447 = None
        unsqueeze_1195: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, -1);  unsqueeze_1194 = None
        mul_448: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_1195);  sub_149 = unsqueeze_1195 = None
        unsqueeze_1196: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
        unsqueeze_1197: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, -1);  unsqueeze_1196 = None
        mul_449: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_1197);  mul_448 = unsqueeze_1197 = None
        unsqueeze_1198: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_1199: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, -1);  unsqueeze_1198 = None
        add_345: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_449, unsqueeze_1199);  mul_449 = unsqueeze_1199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_149: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_345);  add_345 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_149: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_149, arg195_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_149 = arg195_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_418: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_149, 1, 0, 512)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_346: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_339, slice_418);  add_339 = slice_418 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_422: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_149, 1, 512, 9223372036854775807);  convolution_149 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_92: "f32[8, 640, 28, 28]" = torch.ops.aten.cat.default([cat_90, slice_422], 1);  cat_90 = slice_422 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_93: "f32[8, 1152, 28, 28]" = torch.ops.aten.cat.default([add_346, cat_92], 1);  add_346 = cat_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1200: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
        unsqueeze_1201: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, -1);  unsqueeze_1200 = None
        sub_150: "f32[8, 1152, 28, 28]" = torch.ops.aten.sub.Tensor(cat_93, unsqueeze_1201);  unsqueeze_1201 = None
        add_347: "f32[1152]" = torch.ops.aten.add.Tensor(arg197_1, 0.001);  arg197_1 = None
        sqrt_150: "f32[1152]" = torch.ops.aten.sqrt.default(add_347);  add_347 = None
        reciprocal_150: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_150);  sqrt_150 = None
        mul_450: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_150, 1);  reciprocal_150 = None
        unsqueeze_1202: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_450, -1);  mul_450 = None
        unsqueeze_1203: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, -1);  unsqueeze_1202 = None
        mul_451: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_1203);  sub_150 = unsqueeze_1203 = None
        unsqueeze_1204: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_1205: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, -1);  unsqueeze_1204 = None
        mul_452: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(mul_451, unsqueeze_1205);  mul_451 = unsqueeze_1205 = None
        unsqueeze_1206: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_1207: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, -1);  unsqueeze_1206 = None
        add_348: "f32[8, 1152, 28, 28]" = torch.ops.aten.add.Tensor(mul_452, unsqueeze_1207);  mul_452 = unsqueeze_1207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_150: "f32[8, 1152, 28, 28]" = torch.ops.aten.relu.default(add_348);  add_348 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_150: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(relu_150, arg200_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_150 = arg200_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:133 in forward, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
        slice_426: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_150, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1208: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg201_1, -1);  arg201_1 = None
        unsqueeze_1209: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, -1);  unsqueeze_1208 = None
        sub_151: "f32[8, 1152, 28, 28]" = torch.ops.aten.sub.Tensor(cat_93, unsqueeze_1209);  cat_93 = unsqueeze_1209 = None
        add_349: "f32[1152]" = torch.ops.aten.add.Tensor(arg202_1, 0.001);  arg202_1 = None
        sqrt_151: "f32[1152]" = torch.ops.aten.sqrt.default(add_349);  add_349 = None
        reciprocal_151: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_151);  sqrt_151 = None
        mul_453: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_151, 1);  reciprocal_151 = None
        unsqueeze_1210: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_453, -1);  mul_453 = None
        unsqueeze_1211: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, -1);  unsqueeze_1210 = None
        mul_454: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_1211);  sub_151 = unsqueeze_1211 = None
        unsqueeze_1212: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_1213: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, -1);  unsqueeze_1212 = None
        mul_455: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(mul_454, unsqueeze_1213);  mul_454 = unsqueeze_1213 = None
        unsqueeze_1214: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_1215: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, -1);  unsqueeze_1214 = None
        add_350: "f32[8, 1152, 28, 28]" = torch.ops.aten.add.Tensor(mul_455, unsqueeze_1215);  mul_455 = unsqueeze_1215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_151: "f32[8, 1152, 28, 28]" = torch.ops.aten.relu.default(add_350);  add_350 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_151: "f32[8, 800, 28, 28]" = torch.ops.aten.convolution.default(relu_151, arg205_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_151 = arg205_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1216: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_1217: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, -1);  unsqueeze_1216 = None
        sub_152: "f32[8, 800, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_1217);  convolution_151 = unsqueeze_1217 = None
        add_351: "f32[800]" = torch.ops.aten.add.Tensor(arg207_1, 0.001);  arg207_1 = None
        sqrt_152: "f32[800]" = torch.ops.aten.sqrt.default(add_351);  add_351 = None
        reciprocal_152: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_152);  sqrt_152 = None
        mul_456: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_152, 1);  reciprocal_152 = None
        unsqueeze_1218: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_456, -1);  mul_456 = None
        unsqueeze_1219: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, -1);  unsqueeze_1218 = None
        mul_457: "f32[8, 800, 28, 28]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_1219);  sub_152 = unsqueeze_1219 = None
        unsqueeze_1220: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_1221: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, -1);  unsqueeze_1220 = None
        mul_458: "f32[8, 800, 28, 28]" = torch.ops.aten.mul.Tensor(mul_457, unsqueeze_1221);  mul_457 = unsqueeze_1221 = None
        unsqueeze_1222: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_1223: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, -1);  unsqueeze_1222 = None
        add_352: "f32[8, 800, 28, 28]" = torch.ops.aten.add.Tensor(mul_458, unsqueeze_1223);  mul_458 = unsqueeze_1223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_152: "f32[8, 800, 28, 28]" = torch.ops.aten.relu.default(add_352);  add_352 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_152: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_152, arg210_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 50);  relu_152 = arg210_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1224: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_1225: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, -1);  unsqueeze_1224 = None
        sub_153: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_1225);  convolution_152 = unsqueeze_1225 = None
        add_353: "f32[800]" = torch.ops.aten.add.Tensor(arg212_1, 0.001);  arg212_1 = None
        sqrt_153: "f32[800]" = torch.ops.aten.sqrt.default(add_353);  add_353 = None
        reciprocal_153: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_153);  sqrt_153 = None
        mul_459: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_153, 1);  reciprocal_153 = None
        unsqueeze_1226: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_459, -1);  mul_459 = None
        unsqueeze_1227: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, -1);  unsqueeze_1226 = None
        mul_460: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_1227);  sub_153 = unsqueeze_1227 = None
        unsqueeze_1228: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_1229: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, -1);  unsqueeze_1228 = None
        mul_461: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_460, unsqueeze_1229);  mul_460 = unsqueeze_1229 = None
        unsqueeze_1230: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_1231: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, -1);  unsqueeze_1230 = None
        add_354: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_1231);  mul_461 = unsqueeze_1231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_153: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_354);  add_354 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_153: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_153, arg215_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_153 = arg215_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_434: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_153, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_355: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(slice_426, slice_434);  slice_426 = slice_434 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:134 in forward, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
        slice_430: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_150, 1, 1024, 9223372036854775807);  convolution_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_438: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_153, 1, 1024, 9223372036854775807);  convolution_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_94: "f32[8, 192, 14, 14]" = torch.ops.aten.cat.default([slice_430, slice_438], 1);  slice_430 = slice_438 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_95: "f32[8, 1216, 14, 14]" = torch.ops.aten.cat.default([add_355, cat_94], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1232: "f32[1216, 1]" = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_1233: "f32[1216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, -1);  unsqueeze_1232 = None
        sub_154: "f32[8, 1216, 14, 14]" = torch.ops.aten.sub.Tensor(cat_95, unsqueeze_1233);  cat_95 = unsqueeze_1233 = None
        add_356: "f32[1216]" = torch.ops.aten.add.Tensor(arg217_1, 0.001);  arg217_1 = None
        sqrt_154: "f32[1216]" = torch.ops.aten.sqrt.default(add_356);  add_356 = None
        reciprocal_154: "f32[1216]" = torch.ops.aten.reciprocal.default(sqrt_154);  sqrt_154 = None
        mul_462: "f32[1216]" = torch.ops.aten.mul.Tensor(reciprocal_154, 1);  reciprocal_154 = None
        unsqueeze_1234: "f32[1216, 1]" = torch.ops.aten.unsqueeze.default(mul_462, -1);  mul_462 = None
        unsqueeze_1235: "f32[1216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, -1);  unsqueeze_1234 = None
        mul_463: "f32[8, 1216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_1235);  sub_154 = unsqueeze_1235 = None
        unsqueeze_1236: "f32[1216, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_1237: "f32[1216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, -1);  unsqueeze_1236 = None
        mul_464: "f32[8, 1216, 14, 14]" = torch.ops.aten.mul.Tensor(mul_463, unsqueeze_1237);  mul_463 = unsqueeze_1237 = None
        unsqueeze_1238: "f32[1216, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_1239: "f32[1216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, -1);  unsqueeze_1238 = None
        add_357: "f32[8, 1216, 14, 14]" = torch.ops.aten.add.Tensor(mul_464, unsqueeze_1239);  mul_464 = unsqueeze_1239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_154: "f32[8, 1216, 14, 14]" = torch.ops.aten.relu.default(add_357);  add_357 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_154: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_154, arg220_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_154 = arg220_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1240: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_1241: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, -1);  unsqueeze_1240 = None
        sub_155: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_1241);  convolution_154 = unsqueeze_1241 = None
        add_358: "f32[800]" = torch.ops.aten.add.Tensor(arg222_1, 0.001);  arg222_1 = None
        sqrt_155: "f32[800]" = torch.ops.aten.sqrt.default(add_358);  add_358 = None
        reciprocal_155: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_155);  sqrt_155 = None
        mul_465: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_155, 1);  reciprocal_155 = None
        unsqueeze_1242: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_465, -1);  mul_465 = None
        unsqueeze_1243: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, -1);  unsqueeze_1242 = None
        mul_466: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_1243);  sub_155 = unsqueeze_1243 = None
        unsqueeze_1244: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_1245: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, -1);  unsqueeze_1244 = None
        mul_467: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_466, unsqueeze_1245);  mul_466 = unsqueeze_1245 = None
        unsqueeze_1246: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_1247: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, -1);  unsqueeze_1246 = None
        add_359: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_467, unsqueeze_1247);  mul_467 = unsqueeze_1247 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_155: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_359);  add_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_155: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_155, arg225_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_155 = arg225_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1248: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
        unsqueeze_1249: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, -1);  unsqueeze_1248 = None
        sub_156: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_1249);  convolution_155 = unsqueeze_1249 = None
        add_360: "f32[800]" = torch.ops.aten.add.Tensor(arg227_1, 0.001);  arg227_1 = None
        sqrt_156: "f32[800]" = torch.ops.aten.sqrt.default(add_360);  add_360 = None
        reciprocal_156: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_156);  sqrt_156 = None
        mul_468: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_156, 1);  reciprocal_156 = None
        unsqueeze_1250: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_468, -1);  mul_468 = None
        unsqueeze_1251: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, -1);  unsqueeze_1250 = None
        mul_469: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_1251);  sub_156 = unsqueeze_1251 = None
        unsqueeze_1252: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_1253: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, -1);  unsqueeze_1252 = None
        mul_470: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_1253);  mul_469 = unsqueeze_1253 = None
        unsqueeze_1254: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_1255: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, -1);  unsqueeze_1254 = None
        add_361: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_470, unsqueeze_1255);  mul_470 = unsqueeze_1255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_156: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_361);  add_361 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_156: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_156, arg230_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_156 = arg230_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_442: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_156, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_362: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_355, slice_442);  add_355 = slice_442 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_446: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_156, 1, 1024, 9223372036854775807);  convolution_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_96: "f32[8, 256, 14, 14]" = torch.ops.aten.cat.default([cat_94, slice_446], 1);  cat_94 = slice_446 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_97: "f32[8, 1280, 14, 14]" = torch.ops.aten.cat.default([add_362, cat_96], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1256: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg231_1, -1);  arg231_1 = None
        unsqueeze_1257: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, -1);  unsqueeze_1256 = None
        sub_157: "f32[8, 1280, 14, 14]" = torch.ops.aten.sub.Tensor(cat_97, unsqueeze_1257);  cat_97 = unsqueeze_1257 = None
        add_363: "f32[1280]" = torch.ops.aten.add.Tensor(arg232_1, 0.001);  arg232_1 = None
        sqrt_157: "f32[1280]" = torch.ops.aten.sqrt.default(add_363);  add_363 = None
        reciprocal_157: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_157);  sqrt_157 = None
        mul_471: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_157, 1);  reciprocal_157 = None
        unsqueeze_1258: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_471, -1);  mul_471 = None
        unsqueeze_1259: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, -1);  unsqueeze_1258 = None
        mul_472: "f32[8, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_1259);  sub_157 = unsqueeze_1259 = None
        unsqueeze_1260: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_1261: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, -1);  unsqueeze_1260 = None
        mul_473: "f32[8, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_472, unsqueeze_1261);  mul_472 = unsqueeze_1261 = None
        unsqueeze_1262: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_1263: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, -1);  unsqueeze_1262 = None
        add_364: "f32[8, 1280, 14, 14]" = torch.ops.aten.add.Tensor(mul_473, unsqueeze_1263);  mul_473 = unsqueeze_1263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_157: "f32[8, 1280, 14, 14]" = torch.ops.aten.relu.default(add_364);  add_364 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_157: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_157, arg235_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_157 = arg235_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1264: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_1265: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, -1);  unsqueeze_1264 = None
        sub_158: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_1265);  convolution_157 = unsqueeze_1265 = None
        add_365: "f32[800]" = torch.ops.aten.add.Tensor(arg237_1, 0.001);  arg237_1 = None
        sqrt_158: "f32[800]" = torch.ops.aten.sqrt.default(add_365);  add_365 = None
        reciprocal_158: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_158);  sqrt_158 = None
        mul_474: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_158, 1);  reciprocal_158 = None
        unsqueeze_1266: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_474, -1);  mul_474 = None
        unsqueeze_1267: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, -1);  unsqueeze_1266 = None
        mul_475: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_1267);  sub_158 = unsqueeze_1267 = None
        unsqueeze_1268: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_1269: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, -1);  unsqueeze_1268 = None
        mul_476: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_475, unsqueeze_1269);  mul_475 = unsqueeze_1269 = None
        unsqueeze_1270: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_1271: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, -1);  unsqueeze_1270 = None
        add_366: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_476, unsqueeze_1271);  mul_476 = unsqueeze_1271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_158: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_366);  add_366 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_158: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_158, arg240_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_158 = arg240_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1272: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg241_1, -1);  arg241_1 = None
        unsqueeze_1273: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, -1);  unsqueeze_1272 = None
        sub_159: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_158, unsqueeze_1273);  convolution_158 = unsqueeze_1273 = None
        add_367: "f32[800]" = torch.ops.aten.add.Tensor(arg242_1, 0.001);  arg242_1 = None
        sqrt_159: "f32[800]" = torch.ops.aten.sqrt.default(add_367);  add_367 = None
        reciprocal_159: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_159);  sqrt_159 = None
        mul_477: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_159, 1);  reciprocal_159 = None
        unsqueeze_1274: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_477, -1);  mul_477 = None
        unsqueeze_1275: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, -1);  unsqueeze_1274 = None
        mul_478: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_1275);  sub_159 = unsqueeze_1275 = None
        unsqueeze_1276: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
        unsqueeze_1277: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, -1);  unsqueeze_1276 = None
        mul_479: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_478, unsqueeze_1277);  mul_478 = unsqueeze_1277 = None
        unsqueeze_1278: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_1279: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, -1);  unsqueeze_1278 = None
        add_368: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_479, unsqueeze_1279);  mul_479 = unsqueeze_1279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_159: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_368);  add_368 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_159: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_159, arg245_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_159 = arg245_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_450: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_159, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_369: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_362, slice_450);  add_362 = slice_450 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_454: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_159, 1, 1024, 9223372036854775807);  convolution_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_98: "f32[8, 320, 14, 14]" = torch.ops.aten.cat.default([cat_96, slice_454], 1);  cat_96 = slice_454 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_99: "f32[8, 1344, 14, 14]" = torch.ops.aten.cat.default([add_369, cat_98], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1280: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(arg246_1, -1);  arg246_1 = None
        unsqueeze_1281: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, -1);  unsqueeze_1280 = None
        sub_160: "f32[8, 1344, 14, 14]" = torch.ops.aten.sub.Tensor(cat_99, unsqueeze_1281);  cat_99 = unsqueeze_1281 = None
        add_370: "f32[1344]" = torch.ops.aten.add.Tensor(arg247_1, 0.001);  arg247_1 = None
        sqrt_160: "f32[1344]" = torch.ops.aten.sqrt.default(add_370);  add_370 = None
        reciprocal_160: "f32[1344]" = torch.ops.aten.reciprocal.default(sqrt_160);  sqrt_160 = None
        mul_480: "f32[1344]" = torch.ops.aten.mul.Tensor(reciprocal_160, 1);  reciprocal_160 = None
        unsqueeze_1282: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(mul_480, -1);  mul_480 = None
        unsqueeze_1283: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, -1);  unsqueeze_1282 = None
        mul_481: "f32[8, 1344, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_1283);  sub_160 = unsqueeze_1283 = None
        unsqueeze_1284: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_1285: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, -1);  unsqueeze_1284 = None
        mul_482: "f32[8, 1344, 14, 14]" = torch.ops.aten.mul.Tensor(mul_481, unsqueeze_1285);  mul_481 = unsqueeze_1285 = None
        unsqueeze_1286: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_1287: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, -1);  unsqueeze_1286 = None
        add_371: "f32[8, 1344, 14, 14]" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_1287);  mul_482 = unsqueeze_1287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_160: "f32[8, 1344, 14, 14]" = torch.ops.aten.relu.default(add_371);  add_371 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_160: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_160, arg250_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_160 = arg250_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1288: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_1289: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, -1);  unsqueeze_1288 = None
        sub_161: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_1289);  convolution_160 = unsqueeze_1289 = None
        add_372: "f32[800]" = torch.ops.aten.add.Tensor(arg252_1, 0.001);  arg252_1 = None
        sqrt_161: "f32[800]" = torch.ops.aten.sqrt.default(add_372);  add_372 = None
        reciprocal_161: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_161);  sqrt_161 = None
        mul_483: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_161, 1);  reciprocal_161 = None
        unsqueeze_1290: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_483, -1);  mul_483 = None
        unsqueeze_1291: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, -1);  unsqueeze_1290 = None
        mul_484: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_1291);  sub_161 = unsqueeze_1291 = None
        unsqueeze_1292: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg253_1, -1);  arg253_1 = None
        unsqueeze_1293: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, -1);  unsqueeze_1292 = None
        mul_485: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_484, unsqueeze_1293);  mul_484 = unsqueeze_1293 = None
        unsqueeze_1294: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_1295: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, -1);  unsqueeze_1294 = None
        add_373: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_485, unsqueeze_1295);  mul_485 = unsqueeze_1295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_161: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_373);  add_373 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_161: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_161, arg255_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_161 = arg255_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1296: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_1297: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, -1);  unsqueeze_1296 = None
        sub_162: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_1297);  convolution_161 = unsqueeze_1297 = None
        add_374: "f32[800]" = torch.ops.aten.add.Tensor(arg257_1, 0.001);  arg257_1 = None
        sqrt_162: "f32[800]" = torch.ops.aten.sqrt.default(add_374);  add_374 = None
        reciprocal_162: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_162);  sqrt_162 = None
        mul_486: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_162, 1);  reciprocal_162 = None
        unsqueeze_1298: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_486, -1);  mul_486 = None
        unsqueeze_1299: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, -1);  unsqueeze_1298 = None
        mul_487: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_1299);  sub_162 = unsqueeze_1299 = None
        unsqueeze_1300: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg258_1, -1);  arg258_1 = None
        unsqueeze_1301: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, -1);  unsqueeze_1300 = None
        mul_488: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_487, unsqueeze_1301);  mul_487 = unsqueeze_1301 = None
        unsqueeze_1302: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_1303: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, -1);  unsqueeze_1302 = None
        add_375: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_488, unsqueeze_1303);  mul_488 = unsqueeze_1303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_162: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_375);  add_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_162: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_162, arg260_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_162 = arg260_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_458: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_162, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_376: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_369, slice_458);  add_369 = slice_458 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_462: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_162, 1, 1024, 9223372036854775807);  convolution_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_100: "f32[8, 384, 14, 14]" = torch.ops.aten.cat.default([cat_98, slice_462], 1);  cat_98 = slice_462 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_101: "f32[8, 1408, 14, 14]" = torch.ops.aten.cat.default([add_376, cat_100], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1304: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_1305: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, -1);  unsqueeze_1304 = None
        sub_163: "f32[8, 1408, 14, 14]" = torch.ops.aten.sub.Tensor(cat_101, unsqueeze_1305);  cat_101 = unsqueeze_1305 = None
        add_377: "f32[1408]" = torch.ops.aten.add.Tensor(arg262_1, 0.001);  arg262_1 = None
        sqrt_163: "f32[1408]" = torch.ops.aten.sqrt.default(add_377);  add_377 = None
        reciprocal_163: "f32[1408]" = torch.ops.aten.reciprocal.default(sqrt_163);  sqrt_163 = None
        mul_489: "f32[1408]" = torch.ops.aten.mul.Tensor(reciprocal_163, 1);  reciprocal_163 = None
        unsqueeze_1306: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(mul_489, -1);  mul_489 = None
        unsqueeze_1307: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, -1);  unsqueeze_1306 = None
        mul_490: "f32[8, 1408, 14, 14]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_1307);  sub_163 = unsqueeze_1307 = None
        unsqueeze_1308: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
        unsqueeze_1309: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, -1);  unsqueeze_1308 = None
        mul_491: "f32[8, 1408, 14, 14]" = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_1309);  mul_490 = unsqueeze_1309 = None
        unsqueeze_1310: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_1311: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, -1);  unsqueeze_1310 = None
        add_378: "f32[8, 1408, 14, 14]" = torch.ops.aten.add.Tensor(mul_491, unsqueeze_1311);  mul_491 = unsqueeze_1311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_163: "f32[8, 1408, 14, 14]" = torch.ops.aten.relu.default(add_378);  add_378 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_163: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_163, arg265_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_163 = arg265_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1312: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
        unsqueeze_1313: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, -1);  unsqueeze_1312 = None
        sub_164: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_163, unsqueeze_1313);  convolution_163 = unsqueeze_1313 = None
        add_379: "f32[800]" = torch.ops.aten.add.Tensor(arg267_1, 0.001);  arg267_1 = None
        sqrt_164: "f32[800]" = torch.ops.aten.sqrt.default(add_379);  add_379 = None
        reciprocal_164: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_164);  sqrt_164 = None
        mul_492: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_164, 1);  reciprocal_164 = None
        unsqueeze_1314: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_492, -1);  mul_492 = None
        unsqueeze_1315: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, -1);  unsqueeze_1314 = None
        mul_493: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_1315);  sub_164 = unsqueeze_1315 = None
        unsqueeze_1316: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
        unsqueeze_1317: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, -1);  unsqueeze_1316 = None
        mul_494: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_493, unsqueeze_1317);  mul_493 = unsqueeze_1317 = None
        unsqueeze_1318: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_1319: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, -1);  unsqueeze_1318 = None
        add_380: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_494, unsqueeze_1319);  mul_494 = unsqueeze_1319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_164: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_380);  add_380 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_164: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_164, arg270_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_164 = arg270_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1320: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_1321: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, -1);  unsqueeze_1320 = None
        sub_165: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_164, unsqueeze_1321);  convolution_164 = unsqueeze_1321 = None
        add_381: "f32[800]" = torch.ops.aten.add.Tensor(arg272_1, 0.001);  arg272_1 = None
        sqrt_165: "f32[800]" = torch.ops.aten.sqrt.default(add_381);  add_381 = None
        reciprocal_165: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_165);  sqrt_165 = None
        mul_495: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_165, 1);  reciprocal_165 = None
        unsqueeze_1322: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_495, -1);  mul_495 = None
        unsqueeze_1323: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, -1);  unsqueeze_1322 = None
        mul_496: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_1323);  sub_165 = unsqueeze_1323 = None
        unsqueeze_1324: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg273_1, -1);  arg273_1 = None
        unsqueeze_1325: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, -1);  unsqueeze_1324 = None
        mul_497: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_496, unsqueeze_1325);  mul_496 = unsqueeze_1325 = None
        unsqueeze_1326: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_1327: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, -1);  unsqueeze_1326 = None
        add_382: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_497, unsqueeze_1327);  mul_497 = unsqueeze_1327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_165: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_382);  add_382 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_165: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_165, arg275_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_165 = arg275_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_466: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_165, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_383: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_376, slice_466);  add_376 = slice_466 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_470: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_165, 1, 1024, 9223372036854775807);  convolution_165 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_102: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([cat_100, slice_470], 1);  cat_100 = slice_470 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_103: "f32[8, 1472, 14, 14]" = torch.ops.aten.cat.default([add_383, cat_102], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1328: "f32[1472, 1]" = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_1329: "f32[1472, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, -1);  unsqueeze_1328 = None
        sub_166: "f32[8, 1472, 14, 14]" = torch.ops.aten.sub.Tensor(cat_103, unsqueeze_1329);  cat_103 = unsqueeze_1329 = None
        add_384: "f32[1472]" = torch.ops.aten.add.Tensor(arg277_1, 0.001);  arg277_1 = None
        sqrt_166: "f32[1472]" = torch.ops.aten.sqrt.default(add_384);  add_384 = None
        reciprocal_166: "f32[1472]" = torch.ops.aten.reciprocal.default(sqrt_166);  sqrt_166 = None
        mul_498: "f32[1472]" = torch.ops.aten.mul.Tensor(reciprocal_166, 1);  reciprocal_166 = None
        unsqueeze_1330: "f32[1472, 1]" = torch.ops.aten.unsqueeze.default(mul_498, -1);  mul_498 = None
        unsqueeze_1331: "f32[1472, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, -1);  unsqueeze_1330 = None
        mul_499: "f32[8, 1472, 14, 14]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_1331);  sub_166 = unsqueeze_1331 = None
        unsqueeze_1332: "f32[1472, 1]" = torch.ops.aten.unsqueeze.default(arg278_1, -1);  arg278_1 = None
        unsqueeze_1333: "f32[1472, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, -1);  unsqueeze_1332 = None
        mul_500: "f32[8, 1472, 14, 14]" = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_1333);  mul_499 = unsqueeze_1333 = None
        unsqueeze_1334: "f32[1472, 1]" = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_1335: "f32[1472, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, -1);  unsqueeze_1334 = None
        add_385: "f32[8, 1472, 14, 14]" = torch.ops.aten.add.Tensor(mul_500, unsqueeze_1335);  mul_500 = unsqueeze_1335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_166: "f32[8, 1472, 14, 14]" = torch.ops.aten.relu.default(add_385);  add_385 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_166: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_166, arg280_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_166 = arg280_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1336: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg281_1, -1);  arg281_1 = None
        unsqueeze_1337: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, -1);  unsqueeze_1336 = None
        sub_167: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_1337);  convolution_166 = unsqueeze_1337 = None
        add_386: "f32[800]" = torch.ops.aten.add.Tensor(arg282_1, 0.001);  arg282_1 = None
        sqrt_167: "f32[800]" = torch.ops.aten.sqrt.default(add_386);  add_386 = None
        reciprocal_167: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_167);  sqrt_167 = None
        mul_501: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_167, 1);  reciprocal_167 = None
        unsqueeze_1338: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_501, -1);  mul_501 = None
        unsqueeze_1339: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, -1);  unsqueeze_1338 = None
        mul_502: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_1339);  sub_167 = unsqueeze_1339 = None
        unsqueeze_1340: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg283_1, -1);  arg283_1 = None
        unsqueeze_1341: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, -1);  unsqueeze_1340 = None
        mul_503: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_502, unsqueeze_1341);  mul_502 = unsqueeze_1341 = None
        unsqueeze_1342: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_1343: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, -1);  unsqueeze_1342 = None
        add_387: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_1343);  mul_503 = unsqueeze_1343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_167: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_387);  add_387 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_167: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_167, arg285_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_167 = arg285_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1344: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg286_1, -1);  arg286_1 = None
        unsqueeze_1345: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, -1);  unsqueeze_1344 = None
        sub_168: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_1345);  convolution_167 = unsqueeze_1345 = None
        add_388: "f32[800]" = torch.ops.aten.add.Tensor(arg287_1, 0.001);  arg287_1 = None
        sqrt_168: "f32[800]" = torch.ops.aten.sqrt.default(add_388);  add_388 = None
        reciprocal_168: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_168);  sqrt_168 = None
        mul_504: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_168, 1);  reciprocal_168 = None
        unsqueeze_1346: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_504, -1);  mul_504 = None
        unsqueeze_1347: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, -1);  unsqueeze_1346 = None
        mul_505: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_1347);  sub_168 = unsqueeze_1347 = None
        unsqueeze_1348: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_1349: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, -1);  unsqueeze_1348 = None
        mul_506: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_505, unsqueeze_1349);  mul_505 = unsqueeze_1349 = None
        unsqueeze_1350: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_1351: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, -1);  unsqueeze_1350 = None
        add_389: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_506, unsqueeze_1351);  mul_506 = unsqueeze_1351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_168: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_389);  add_389 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_168: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_168, arg290_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_168 = arg290_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_474: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_168, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_390: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_383, slice_474);  add_383 = slice_474 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_478: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_168, 1, 1024, 9223372036854775807);  convolution_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_104: "f32[8, 512, 14, 14]" = torch.ops.aten.cat.default([cat_102, slice_478], 1);  cat_102 = slice_478 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_105: "f32[8, 1536, 14, 14]" = torch.ops.aten.cat.default([add_390, cat_104], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1352: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg291_1, -1);  arg291_1 = None
        unsqueeze_1353: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, -1);  unsqueeze_1352 = None
        sub_169: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(cat_105, unsqueeze_1353);  cat_105 = unsqueeze_1353 = None
        add_391: "f32[1536]" = torch.ops.aten.add.Tensor(arg292_1, 0.001);  arg292_1 = None
        sqrt_169: "f32[1536]" = torch.ops.aten.sqrt.default(add_391);  add_391 = None
        reciprocal_169: "f32[1536]" = torch.ops.aten.reciprocal.default(sqrt_169);  sqrt_169 = None
        mul_507: "f32[1536]" = torch.ops.aten.mul.Tensor(reciprocal_169, 1);  reciprocal_169 = None
        unsqueeze_1354: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(mul_507, -1);  mul_507 = None
        unsqueeze_1355: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, -1);  unsqueeze_1354 = None
        mul_508: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_1355);  sub_169 = unsqueeze_1355 = None
        unsqueeze_1356: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg293_1, -1);  arg293_1 = None
        unsqueeze_1357: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, -1);  unsqueeze_1356 = None
        mul_509: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_508, unsqueeze_1357);  mul_508 = unsqueeze_1357 = None
        unsqueeze_1358: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_1359: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, -1);  unsqueeze_1358 = None
        add_392: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_509, unsqueeze_1359);  mul_509 = unsqueeze_1359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_169: "f32[8, 1536, 14, 14]" = torch.ops.aten.relu.default(add_392);  add_392 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_169: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_169, arg295_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_169 = arg295_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1360: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg296_1, -1);  arg296_1 = None
        unsqueeze_1361: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, -1);  unsqueeze_1360 = None
        sub_170: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_169, unsqueeze_1361);  convolution_169 = unsqueeze_1361 = None
        add_393: "f32[800]" = torch.ops.aten.add.Tensor(arg297_1, 0.001);  arg297_1 = None
        sqrt_170: "f32[800]" = torch.ops.aten.sqrt.default(add_393);  add_393 = None
        reciprocal_170: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_170);  sqrt_170 = None
        mul_510: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_170, 1);  reciprocal_170 = None
        unsqueeze_1362: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_510, -1);  mul_510 = None
        unsqueeze_1363: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, -1);  unsqueeze_1362 = None
        mul_511: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_1363);  sub_170 = unsqueeze_1363 = None
        unsqueeze_1364: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
        unsqueeze_1365: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, -1);  unsqueeze_1364 = None
        mul_512: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_511, unsqueeze_1365);  mul_511 = unsqueeze_1365 = None
        unsqueeze_1366: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_1367: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, -1);  unsqueeze_1366 = None
        add_394: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_512, unsqueeze_1367);  mul_512 = unsqueeze_1367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_170: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_394);  add_394 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_170: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_170, arg300_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_170 = arg300_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1368: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg301_1, -1);  arg301_1 = None
        unsqueeze_1369: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, -1);  unsqueeze_1368 = None
        sub_171: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_170, unsqueeze_1369);  convolution_170 = unsqueeze_1369 = None
        add_395: "f32[800]" = torch.ops.aten.add.Tensor(arg302_1, 0.001);  arg302_1 = None
        sqrt_171: "f32[800]" = torch.ops.aten.sqrt.default(add_395);  add_395 = None
        reciprocal_171: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_171);  sqrt_171 = None
        mul_513: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_171, 1);  reciprocal_171 = None
        unsqueeze_1370: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_513, -1);  mul_513 = None
        unsqueeze_1371: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, -1);  unsqueeze_1370 = None
        mul_514: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_1371);  sub_171 = unsqueeze_1371 = None
        unsqueeze_1372: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_1373: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, -1);  unsqueeze_1372 = None
        mul_515: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_514, unsqueeze_1373);  mul_514 = unsqueeze_1373 = None
        unsqueeze_1374: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_1375: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, -1);  unsqueeze_1374 = None
        add_396: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_515, unsqueeze_1375);  mul_515 = unsqueeze_1375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_171: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_396);  add_396 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_171: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_171, arg305_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_171 = arg305_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_482: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_171, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_397: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_390, slice_482);  add_390 = slice_482 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_486: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_171, 1, 1024, 9223372036854775807);  convolution_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_106: "f32[8, 576, 14, 14]" = torch.ops.aten.cat.default([cat_104, slice_486], 1);  cat_104 = slice_486 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_107: "f32[8, 1600, 14, 14]" = torch.ops.aten.cat.default([add_397, cat_106], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1376: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg306_1, -1);  arg306_1 = None
        unsqueeze_1377: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, -1);  unsqueeze_1376 = None
        sub_172: "f32[8, 1600, 14, 14]" = torch.ops.aten.sub.Tensor(cat_107, unsqueeze_1377);  cat_107 = unsqueeze_1377 = None
        add_398: "f32[1600]" = torch.ops.aten.add.Tensor(arg307_1, 0.001);  arg307_1 = None
        sqrt_172: "f32[1600]" = torch.ops.aten.sqrt.default(add_398);  add_398 = None
        reciprocal_172: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_172);  sqrt_172 = None
        mul_516: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_172, 1);  reciprocal_172 = None
        unsqueeze_1378: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_516, -1);  mul_516 = None
        unsqueeze_1379: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, -1);  unsqueeze_1378 = None
        mul_517: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_1379);  sub_172 = unsqueeze_1379 = None
        unsqueeze_1380: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_1381: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, -1);  unsqueeze_1380 = None
        mul_518: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(mul_517, unsqueeze_1381);  mul_517 = unsqueeze_1381 = None
        unsqueeze_1382: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_1383: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, -1);  unsqueeze_1382 = None
        add_399: "f32[8, 1600, 14, 14]" = torch.ops.aten.add.Tensor(mul_518, unsqueeze_1383);  mul_518 = unsqueeze_1383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_172: "f32[8, 1600, 14, 14]" = torch.ops.aten.relu.default(add_399);  add_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_172: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_172, arg310_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_172 = arg310_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1384: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_1385: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, -1);  unsqueeze_1384 = None
        sub_173: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_172, unsqueeze_1385);  convolution_172 = unsqueeze_1385 = None
        add_400: "f32[800]" = torch.ops.aten.add.Tensor(arg312_1, 0.001);  arg312_1 = None
        sqrt_173: "f32[800]" = torch.ops.aten.sqrt.default(add_400);  add_400 = None
        reciprocal_173: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_173);  sqrt_173 = None
        mul_519: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_173, 1);  reciprocal_173 = None
        unsqueeze_1386: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_519, -1);  mul_519 = None
        unsqueeze_1387: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, -1);  unsqueeze_1386 = None
        mul_520: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_1387);  sub_173 = unsqueeze_1387 = None
        unsqueeze_1388: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
        unsqueeze_1389: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, -1);  unsqueeze_1388 = None
        mul_521: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_520, unsqueeze_1389);  mul_520 = unsqueeze_1389 = None
        unsqueeze_1390: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_1391: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, -1);  unsqueeze_1390 = None
        add_401: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_521, unsqueeze_1391);  mul_521 = unsqueeze_1391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_173: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_401);  add_401 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_173: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_173, arg315_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_173 = arg315_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1392: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg316_1, -1);  arg316_1 = None
        unsqueeze_1393: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1392, -1);  unsqueeze_1392 = None
        sub_174: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_173, unsqueeze_1393);  convolution_173 = unsqueeze_1393 = None
        add_402: "f32[800]" = torch.ops.aten.add.Tensor(arg317_1, 0.001);  arg317_1 = None
        sqrt_174: "f32[800]" = torch.ops.aten.sqrt.default(add_402);  add_402 = None
        reciprocal_174: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_174);  sqrt_174 = None
        mul_522: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_174, 1);  reciprocal_174 = None
        unsqueeze_1394: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_522, -1);  mul_522 = None
        unsqueeze_1395: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, -1);  unsqueeze_1394 = None
        mul_523: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_1395);  sub_174 = unsqueeze_1395 = None
        unsqueeze_1396: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg318_1, -1);  arg318_1 = None
        unsqueeze_1397: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1396, -1);  unsqueeze_1396 = None
        mul_524: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_523, unsqueeze_1397);  mul_523 = unsqueeze_1397 = None
        unsqueeze_1398: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_1399: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, -1);  unsqueeze_1398 = None
        add_403: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_524, unsqueeze_1399);  mul_524 = unsqueeze_1399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_174: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_403);  add_403 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_174: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_174, arg320_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_174 = arg320_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_490: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_174, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_404: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_397, slice_490);  add_397 = slice_490 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_494: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_174, 1, 1024, 9223372036854775807);  convolution_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_108: "f32[8, 640, 14, 14]" = torch.ops.aten.cat.default([cat_106, slice_494], 1);  cat_106 = slice_494 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_109: "f32[8, 1664, 14, 14]" = torch.ops.aten.cat.default([add_404, cat_108], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1400: "f32[1664, 1]" = torch.ops.aten.unsqueeze.default(arg321_1, -1);  arg321_1 = None
        unsqueeze_1401: "f32[1664, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, -1);  unsqueeze_1400 = None
        sub_175: "f32[8, 1664, 14, 14]" = torch.ops.aten.sub.Tensor(cat_109, unsqueeze_1401);  cat_109 = unsqueeze_1401 = None
        add_405: "f32[1664]" = torch.ops.aten.add.Tensor(arg322_1, 0.001);  arg322_1 = None
        sqrt_175: "f32[1664]" = torch.ops.aten.sqrt.default(add_405);  add_405 = None
        reciprocal_175: "f32[1664]" = torch.ops.aten.reciprocal.default(sqrt_175);  sqrt_175 = None
        mul_525: "f32[1664]" = torch.ops.aten.mul.Tensor(reciprocal_175, 1);  reciprocal_175 = None
        unsqueeze_1402: "f32[1664, 1]" = torch.ops.aten.unsqueeze.default(mul_525, -1);  mul_525 = None
        unsqueeze_1403: "f32[1664, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1402, -1);  unsqueeze_1402 = None
        mul_526: "f32[8, 1664, 14, 14]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_1403);  sub_175 = unsqueeze_1403 = None
        unsqueeze_1404: "f32[1664, 1]" = torch.ops.aten.unsqueeze.default(arg323_1, -1);  arg323_1 = None
        unsqueeze_1405: "f32[1664, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1404, -1);  unsqueeze_1404 = None
        mul_527: "f32[8, 1664, 14, 14]" = torch.ops.aten.mul.Tensor(mul_526, unsqueeze_1405);  mul_526 = unsqueeze_1405 = None
        unsqueeze_1406: "f32[1664, 1]" = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_1407: "f32[1664, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, -1);  unsqueeze_1406 = None
        add_406: "f32[8, 1664, 14, 14]" = torch.ops.aten.add.Tensor(mul_527, unsqueeze_1407);  mul_527 = unsqueeze_1407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_175: "f32[8, 1664, 14, 14]" = torch.ops.aten.relu.default(add_406);  add_406 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_175: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_175, arg325_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_175 = arg325_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1408: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg326_1, -1);  arg326_1 = None
        unsqueeze_1409: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1408, -1);  unsqueeze_1408 = None
        sub_176: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_175, unsqueeze_1409);  convolution_175 = unsqueeze_1409 = None
        add_407: "f32[800]" = torch.ops.aten.add.Tensor(arg327_1, 0.001);  arg327_1 = None
        sqrt_176: "f32[800]" = torch.ops.aten.sqrt.default(add_407);  add_407 = None
        reciprocal_176: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_176);  sqrt_176 = None
        mul_528: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_176, 1);  reciprocal_176 = None
        unsqueeze_1410: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_528, -1);  mul_528 = None
        unsqueeze_1411: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, -1);  unsqueeze_1410 = None
        mul_529: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_1411);  sub_176 = unsqueeze_1411 = None
        unsqueeze_1412: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
        unsqueeze_1413: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, -1);  unsqueeze_1412 = None
        mul_530: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_529, unsqueeze_1413);  mul_529 = unsqueeze_1413 = None
        unsqueeze_1414: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_1415: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1414, -1);  unsqueeze_1414 = None
        add_408: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_530, unsqueeze_1415);  mul_530 = unsqueeze_1415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_176: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_408);  add_408 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_176: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_176, arg330_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_176 = arg330_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1416: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg331_1, -1);  arg331_1 = None
        unsqueeze_1417: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1416, -1);  unsqueeze_1416 = None
        sub_177: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_176, unsqueeze_1417);  convolution_176 = unsqueeze_1417 = None
        add_409: "f32[800]" = torch.ops.aten.add.Tensor(arg332_1, 0.001);  arg332_1 = None
        sqrt_177: "f32[800]" = torch.ops.aten.sqrt.default(add_409);  add_409 = None
        reciprocal_177: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_177);  sqrt_177 = None
        mul_531: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_177, 1);  reciprocal_177 = None
        unsqueeze_1418: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_531, -1);  mul_531 = None
        unsqueeze_1419: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, -1);  unsqueeze_1418 = None
        mul_532: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_1419);  sub_177 = unsqueeze_1419 = None
        unsqueeze_1420: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg333_1, -1);  arg333_1 = None
        unsqueeze_1421: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1420, -1);  unsqueeze_1420 = None
        mul_533: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_1421);  mul_532 = unsqueeze_1421 = None
        unsqueeze_1422: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_1423: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, -1);  unsqueeze_1422 = None
        add_410: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_533, unsqueeze_1423);  mul_533 = unsqueeze_1423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_177: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_410);  add_410 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_177: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_177, arg335_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_177 = arg335_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_498: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_177, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_411: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_404, slice_498);  add_404 = slice_498 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_502: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_177, 1, 1024, 9223372036854775807);  convolution_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_110: "f32[8, 704, 14, 14]" = torch.ops.aten.cat.default([cat_108, slice_502], 1);  cat_108 = slice_502 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_111: "f32[8, 1728, 14, 14]" = torch.ops.aten.cat.default([add_411, cat_110], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1424: "f32[1728, 1]" = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
        unsqueeze_1425: "f32[1728, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, -1);  unsqueeze_1424 = None
        sub_178: "f32[8, 1728, 14, 14]" = torch.ops.aten.sub.Tensor(cat_111, unsqueeze_1425);  cat_111 = unsqueeze_1425 = None
        add_412: "f32[1728]" = torch.ops.aten.add.Tensor(arg337_1, 0.001);  arg337_1 = None
        sqrt_178: "f32[1728]" = torch.ops.aten.sqrt.default(add_412);  add_412 = None
        reciprocal_178: "f32[1728]" = torch.ops.aten.reciprocal.default(sqrt_178);  sqrt_178 = None
        mul_534: "f32[1728]" = torch.ops.aten.mul.Tensor(reciprocal_178, 1);  reciprocal_178 = None
        unsqueeze_1426: "f32[1728, 1]" = torch.ops.aten.unsqueeze.default(mul_534, -1);  mul_534 = None
        unsqueeze_1427: "f32[1728, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1426, -1);  unsqueeze_1426 = None
        mul_535: "f32[8, 1728, 14, 14]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_1427);  sub_178 = unsqueeze_1427 = None
        unsqueeze_1428: "f32[1728, 1]" = torch.ops.aten.unsqueeze.default(arg338_1, -1);  arg338_1 = None
        unsqueeze_1429: "f32[1728, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1428, -1);  unsqueeze_1428 = None
        mul_536: "f32[8, 1728, 14, 14]" = torch.ops.aten.mul.Tensor(mul_535, unsqueeze_1429);  mul_535 = unsqueeze_1429 = None
        unsqueeze_1430: "f32[1728, 1]" = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_1431: "f32[1728, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1430, -1);  unsqueeze_1430 = None
        add_413: "f32[8, 1728, 14, 14]" = torch.ops.aten.add.Tensor(mul_536, unsqueeze_1431);  mul_536 = unsqueeze_1431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_178: "f32[8, 1728, 14, 14]" = torch.ops.aten.relu.default(add_413);  add_413 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_178: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_178, arg340_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_178 = arg340_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1432: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg341_1, -1);  arg341_1 = None
        unsqueeze_1433: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, -1);  unsqueeze_1432 = None
        sub_179: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_178, unsqueeze_1433);  convolution_178 = unsqueeze_1433 = None
        add_414: "f32[800]" = torch.ops.aten.add.Tensor(arg342_1, 0.001);  arg342_1 = None
        sqrt_179: "f32[800]" = torch.ops.aten.sqrt.default(add_414);  add_414 = None
        reciprocal_179: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_179);  sqrt_179 = None
        mul_537: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_179, 1);  reciprocal_179 = None
        unsqueeze_1434: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_537, -1);  mul_537 = None
        unsqueeze_1435: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, -1);  unsqueeze_1434 = None
        mul_538: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_1435);  sub_179 = unsqueeze_1435 = None
        unsqueeze_1436: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg343_1, -1);  arg343_1 = None
        unsqueeze_1437: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1436, -1);  unsqueeze_1436 = None
        mul_539: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_538, unsqueeze_1437);  mul_538 = unsqueeze_1437 = None
        unsqueeze_1438: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_1439: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, -1);  unsqueeze_1438 = None
        add_415: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_539, unsqueeze_1439);  mul_539 = unsqueeze_1439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_179: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_415);  add_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_179: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_179, arg345_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_179 = arg345_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1440: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg346_1, -1);  arg346_1 = None
        unsqueeze_1441: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1440, -1);  unsqueeze_1440 = None
        sub_180: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_179, unsqueeze_1441);  convolution_179 = unsqueeze_1441 = None
        add_416: "f32[800]" = torch.ops.aten.add.Tensor(arg347_1, 0.001);  arg347_1 = None
        sqrt_180: "f32[800]" = torch.ops.aten.sqrt.default(add_416);  add_416 = None
        reciprocal_180: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_180);  sqrt_180 = None
        mul_540: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_180, 1);  reciprocal_180 = None
        unsqueeze_1442: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_540, -1);  mul_540 = None
        unsqueeze_1443: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1442, -1);  unsqueeze_1442 = None
        mul_541: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_1443);  sub_180 = unsqueeze_1443 = None
        unsqueeze_1444: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg348_1, -1);  arg348_1 = None
        unsqueeze_1445: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, -1);  unsqueeze_1444 = None
        mul_542: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_541, unsqueeze_1445);  mul_541 = unsqueeze_1445 = None
        unsqueeze_1446: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_1447: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, -1);  unsqueeze_1446 = None
        add_417: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_542, unsqueeze_1447);  mul_542 = unsqueeze_1447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_180: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_417);  add_417 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_180: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_180, arg350_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_180 = arg350_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_506: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_180, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_418: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_411, slice_506);  add_411 = slice_506 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_510: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_180, 1, 1024, 9223372036854775807);  convolution_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_112: "f32[8, 768, 14, 14]" = torch.ops.aten.cat.default([cat_110, slice_510], 1);  cat_110 = slice_510 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_113: "f32[8, 1792, 14, 14]" = torch.ops.aten.cat.default([add_418, cat_112], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1448: "f32[1792, 1]" = torch.ops.aten.unsqueeze.default(arg351_1, -1);  arg351_1 = None
        unsqueeze_1449: "f32[1792, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, -1);  unsqueeze_1448 = None
        sub_181: "f32[8, 1792, 14, 14]" = torch.ops.aten.sub.Tensor(cat_113, unsqueeze_1449);  cat_113 = unsqueeze_1449 = None
        add_419: "f32[1792]" = torch.ops.aten.add.Tensor(arg352_1, 0.001);  arg352_1 = None
        sqrt_181: "f32[1792]" = torch.ops.aten.sqrt.default(add_419);  add_419 = None
        reciprocal_181: "f32[1792]" = torch.ops.aten.reciprocal.default(sqrt_181);  sqrt_181 = None
        mul_543: "f32[1792]" = torch.ops.aten.mul.Tensor(reciprocal_181, 1);  reciprocal_181 = None
        unsqueeze_1450: "f32[1792, 1]" = torch.ops.aten.unsqueeze.default(mul_543, -1);  mul_543 = None
        unsqueeze_1451: "f32[1792, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, -1);  unsqueeze_1450 = None
        mul_544: "f32[8, 1792, 14, 14]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_1451);  sub_181 = unsqueeze_1451 = None
        unsqueeze_1452: "f32[1792, 1]" = torch.ops.aten.unsqueeze.default(arg353_1, -1);  arg353_1 = None
        unsqueeze_1453: "f32[1792, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1452, -1);  unsqueeze_1452 = None
        mul_545: "f32[8, 1792, 14, 14]" = torch.ops.aten.mul.Tensor(mul_544, unsqueeze_1453);  mul_544 = unsqueeze_1453 = None
        unsqueeze_1454: "f32[1792, 1]" = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_1455: "f32[1792, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, -1);  unsqueeze_1454 = None
        add_420: "f32[8, 1792, 14, 14]" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_1455);  mul_545 = unsqueeze_1455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_181: "f32[8, 1792, 14, 14]" = torch.ops.aten.relu.default(add_420);  add_420 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_181: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_181, arg355_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_181 = arg355_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1456: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg356_1, -1);  arg356_1 = None
        unsqueeze_1457: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, -1);  unsqueeze_1456 = None
        sub_182: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_181, unsqueeze_1457);  convolution_181 = unsqueeze_1457 = None
        add_421: "f32[800]" = torch.ops.aten.add.Tensor(arg357_1, 0.001);  arg357_1 = None
        sqrt_182: "f32[800]" = torch.ops.aten.sqrt.default(add_421);  add_421 = None
        reciprocal_182: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_182);  sqrt_182 = None
        mul_546: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_182, 1);  reciprocal_182 = None
        unsqueeze_1458: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_546, -1);  mul_546 = None
        unsqueeze_1459: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, -1);  unsqueeze_1458 = None
        mul_547: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_1459);  sub_182 = unsqueeze_1459 = None
        unsqueeze_1460: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg358_1, -1);  arg358_1 = None
        unsqueeze_1461: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, -1);  unsqueeze_1460 = None
        mul_548: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_547, unsqueeze_1461);  mul_547 = unsqueeze_1461 = None
        unsqueeze_1462: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_1463: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, -1);  unsqueeze_1462 = None
        add_422: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_548, unsqueeze_1463);  mul_548 = unsqueeze_1463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_182: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_422);  add_422 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_182: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_182, arg360_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_182 = arg360_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1464: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg361_1, -1);  arg361_1 = None
        unsqueeze_1465: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1464, -1);  unsqueeze_1464 = None
        sub_183: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_182, unsqueeze_1465);  convolution_182 = unsqueeze_1465 = None
        add_423: "f32[800]" = torch.ops.aten.add.Tensor(arg362_1, 0.001);  arg362_1 = None
        sqrt_183: "f32[800]" = torch.ops.aten.sqrt.default(add_423);  add_423 = None
        reciprocal_183: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_183);  sqrt_183 = None
        mul_549: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_183, 1);  reciprocal_183 = None
        unsqueeze_1466: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_549, -1);  mul_549 = None
        unsqueeze_1467: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, -1);  unsqueeze_1466 = None
        mul_550: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_1467);  sub_183 = unsqueeze_1467 = None
        unsqueeze_1468: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg363_1, -1);  arg363_1 = None
        unsqueeze_1469: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, -1);  unsqueeze_1468 = None
        mul_551: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_550, unsqueeze_1469);  mul_550 = unsqueeze_1469 = None
        unsqueeze_1470: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
        unsqueeze_1471: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, -1);  unsqueeze_1470 = None
        add_424: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_551, unsqueeze_1471);  mul_551 = unsqueeze_1471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_183: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_424);  add_424 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_183: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_183, arg365_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_183 = arg365_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_514: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_183, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_425: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_418, slice_514);  add_418 = slice_514 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_518: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_183, 1, 1024, 9223372036854775807);  convolution_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_114: "f32[8, 832, 14, 14]" = torch.ops.aten.cat.default([cat_112, slice_518], 1);  cat_112 = slice_518 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_115: "f32[8, 1856, 14, 14]" = torch.ops.aten.cat.default([add_425, cat_114], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1472: "f32[1856, 1]" = torch.ops.aten.unsqueeze.default(arg366_1, -1);  arg366_1 = None
        unsqueeze_1473: "f32[1856, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, -1);  unsqueeze_1472 = None
        sub_184: "f32[8, 1856, 14, 14]" = torch.ops.aten.sub.Tensor(cat_115, unsqueeze_1473);  cat_115 = unsqueeze_1473 = None
        add_426: "f32[1856]" = torch.ops.aten.add.Tensor(arg367_1, 0.001);  arg367_1 = None
        sqrt_184: "f32[1856]" = torch.ops.aten.sqrt.default(add_426);  add_426 = None
        reciprocal_184: "f32[1856]" = torch.ops.aten.reciprocal.default(sqrt_184);  sqrt_184 = None
        mul_552: "f32[1856]" = torch.ops.aten.mul.Tensor(reciprocal_184, 1);  reciprocal_184 = None
        unsqueeze_1474: "f32[1856, 1]" = torch.ops.aten.unsqueeze.default(mul_552, -1);  mul_552 = None
        unsqueeze_1475: "f32[1856, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, -1);  unsqueeze_1474 = None
        mul_553: "f32[8, 1856, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_1475);  sub_184 = unsqueeze_1475 = None
        unsqueeze_1476: "f32[1856, 1]" = torch.ops.aten.unsqueeze.default(arg368_1, -1);  arg368_1 = None
        unsqueeze_1477: "f32[1856, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1476, -1);  unsqueeze_1476 = None
        mul_554: "f32[8, 1856, 14, 14]" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_1477);  mul_553 = unsqueeze_1477 = None
        unsqueeze_1478: "f32[1856, 1]" = torch.ops.aten.unsqueeze.default(arg369_1, -1);  arg369_1 = None
        unsqueeze_1479: "f32[1856, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, -1);  unsqueeze_1478 = None
        add_427: "f32[8, 1856, 14, 14]" = torch.ops.aten.add.Tensor(mul_554, unsqueeze_1479);  mul_554 = unsqueeze_1479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_184: "f32[8, 1856, 14, 14]" = torch.ops.aten.relu.default(add_427);  add_427 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_184: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_184, arg370_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_184 = arg370_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1480: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg371_1, -1);  arg371_1 = None
        unsqueeze_1481: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1480, -1);  unsqueeze_1480 = None
        sub_185: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_184, unsqueeze_1481);  convolution_184 = unsqueeze_1481 = None
        add_428: "f32[800]" = torch.ops.aten.add.Tensor(arg372_1, 0.001);  arg372_1 = None
        sqrt_185: "f32[800]" = torch.ops.aten.sqrt.default(add_428);  add_428 = None
        reciprocal_185: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_185);  sqrt_185 = None
        mul_555: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_185, 1);  reciprocal_185 = None
        unsqueeze_1482: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_555, -1);  mul_555 = None
        unsqueeze_1483: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, -1);  unsqueeze_1482 = None
        mul_556: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_1483);  sub_185 = unsqueeze_1483 = None
        unsqueeze_1484: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg373_1, -1);  arg373_1 = None
        unsqueeze_1485: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, -1);  unsqueeze_1484 = None
        mul_557: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_556, unsqueeze_1485);  mul_556 = unsqueeze_1485 = None
        unsqueeze_1486: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
        unsqueeze_1487: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1486, -1);  unsqueeze_1486 = None
        add_429: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_557, unsqueeze_1487);  mul_557 = unsqueeze_1487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_185: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_429);  add_429 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_185: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_185, arg375_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_185 = arg375_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1488: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg376_1, -1);  arg376_1 = None
        unsqueeze_1489: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1488, -1);  unsqueeze_1488 = None
        sub_186: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_185, unsqueeze_1489);  convolution_185 = unsqueeze_1489 = None
        add_430: "f32[800]" = torch.ops.aten.add.Tensor(arg377_1, 0.001);  arg377_1 = None
        sqrt_186: "f32[800]" = torch.ops.aten.sqrt.default(add_430);  add_430 = None
        reciprocal_186: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_186);  sqrt_186 = None
        mul_558: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_186, 1);  reciprocal_186 = None
        unsqueeze_1490: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_558, -1);  mul_558 = None
        unsqueeze_1491: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, -1);  unsqueeze_1490 = None
        mul_559: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_1491);  sub_186 = unsqueeze_1491 = None
        unsqueeze_1492: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg378_1, -1);  arg378_1 = None
        unsqueeze_1493: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1492, -1);  unsqueeze_1492 = None
        mul_560: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_559, unsqueeze_1493);  mul_559 = unsqueeze_1493 = None
        unsqueeze_1494: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_1495: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, -1);  unsqueeze_1494 = None
        add_431: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_560, unsqueeze_1495);  mul_560 = unsqueeze_1495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_186: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_431);  add_431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_186: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_186, arg380_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_186 = arg380_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_522: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_186, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_432: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_425, slice_522);  add_425 = slice_522 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_526: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_186, 1, 1024, 9223372036854775807);  convolution_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_116: "f32[8, 896, 14, 14]" = torch.ops.aten.cat.default([cat_114, slice_526], 1);  cat_114 = slice_526 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_117: "f32[8, 1920, 14, 14]" = torch.ops.aten.cat.default([add_432, cat_116], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1496: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(arg381_1, -1);  arg381_1 = None
        unsqueeze_1497: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1496, -1);  unsqueeze_1496 = None
        sub_187: "f32[8, 1920, 14, 14]" = torch.ops.aten.sub.Tensor(cat_117, unsqueeze_1497);  cat_117 = unsqueeze_1497 = None
        add_433: "f32[1920]" = torch.ops.aten.add.Tensor(arg382_1, 0.001);  arg382_1 = None
        sqrt_187: "f32[1920]" = torch.ops.aten.sqrt.default(add_433);  add_433 = None
        reciprocal_187: "f32[1920]" = torch.ops.aten.reciprocal.default(sqrt_187);  sqrt_187 = None
        mul_561: "f32[1920]" = torch.ops.aten.mul.Tensor(reciprocal_187, 1);  reciprocal_187 = None
        unsqueeze_1498: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(mul_561, -1);  mul_561 = None
        unsqueeze_1499: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, -1);  unsqueeze_1498 = None
        mul_562: "f32[8, 1920, 14, 14]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_1499);  sub_187 = unsqueeze_1499 = None
        unsqueeze_1500: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(arg383_1, -1);  arg383_1 = None
        unsqueeze_1501: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1500, -1);  unsqueeze_1500 = None
        mul_563: "f32[8, 1920, 14, 14]" = torch.ops.aten.mul.Tensor(mul_562, unsqueeze_1501);  mul_562 = unsqueeze_1501 = None
        unsqueeze_1502: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(arg384_1, -1);  arg384_1 = None
        unsqueeze_1503: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1502, -1);  unsqueeze_1502 = None
        add_434: "f32[8, 1920, 14, 14]" = torch.ops.aten.add.Tensor(mul_563, unsqueeze_1503);  mul_563 = unsqueeze_1503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_187: "f32[8, 1920, 14, 14]" = torch.ops.aten.relu.default(add_434);  add_434 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_187: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_187, arg385_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_187 = arg385_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1504: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg386_1, -1);  arg386_1 = None
        unsqueeze_1505: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1504, -1);  unsqueeze_1504 = None
        sub_188: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_187, unsqueeze_1505);  convolution_187 = unsqueeze_1505 = None
        add_435: "f32[800]" = torch.ops.aten.add.Tensor(arg387_1, 0.001);  arg387_1 = None
        sqrt_188: "f32[800]" = torch.ops.aten.sqrt.default(add_435);  add_435 = None
        reciprocal_188: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_188);  sqrt_188 = None
        mul_564: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_188, 1);  reciprocal_188 = None
        unsqueeze_1506: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_564, -1);  mul_564 = None
        unsqueeze_1507: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, -1);  unsqueeze_1506 = None
        mul_565: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_1507);  sub_188 = unsqueeze_1507 = None
        unsqueeze_1508: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
        unsqueeze_1509: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1508, -1);  unsqueeze_1508 = None
        mul_566: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_565, unsqueeze_1509);  mul_565 = unsqueeze_1509 = None
        unsqueeze_1510: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg389_1, -1);  arg389_1 = None
        unsqueeze_1511: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1510, -1);  unsqueeze_1510 = None
        add_436: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_1511);  mul_566 = unsqueeze_1511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_188: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_436);  add_436 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_188: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_188, arg390_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_188 = arg390_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1512: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg391_1, -1);  arg391_1 = None
        unsqueeze_1513: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1512, -1);  unsqueeze_1512 = None
        sub_189: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_188, unsqueeze_1513);  convolution_188 = unsqueeze_1513 = None
        add_437: "f32[800]" = torch.ops.aten.add.Tensor(arg392_1, 0.001);  arg392_1 = None
        sqrt_189: "f32[800]" = torch.ops.aten.sqrt.default(add_437);  add_437 = None
        reciprocal_189: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_189);  sqrt_189 = None
        mul_567: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_189, 1);  reciprocal_189 = None
        unsqueeze_1514: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_567, -1);  mul_567 = None
        unsqueeze_1515: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1514, -1);  unsqueeze_1514 = None
        mul_568: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_1515);  sub_189 = unsqueeze_1515 = None
        unsqueeze_1516: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg393_1, -1);  arg393_1 = None
        unsqueeze_1517: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1516, -1);  unsqueeze_1516 = None
        mul_569: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_568, unsqueeze_1517);  mul_568 = unsqueeze_1517 = None
        unsqueeze_1518: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
        unsqueeze_1519: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, -1);  unsqueeze_1518 = None
        add_438: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_569, unsqueeze_1519);  mul_569 = unsqueeze_1519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_189: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_438);  add_438 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_189: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_189, arg395_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_189 = arg395_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_530: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_189, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_439: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_432, slice_530);  add_432 = slice_530 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_534: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_189, 1, 1024, 9223372036854775807);  convolution_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_118: "f32[8, 960, 14, 14]" = torch.ops.aten.cat.default([cat_116, slice_534], 1);  cat_116 = slice_534 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_119: "f32[8, 1984, 14, 14]" = torch.ops.aten.cat.default([add_439, cat_118], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1520: "f32[1984, 1]" = torch.ops.aten.unsqueeze.default(arg396_1, -1);  arg396_1 = None
        unsqueeze_1521: "f32[1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1520, -1);  unsqueeze_1520 = None
        sub_190: "f32[8, 1984, 14, 14]" = torch.ops.aten.sub.Tensor(cat_119, unsqueeze_1521);  cat_119 = unsqueeze_1521 = None
        add_440: "f32[1984]" = torch.ops.aten.add.Tensor(arg397_1, 0.001);  arg397_1 = None
        sqrt_190: "f32[1984]" = torch.ops.aten.sqrt.default(add_440);  add_440 = None
        reciprocal_190: "f32[1984]" = torch.ops.aten.reciprocal.default(sqrt_190);  sqrt_190 = None
        mul_570: "f32[1984]" = torch.ops.aten.mul.Tensor(reciprocal_190, 1);  reciprocal_190 = None
        unsqueeze_1522: "f32[1984, 1]" = torch.ops.aten.unsqueeze.default(mul_570, -1);  mul_570 = None
        unsqueeze_1523: "f32[1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1522, -1);  unsqueeze_1522 = None
        mul_571: "f32[8, 1984, 14, 14]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_1523);  sub_190 = unsqueeze_1523 = None
        unsqueeze_1524: "f32[1984, 1]" = torch.ops.aten.unsqueeze.default(arg398_1, -1);  arg398_1 = None
        unsqueeze_1525: "f32[1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1524, -1);  unsqueeze_1524 = None
        mul_572: "f32[8, 1984, 14, 14]" = torch.ops.aten.mul.Tensor(mul_571, unsqueeze_1525);  mul_571 = unsqueeze_1525 = None
        unsqueeze_1526: "f32[1984, 1]" = torch.ops.aten.unsqueeze.default(arg399_1, -1);  arg399_1 = None
        unsqueeze_1527: "f32[1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1526, -1);  unsqueeze_1526 = None
        add_441: "f32[8, 1984, 14, 14]" = torch.ops.aten.add.Tensor(mul_572, unsqueeze_1527);  mul_572 = unsqueeze_1527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_190: "f32[8, 1984, 14, 14]" = torch.ops.aten.relu.default(add_441);  add_441 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_190: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_190, arg400_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_190 = arg400_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1528: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg401_1, -1);  arg401_1 = None
        unsqueeze_1529: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1528, -1);  unsqueeze_1528 = None
        sub_191: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_190, unsqueeze_1529);  convolution_190 = unsqueeze_1529 = None
        add_442: "f32[800]" = torch.ops.aten.add.Tensor(arg402_1, 0.001);  arg402_1 = None
        sqrt_191: "f32[800]" = torch.ops.aten.sqrt.default(add_442);  add_442 = None
        reciprocal_191: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_191);  sqrt_191 = None
        mul_573: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_191, 1);  reciprocal_191 = None
        unsqueeze_1530: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_573, -1);  mul_573 = None
        unsqueeze_1531: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, -1);  unsqueeze_1530 = None
        mul_574: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_1531);  sub_191 = unsqueeze_1531 = None
        unsqueeze_1532: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg403_1, -1);  arg403_1 = None
        unsqueeze_1533: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1532, -1);  unsqueeze_1532 = None
        mul_575: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_1533);  mul_574 = unsqueeze_1533 = None
        unsqueeze_1534: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
        unsqueeze_1535: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1534, -1);  unsqueeze_1534 = None
        add_443: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_575, unsqueeze_1535);  mul_575 = unsqueeze_1535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_191: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_443);  add_443 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_191: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_191, arg405_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_191 = arg405_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1536: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg406_1, -1);  arg406_1 = None
        unsqueeze_1537: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1536, -1);  unsqueeze_1536 = None
        sub_192: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_191, unsqueeze_1537);  convolution_191 = unsqueeze_1537 = None
        add_444: "f32[800]" = torch.ops.aten.add.Tensor(arg407_1, 0.001);  arg407_1 = None
        sqrt_192: "f32[800]" = torch.ops.aten.sqrt.default(add_444);  add_444 = None
        reciprocal_192: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_192);  sqrt_192 = None
        mul_576: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_192, 1);  reciprocal_192 = None
        unsqueeze_1538: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_576, -1);  mul_576 = None
        unsqueeze_1539: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1538, -1);  unsqueeze_1538 = None
        mul_577: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_1539);  sub_192 = unsqueeze_1539 = None
        unsqueeze_1540: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg408_1, -1);  arg408_1 = None
        unsqueeze_1541: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1540, -1);  unsqueeze_1540 = None
        mul_578: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_577, unsqueeze_1541);  mul_577 = unsqueeze_1541 = None
        unsqueeze_1542: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
        unsqueeze_1543: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, -1);  unsqueeze_1542 = None
        add_445: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_578, unsqueeze_1543);  mul_578 = unsqueeze_1543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_192: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_445);  add_445 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_192: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_192, arg410_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_192 = arg410_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_538: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_192, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_446: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_439, slice_538);  add_439 = slice_538 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_542: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_192, 1, 1024, 9223372036854775807);  convolution_192 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_120: "f32[8, 1024, 14, 14]" = torch.ops.aten.cat.default([cat_118, slice_542], 1);  cat_118 = slice_542 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_121: "f32[8, 2048, 14, 14]" = torch.ops.aten.cat.default([add_446, cat_120], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1544: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg411_1, -1);  arg411_1 = None
        unsqueeze_1545: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1544, -1);  unsqueeze_1544 = None
        sub_193: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(cat_121, unsqueeze_1545);  cat_121 = unsqueeze_1545 = None
        add_447: "f32[2048]" = torch.ops.aten.add.Tensor(arg412_1, 0.001);  arg412_1 = None
        sqrt_193: "f32[2048]" = torch.ops.aten.sqrt.default(add_447);  add_447 = None
        reciprocal_193: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_193);  sqrt_193 = None
        mul_579: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_193, 1);  reciprocal_193 = None
        unsqueeze_1546: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_579, -1);  mul_579 = None
        unsqueeze_1547: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1546, -1);  unsqueeze_1546 = None
        mul_580: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_1547);  sub_193 = unsqueeze_1547 = None
        unsqueeze_1548: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg413_1, -1);  arg413_1 = None
        unsqueeze_1549: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1548, -1);  unsqueeze_1548 = None
        mul_581: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_580, unsqueeze_1549);  mul_580 = unsqueeze_1549 = None
        unsqueeze_1550: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_1551: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1550, -1);  unsqueeze_1550 = None
        add_448: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_581, unsqueeze_1551);  mul_581 = unsqueeze_1551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_193: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_448);  add_448 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_193: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_193, arg415_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_193 = arg415_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1552: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg416_1, -1);  arg416_1 = None
        unsqueeze_1553: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1552, -1);  unsqueeze_1552 = None
        sub_194: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_193, unsqueeze_1553);  convolution_193 = unsqueeze_1553 = None
        add_449: "f32[800]" = torch.ops.aten.add.Tensor(arg417_1, 0.001);  arg417_1 = None
        sqrt_194: "f32[800]" = torch.ops.aten.sqrt.default(add_449);  add_449 = None
        reciprocal_194: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_194);  sqrt_194 = None
        mul_582: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_194, 1);  reciprocal_194 = None
        unsqueeze_1554: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_582, -1);  mul_582 = None
        unsqueeze_1555: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, -1);  unsqueeze_1554 = None
        mul_583: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_1555);  sub_194 = unsqueeze_1555 = None
        unsqueeze_1556: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg418_1, -1);  arg418_1 = None
        unsqueeze_1557: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1556, -1);  unsqueeze_1556 = None
        mul_584: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_583, unsqueeze_1557);  mul_583 = unsqueeze_1557 = None
        unsqueeze_1558: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg419_1, -1);  arg419_1 = None
        unsqueeze_1559: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1558, -1);  unsqueeze_1558 = None
        add_450: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_584, unsqueeze_1559);  mul_584 = unsqueeze_1559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_194: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_450);  add_450 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_194: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_194, arg420_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_194 = arg420_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1560: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg421_1, -1);  arg421_1 = None
        unsqueeze_1561: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1560, -1);  unsqueeze_1560 = None
        sub_195: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_194, unsqueeze_1561);  convolution_194 = unsqueeze_1561 = None
        add_451: "f32[800]" = torch.ops.aten.add.Tensor(arg422_1, 0.001);  arg422_1 = None
        sqrt_195: "f32[800]" = torch.ops.aten.sqrt.default(add_451);  add_451 = None
        reciprocal_195: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_195);  sqrt_195 = None
        mul_585: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_195, 1);  reciprocal_195 = None
        unsqueeze_1562: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_585, -1);  mul_585 = None
        unsqueeze_1563: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1562, -1);  unsqueeze_1562 = None
        mul_586: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_1563);  sub_195 = unsqueeze_1563 = None
        unsqueeze_1564: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg423_1, -1);  arg423_1 = None
        unsqueeze_1565: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1564, -1);  unsqueeze_1564 = None
        mul_587: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_586, unsqueeze_1565);  mul_586 = unsqueeze_1565 = None
        unsqueeze_1566: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg424_1, -1);  arg424_1 = None
        unsqueeze_1567: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, -1);  unsqueeze_1566 = None
        add_452: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_1567);  mul_587 = unsqueeze_1567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_195: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_452);  add_452 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_195: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_195, arg425_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_195 = arg425_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_546: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_195, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_453: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_446, slice_546);  add_446 = slice_546 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_550: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_195, 1, 1024, 9223372036854775807);  convolution_195 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_122: "f32[8, 1088, 14, 14]" = torch.ops.aten.cat.default([cat_120, slice_550], 1);  cat_120 = slice_550 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_123: "f32[8, 2112, 14, 14]" = torch.ops.aten.cat.default([add_453, cat_122], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1568: "f32[2112, 1]" = torch.ops.aten.unsqueeze.default(arg426_1, -1);  arg426_1 = None
        unsqueeze_1569: "f32[2112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1568, -1);  unsqueeze_1568 = None
        sub_196: "f32[8, 2112, 14, 14]" = torch.ops.aten.sub.Tensor(cat_123, unsqueeze_1569);  cat_123 = unsqueeze_1569 = None
        add_454: "f32[2112]" = torch.ops.aten.add.Tensor(arg427_1, 0.001);  arg427_1 = None
        sqrt_196: "f32[2112]" = torch.ops.aten.sqrt.default(add_454);  add_454 = None
        reciprocal_196: "f32[2112]" = torch.ops.aten.reciprocal.default(sqrt_196);  sqrt_196 = None
        mul_588: "f32[2112]" = torch.ops.aten.mul.Tensor(reciprocal_196, 1);  reciprocal_196 = None
        unsqueeze_1570: "f32[2112, 1]" = torch.ops.aten.unsqueeze.default(mul_588, -1);  mul_588 = None
        unsqueeze_1571: "f32[2112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1570, -1);  unsqueeze_1570 = None
        mul_589: "f32[8, 2112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_1571);  sub_196 = unsqueeze_1571 = None
        unsqueeze_1572: "f32[2112, 1]" = torch.ops.aten.unsqueeze.default(arg428_1, -1);  arg428_1 = None
        unsqueeze_1573: "f32[2112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1572, -1);  unsqueeze_1572 = None
        mul_590: "f32[8, 2112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_589, unsqueeze_1573);  mul_589 = unsqueeze_1573 = None
        unsqueeze_1574: "f32[2112, 1]" = torch.ops.aten.unsqueeze.default(arg429_1, -1);  arg429_1 = None
        unsqueeze_1575: "f32[2112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1574, -1);  unsqueeze_1574 = None
        add_455: "f32[8, 2112, 14, 14]" = torch.ops.aten.add.Tensor(mul_590, unsqueeze_1575);  mul_590 = unsqueeze_1575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_196: "f32[8, 2112, 14, 14]" = torch.ops.aten.relu.default(add_455);  add_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_196: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_196, arg430_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_196 = arg430_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1576: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg431_1, -1);  arg431_1 = None
        unsqueeze_1577: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1576, -1);  unsqueeze_1576 = None
        sub_197: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_196, unsqueeze_1577);  convolution_196 = unsqueeze_1577 = None
        add_456: "f32[800]" = torch.ops.aten.add.Tensor(arg432_1, 0.001);  arg432_1 = None
        sqrt_197: "f32[800]" = torch.ops.aten.sqrt.default(add_456);  add_456 = None
        reciprocal_197: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_197);  sqrt_197 = None
        mul_591: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_197, 1);  reciprocal_197 = None
        unsqueeze_1578: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_591, -1);  mul_591 = None
        unsqueeze_1579: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, -1);  unsqueeze_1578 = None
        mul_592: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_1579);  sub_197 = unsqueeze_1579 = None
        unsqueeze_1580: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg433_1, -1);  arg433_1 = None
        unsqueeze_1581: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1580, -1);  unsqueeze_1580 = None
        mul_593: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_592, unsqueeze_1581);  mul_592 = unsqueeze_1581 = None
        unsqueeze_1582: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg434_1, -1);  arg434_1 = None
        unsqueeze_1583: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1582, -1);  unsqueeze_1582 = None
        add_457: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_593, unsqueeze_1583);  mul_593 = unsqueeze_1583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_197: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_457);  add_457 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_197: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_197, arg435_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_197 = arg435_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1584: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg436_1, -1);  arg436_1 = None
        unsqueeze_1585: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1584, -1);  unsqueeze_1584 = None
        sub_198: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_197, unsqueeze_1585);  convolution_197 = unsqueeze_1585 = None
        add_458: "f32[800]" = torch.ops.aten.add.Tensor(arg437_1, 0.001);  arg437_1 = None
        sqrt_198: "f32[800]" = torch.ops.aten.sqrt.default(add_458);  add_458 = None
        reciprocal_198: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_198);  sqrt_198 = None
        mul_594: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_198, 1);  reciprocal_198 = None
        unsqueeze_1586: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_594, -1);  mul_594 = None
        unsqueeze_1587: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1586, -1);  unsqueeze_1586 = None
        mul_595: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_1587);  sub_198 = unsqueeze_1587 = None
        unsqueeze_1588: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg438_1, -1);  arg438_1 = None
        unsqueeze_1589: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1588, -1);  unsqueeze_1588 = None
        mul_596: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_1589);  mul_595 = unsqueeze_1589 = None
        unsqueeze_1590: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg439_1, -1);  arg439_1 = None
        unsqueeze_1591: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, -1);  unsqueeze_1590 = None
        add_459: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_596, unsqueeze_1591);  mul_596 = unsqueeze_1591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_198: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_459);  add_459 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_198: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_198, arg440_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_198 = arg440_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_554: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_198, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_460: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_453, slice_554);  add_453 = slice_554 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_558: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_198, 1, 1024, 9223372036854775807);  convolution_198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_124: "f32[8, 1152, 14, 14]" = torch.ops.aten.cat.default([cat_122, slice_558], 1);  cat_122 = slice_558 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_125: "f32[8, 2176, 14, 14]" = torch.ops.aten.cat.default([add_460, cat_124], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1592: "f32[2176, 1]" = torch.ops.aten.unsqueeze.default(arg441_1, -1);  arg441_1 = None
        unsqueeze_1593: "f32[2176, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1592, -1);  unsqueeze_1592 = None
        sub_199: "f32[8, 2176, 14, 14]" = torch.ops.aten.sub.Tensor(cat_125, unsqueeze_1593);  cat_125 = unsqueeze_1593 = None
        add_461: "f32[2176]" = torch.ops.aten.add.Tensor(arg442_1, 0.001);  arg442_1 = None
        sqrt_199: "f32[2176]" = torch.ops.aten.sqrt.default(add_461);  add_461 = None
        reciprocal_199: "f32[2176]" = torch.ops.aten.reciprocal.default(sqrt_199);  sqrt_199 = None
        mul_597: "f32[2176]" = torch.ops.aten.mul.Tensor(reciprocal_199, 1);  reciprocal_199 = None
        unsqueeze_1594: "f32[2176, 1]" = torch.ops.aten.unsqueeze.default(mul_597, -1);  mul_597 = None
        unsqueeze_1595: "f32[2176, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1594, -1);  unsqueeze_1594 = None
        mul_598: "f32[8, 2176, 14, 14]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_1595);  sub_199 = unsqueeze_1595 = None
        unsqueeze_1596: "f32[2176, 1]" = torch.ops.aten.unsqueeze.default(arg443_1, -1);  arg443_1 = None
        unsqueeze_1597: "f32[2176, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1596, -1);  unsqueeze_1596 = None
        mul_599: "f32[8, 2176, 14, 14]" = torch.ops.aten.mul.Tensor(mul_598, unsqueeze_1597);  mul_598 = unsqueeze_1597 = None
        unsqueeze_1598: "f32[2176, 1]" = torch.ops.aten.unsqueeze.default(arg444_1, -1);  arg444_1 = None
        unsqueeze_1599: "f32[2176, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1598, -1);  unsqueeze_1598 = None
        add_462: "f32[8, 2176, 14, 14]" = torch.ops.aten.add.Tensor(mul_599, unsqueeze_1599);  mul_599 = unsqueeze_1599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_199: "f32[8, 2176, 14, 14]" = torch.ops.aten.relu.default(add_462);  add_462 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_199: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_199, arg445_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_199 = arg445_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1600: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg446_1, -1);  arg446_1 = None
        unsqueeze_1601: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1600, -1);  unsqueeze_1600 = None
        sub_200: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_199, unsqueeze_1601);  convolution_199 = unsqueeze_1601 = None
        add_463: "f32[800]" = torch.ops.aten.add.Tensor(arg447_1, 0.001);  arg447_1 = None
        sqrt_200: "f32[800]" = torch.ops.aten.sqrt.default(add_463);  add_463 = None
        reciprocal_200: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_200);  sqrt_200 = None
        mul_600: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_200, 1);  reciprocal_200 = None
        unsqueeze_1602: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_600, -1);  mul_600 = None
        unsqueeze_1603: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, -1);  unsqueeze_1602 = None
        mul_601: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_1603);  sub_200 = unsqueeze_1603 = None
        unsqueeze_1604: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg448_1, -1);  arg448_1 = None
        unsqueeze_1605: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1604, -1);  unsqueeze_1604 = None
        mul_602: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_601, unsqueeze_1605);  mul_601 = unsqueeze_1605 = None
        unsqueeze_1606: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
        unsqueeze_1607: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1606, -1);  unsqueeze_1606 = None
        add_464: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_602, unsqueeze_1607);  mul_602 = unsqueeze_1607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_200: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_464);  add_464 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_200: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_200, arg450_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_200 = arg450_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1608: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg451_1, -1);  arg451_1 = None
        unsqueeze_1609: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1608, -1);  unsqueeze_1608 = None
        sub_201: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_200, unsqueeze_1609);  convolution_200 = unsqueeze_1609 = None
        add_465: "f32[800]" = torch.ops.aten.add.Tensor(arg452_1, 0.001);  arg452_1 = None
        sqrt_201: "f32[800]" = torch.ops.aten.sqrt.default(add_465);  add_465 = None
        reciprocal_201: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_201);  sqrt_201 = None
        mul_603: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_201, 1);  reciprocal_201 = None
        unsqueeze_1610: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_603, -1);  mul_603 = None
        unsqueeze_1611: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1610, -1);  unsqueeze_1610 = None
        mul_604: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_1611);  sub_201 = unsqueeze_1611 = None
        unsqueeze_1612: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg453_1, -1);  arg453_1 = None
        unsqueeze_1613: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1612, -1);  unsqueeze_1612 = None
        mul_605: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_604, unsqueeze_1613);  mul_604 = unsqueeze_1613 = None
        unsqueeze_1614: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
        unsqueeze_1615: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, -1);  unsqueeze_1614 = None
        add_466: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_605, unsqueeze_1615);  mul_605 = unsqueeze_1615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_201: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_466);  add_466 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_201: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_201, arg455_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_201 = arg455_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_562: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_201, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_467: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_460, slice_562);  add_460 = slice_562 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_566: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_201, 1, 1024, 9223372036854775807);  convolution_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_126: "f32[8, 1216, 14, 14]" = torch.ops.aten.cat.default([cat_124, slice_566], 1);  cat_124 = slice_566 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_127: "f32[8, 2240, 14, 14]" = torch.ops.aten.cat.default([add_467, cat_126], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1616: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(arg456_1, -1);  arg456_1 = None
        unsqueeze_1617: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1616, -1);  unsqueeze_1616 = None
        sub_202: "f32[8, 2240, 14, 14]" = torch.ops.aten.sub.Tensor(cat_127, unsqueeze_1617);  cat_127 = unsqueeze_1617 = None
        add_468: "f32[2240]" = torch.ops.aten.add.Tensor(arg457_1, 0.001);  arg457_1 = None
        sqrt_202: "f32[2240]" = torch.ops.aten.sqrt.default(add_468);  add_468 = None
        reciprocal_202: "f32[2240]" = torch.ops.aten.reciprocal.default(sqrt_202);  sqrt_202 = None
        mul_606: "f32[2240]" = torch.ops.aten.mul.Tensor(reciprocal_202, 1);  reciprocal_202 = None
        unsqueeze_1618: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(mul_606, -1);  mul_606 = None
        unsqueeze_1619: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1618, -1);  unsqueeze_1618 = None
        mul_607: "f32[8, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_1619);  sub_202 = unsqueeze_1619 = None
        unsqueeze_1620: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(arg458_1, -1);  arg458_1 = None
        unsqueeze_1621: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1620, -1);  unsqueeze_1620 = None
        mul_608: "f32[8, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_607, unsqueeze_1621);  mul_607 = unsqueeze_1621 = None
        unsqueeze_1622: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(arg459_1, -1);  arg459_1 = None
        unsqueeze_1623: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1622, -1);  unsqueeze_1622 = None
        add_469: "f32[8, 2240, 14, 14]" = torch.ops.aten.add.Tensor(mul_608, unsqueeze_1623);  mul_608 = unsqueeze_1623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_202: "f32[8, 2240, 14, 14]" = torch.ops.aten.relu.default(add_469);  add_469 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_202: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_202, arg460_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_202 = arg460_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1624: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg461_1, -1);  arg461_1 = None
        unsqueeze_1625: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1624, -1);  unsqueeze_1624 = None
        sub_203: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_202, unsqueeze_1625);  convolution_202 = unsqueeze_1625 = None
        add_470: "f32[800]" = torch.ops.aten.add.Tensor(arg462_1, 0.001);  arg462_1 = None
        sqrt_203: "f32[800]" = torch.ops.aten.sqrt.default(add_470);  add_470 = None
        reciprocal_203: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_203);  sqrt_203 = None
        mul_609: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_203, 1);  reciprocal_203 = None
        unsqueeze_1626: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_609, -1);  mul_609 = None
        unsqueeze_1627: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, -1);  unsqueeze_1626 = None
        mul_610: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_1627);  sub_203 = unsqueeze_1627 = None
        unsqueeze_1628: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg463_1, -1);  arg463_1 = None
        unsqueeze_1629: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1628, -1);  unsqueeze_1628 = None
        mul_611: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_610, unsqueeze_1629);  mul_610 = unsqueeze_1629 = None
        unsqueeze_1630: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
        unsqueeze_1631: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1630, -1);  unsqueeze_1630 = None
        add_471: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_611, unsqueeze_1631);  mul_611 = unsqueeze_1631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_203: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_471);  add_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_203: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_203, arg465_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_203 = arg465_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1632: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg466_1, -1);  arg466_1 = None
        unsqueeze_1633: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1632, -1);  unsqueeze_1632 = None
        sub_204: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_203, unsqueeze_1633);  convolution_203 = unsqueeze_1633 = None
        add_472: "f32[800]" = torch.ops.aten.add.Tensor(arg467_1, 0.001);  arg467_1 = None
        sqrt_204: "f32[800]" = torch.ops.aten.sqrt.default(add_472);  add_472 = None
        reciprocal_204: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_204);  sqrt_204 = None
        mul_612: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_204, 1);  reciprocal_204 = None
        unsqueeze_1634: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_612, -1);  mul_612 = None
        unsqueeze_1635: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1634, -1);  unsqueeze_1634 = None
        mul_613: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_1635);  sub_204 = unsqueeze_1635 = None
        unsqueeze_1636: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg468_1, -1);  arg468_1 = None
        unsqueeze_1637: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1636, -1);  unsqueeze_1636 = None
        mul_614: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_613, unsqueeze_1637);  mul_613 = unsqueeze_1637 = None
        unsqueeze_1638: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
        unsqueeze_1639: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, -1);  unsqueeze_1638 = None
        add_473: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_614, unsqueeze_1639);  mul_614 = unsqueeze_1639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_204: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_473);  add_473 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_204: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_204, arg470_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_204 = arg470_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_570: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_204, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_474: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_467, slice_570);  add_467 = slice_570 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_574: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_204, 1, 1024, 9223372036854775807);  convolution_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_128: "f32[8, 1280, 14, 14]" = torch.ops.aten.cat.default([cat_126, slice_574], 1);  cat_126 = slice_574 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_129: "f32[8, 2304, 14, 14]" = torch.ops.aten.cat.default([add_474, cat_128], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1640: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(arg471_1, -1);  arg471_1 = None
        unsqueeze_1641: "f32[2304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1640, -1);  unsqueeze_1640 = None
        sub_205: "f32[8, 2304, 14, 14]" = torch.ops.aten.sub.Tensor(cat_129, unsqueeze_1641);  cat_129 = unsqueeze_1641 = None
        add_475: "f32[2304]" = torch.ops.aten.add.Tensor(arg472_1, 0.001);  arg472_1 = None
        sqrt_205: "f32[2304]" = torch.ops.aten.sqrt.default(add_475);  add_475 = None
        reciprocal_205: "f32[2304]" = torch.ops.aten.reciprocal.default(sqrt_205);  sqrt_205 = None
        mul_615: "f32[2304]" = torch.ops.aten.mul.Tensor(reciprocal_205, 1);  reciprocal_205 = None
        unsqueeze_1642: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(mul_615, -1);  mul_615 = None
        unsqueeze_1643: "f32[2304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1642, -1);  unsqueeze_1642 = None
        mul_616: "f32[8, 2304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_1643);  sub_205 = unsqueeze_1643 = None
        unsqueeze_1644: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(arg473_1, -1);  arg473_1 = None
        unsqueeze_1645: "f32[2304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1644, -1);  unsqueeze_1644 = None
        mul_617: "f32[8, 2304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_1645);  mul_616 = unsqueeze_1645 = None
        unsqueeze_1646: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(arg474_1, -1);  arg474_1 = None
        unsqueeze_1647: "f32[2304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1646, -1);  unsqueeze_1646 = None
        add_476: "f32[8, 2304, 14, 14]" = torch.ops.aten.add.Tensor(mul_617, unsqueeze_1647);  mul_617 = unsqueeze_1647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_205: "f32[8, 2304, 14, 14]" = torch.ops.aten.relu.default(add_476);  add_476 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_205: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_205, arg475_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_205 = arg475_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1648: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg476_1, -1);  arg476_1 = None
        unsqueeze_1649: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1648, -1);  unsqueeze_1648 = None
        sub_206: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_205, unsqueeze_1649);  convolution_205 = unsqueeze_1649 = None
        add_477: "f32[800]" = torch.ops.aten.add.Tensor(arg477_1, 0.001);  arg477_1 = None
        sqrt_206: "f32[800]" = torch.ops.aten.sqrt.default(add_477);  add_477 = None
        reciprocal_206: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_206);  sqrt_206 = None
        mul_618: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_206, 1);  reciprocal_206 = None
        unsqueeze_1650: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_618, -1);  mul_618 = None
        unsqueeze_1651: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, -1);  unsqueeze_1650 = None
        mul_619: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_1651);  sub_206 = unsqueeze_1651 = None
        unsqueeze_1652: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg478_1, -1);  arg478_1 = None
        unsqueeze_1653: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1652, -1);  unsqueeze_1652 = None
        mul_620: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_619, unsqueeze_1653);  mul_619 = unsqueeze_1653 = None
        unsqueeze_1654: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg479_1, -1);  arg479_1 = None
        unsqueeze_1655: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1654, -1);  unsqueeze_1654 = None
        add_478: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_620, unsqueeze_1655);  mul_620 = unsqueeze_1655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_206: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_478);  add_478 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_206: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_206, arg480_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_206 = arg480_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1656: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg481_1, -1);  arg481_1 = None
        unsqueeze_1657: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1656, -1);  unsqueeze_1656 = None
        sub_207: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_206, unsqueeze_1657);  convolution_206 = unsqueeze_1657 = None
        add_479: "f32[800]" = torch.ops.aten.add.Tensor(arg482_1, 0.001);  arg482_1 = None
        sqrt_207: "f32[800]" = torch.ops.aten.sqrt.default(add_479);  add_479 = None
        reciprocal_207: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_207);  sqrt_207 = None
        mul_621: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_207, 1);  reciprocal_207 = None
        unsqueeze_1658: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_621, -1);  mul_621 = None
        unsqueeze_1659: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1658, -1);  unsqueeze_1658 = None
        mul_622: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_1659);  sub_207 = unsqueeze_1659 = None
        unsqueeze_1660: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg483_1, -1);  arg483_1 = None
        unsqueeze_1661: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1660, -1);  unsqueeze_1660 = None
        mul_623: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_622, unsqueeze_1661);  mul_622 = unsqueeze_1661 = None
        unsqueeze_1662: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg484_1, -1);  arg484_1 = None
        unsqueeze_1663: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, -1);  unsqueeze_1662 = None
        add_480: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_623, unsqueeze_1663);  mul_623 = unsqueeze_1663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_207: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_480);  add_480 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_207: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_207, arg485_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_207 = arg485_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_578: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_207, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_481: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_474, slice_578);  add_474 = slice_578 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_582: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_207, 1, 1024, 9223372036854775807);  convolution_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_130: "f32[8, 1344, 14, 14]" = torch.ops.aten.cat.default([cat_128, slice_582], 1);  cat_128 = slice_582 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_131: "f32[8, 2368, 14, 14]" = torch.ops.aten.cat.default([add_481, cat_130], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1664: "f32[2368, 1]" = torch.ops.aten.unsqueeze.default(arg486_1, -1);  arg486_1 = None
        unsqueeze_1665: "f32[2368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1664, -1);  unsqueeze_1664 = None
        sub_208: "f32[8, 2368, 14, 14]" = torch.ops.aten.sub.Tensor(cat_131, unsqueeze_1665);  cat_131 = unsqueeze_1665 = None
        add_482: "f32[2368]" = torch.ops.aten.add.Tensor(arg487_1, 0.001);  arg487_1 = None
        sqrt_208: "f32[2368]" = torch.ops.aten.sqrt.default(add_482);  add_482 = None
        reciprocal_208: "f32[2368]" = torch.ops.aten.reciprocal.default(sqrt_208);  sqrt_208 = None
        mul_624: "f32[2368]" = torch.ops.aten.mul.Tensor(reciprocal_208, 1);  reciprocal_208 = None
        unsqueeze_1666: "f32[2368, 1]" = torch.ops.aten.unsqueeze.default(mul_624, -1);  mul_624 = None
        unsqueeze_1667: "f32[2368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1666, -1);  unsqueeze_1666 = None
        mul_625: "f32[8, 2368, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_1667);  sub_208 = unsqueeze_1667 = None
        unsqueeze_1668: "f32[2368, 1]" = torch.ops.aten.unsqueeze.default(arg488_1, -1);  arg488_1 = None
        unsqueeze_1669: "f32[2368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1668, -1);  unsqueeze_1668 = None
        mul_626: "f32[8, 2368, 14, 14]" = torch.ops.aten.mul.Tensor(mul_625, unsqueeze_1669);  mul_625 = unsqueeze_1669 = None
        unsqueeze_1670: "f32[2368, 1]" = torch.ops.aten.unsqueeze.default(arg489_1, -1);  arg489_1 = None
        unsqueeze_1671: "f32[2368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1670, -1);  unsqueeze_1670 = None
        add_483: "f32[8, 2368, 14, 14]" = torch.ops.aten.add.Tensor(mul_626, unsqueeze_1671);  mul_626 = unsqueeze_1671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_208: "f32[8, 2368, 14, 14]" = torch.ops.aten.relu.default(add_483);  add_483 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_208: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_208, arg490_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_208 = arg490_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1672: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg491_1, -1);  arg491_1 = None
        unsqueeze_1673: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1672, -1);  unsqueeze_1672 = None
        sub_209: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_208, unsqueeze_1673);  convolution_208 = unsqueeze_1673 = None
        add_484: "f32[800]" = torch.ops.aten.add.Tensor(arg492_1, 0.001);  arg492_1 = None
        sqrt_209: "f32[800]" = torch.ops.aten.sqrt.default(add_484);  add_484 = None
        reciprocal_209: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_209);  sqrt_209 = None
        mul_627: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_209, 1);  reciprocal_209 = None
        unsqueeze_1674: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_627, -1);  mul_627 = None
        unsqueeze_1675: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1674, -1);  unsqueeze_1674 = None
        mul_628: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_1675);  sub_209 = unsqueeze_1675 = None
        unsqueeze_1676: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg493_1, -1);  arg493_1 = None
        unsqueeze_1677: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1676, -1);  unsqueeze_1676 = None
        mul_629: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_628, unsqueeze_1677);  mul_628 = unsqueeze_1677 = None
        unsqueeze_1678: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg494_1, -1);  arg494_1 = None
        unsqueeze_1679: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1678, -1);  unsqueeze_1678 = None
        add_485: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_629, unsqueeze_1679);  mul_629 = unsqueeze_1679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_209: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_485);  add_485 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_209: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_209, arg495_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_209 = arg495_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1680: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg496_1, -1);  arg496_1 = None
        unsqueeze_1681: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1680, -1);  unsqueeze_1680 = None
        sub_210: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_209, unsqueeze_1681);  convolution_209 = unsqueeze_1681 = None
        add_486: "f32[800]" = torch.ops.aten.add.Tensor(arg497_1, 0.001);  arg497_1 = None
        sqrt_210: "f32[800]" = torch.ops.aten.sqrt.default(add_486);  add_486 = None
        reciprocal_210: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_210);  sqrt_210 = None
        mul_630: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_210, 1);  reciprocal_210 = None
        unsqueeze_1682: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_630, -1);  mul_630 = None
        unsqueeze_1683: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1682, -1);  unsqueeze_1682 = None
        mul_631: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_1683);  sub_210 = unsqueeze_1683 = None
        unsqueeze_1684: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg498_1, -1);  arg498_1 = None
        unsqueeze_1685: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1684, -1);  unsqueeze_1684 = None
        mul_632: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_631, unsqueeze_1685);  mul_631 = unsqueeze_1685 = None
        unsqueeze_1686: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg499_1, -1);  arg499_1 = None
        unsqueeze_1687: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1686, -1);  unsqueeze_1686 = None
        add_487: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_632, unsqueeze_1687);  mul_632 = unsqueeze_1687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_210: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_487);  add_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_210: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_210, arg500_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_210 = arg500_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_586: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_210, 1, 0, 1024)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_488: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_481, slice_586);  add_481 = slice_586 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_590: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_210, 1, 1024, 9223372036854775807);  convolution_210 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_132: "f32[8, 1408, 14, 14]" = torch.ops.aten.cat.default([cat_130, slice_590], 1);  cat_130 = slice_590 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_133: "f32[8, 2432, 14, 14]" = torch.ops.aten.cat.default([add_488, cat_132], 1);  add_488 = cat_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1688: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg501_1, -1);  arg501_1 = None
        unsqueeze_1689: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1688, -1);  unsqueeze_1688 = None
        sub_211: "f32[8, 2432, 14, 14]" = torch.ops.aten.sub.Tensor(cat_133, unsqueeze_1689);  unsqueeze_1689 = None
        add_489: "f32[2432]" = torch.ops.aten.add.Tensor(arg502_1, 0.001);  arg502_1 = None
        sqrt_211: "f32[2432]" = torch.ops.aten.sqrt.default(add_489);  add_489 = None
        reciprocal_211: "f32[2432]" = torch.ops.aten.reciprocal.default(sqrt_211);  sqrt_211 = None
        mul_633: "f32[2432]" = torch.ops.aten.mul.Tensor(reciprocal_211, 1);  reciprocal_211 = None
        unsqueeze_1690: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(mul_633, -1);  mul_633 = None
        unsqueeze_1691: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1690, -1);  unsqueeze_1690 = None
        mul_634: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_1691);  sub_211 = unsqueeze_1691 = None
        unsqueeze_1692: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg503_1, -1);  arg503_1 = None
        unsqueeze_1693: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1692, -1);  unsqueeze_1692 = None
        mul_635: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(mul_634, unsqueeze_1693);  mul_634 = unsqueeze_1693 = None
        unsqueeze_1694: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg504_1, -1);  arg504_1 = None
        unsqueeze_1695: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1694, -1);  unsqueeze_1694 = None
        add_490: "f32[8, 2432, 14, 14]" = torch.ops.aten.add.Tensor(mul_635, unsqueeze_1695);  mul_635 = unsqueeze_1695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_211: "f32[8, 2432, 14, 14]" = torch.ops.aten.relu.default(add_490);  add_490 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_211: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(relu_211, arg505_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_211 = arg505_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:133 in forward, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
        slice_594: "f32[8, 2048, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_211, 1, 0, 2048)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1696: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg506_1, -1);  arg506_1 = None
        unsqueeze_1697: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1696, -1);  unsqueeze_1696 = None
        sub_212: "f32[8, 2432, 14, 14]" = torch.ops.aten.sub.Tensor(cat_133, unsqueeze_1697);  cat_133 = unsqueeze_1697 = None
        add_491: "f32[2432]" = torch.ops.aten.add.Tensor(arg507_1, 0.001);  arg507_1 = None
        sqrt_212: "f32[2432]" = torch.ops.aten.sqrt.default(add_491);  add_491 = None
        reciprocal_212: "f32[2432]" = torch.ops.aten.reciprocal.default(sqrt_212);  sqrt_212 = None
        mul_636: "f32[2432]" = torch.ops.aten.mul.Tensor(reciprocal_212, 1);  reciprocal_212 = None
        unsqueeze_1698: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(mul_636, -1);  mul_636 = None
        unsqueeze_1699: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1698, -1);  unsqueeze_1698 = None
        mul_637: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_1699);  sub_212 = unsqueeze_1699 = None
        unsqueeze_1700: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg508_1, -1);  arg508_1 = None
        unsqueeze_1701: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1700, -1);  unsqueeze_1700 = None
        mul_638: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_1701);  mul_637 = unsqueeze_1701 = None
        unsqueeze_1702: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg509_1, -1);  arg509_1 = None
        unsqueeze_1703: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1702, -1);  unsqueeze_1702 = None
        add_492: "f32[8, 2432, 14, 14]" = torch.ops.aten.add.Tensor(mul_638, unsqueeze_1703);  mul_638 = unsqueeze_1703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_212: "f32[8, 2432, 14, 14]" = torch.ops.aten.relu.default(add_492);  add_492 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_212: "f32[8, 1600, 14, 14]" = torch.ops.aten.convolution.default(relu_212, arg510_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_212 = arg510_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1704: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg511_1, -1);  arg511_1 = None
        unsqueeze_1705: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1704, -1);  unsqueeze_1704 = None
        sub_213: "f32[8, 1600, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_212, unsqueeze_1705);  convolution_212 = unsqueeze_1705 = None
        add_493: "f32[1600]" = torch.ops.aten.add.Tensor(arg512_1, 0.001);  arg512_1 = None
        sqrt_213: "f32[1600]" = torch.ops.aten.sqrt.default(add_493);  add_493 = None
        reciprocal_213: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_213);  sqrt_213 = None
        mul_639: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_213, 1);  reciprocal_213 = None
        unsqueeze_1706: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_639, -1);  mul_639 = None
        unsqueeze_1707: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1706, -1);  unsqueeze_1706 = None
        mul_640: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_1707);  sub_213 = unsqueeze_1707 = None
        unsqueeze_1708: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg513_1, -1);  arg513_1 = None
        unsqueeze_1709: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1708, -1);  unsqueeze_1708 = None
        mul_641: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(mul_640, unsqueeze_1709);  mul_640 = unsqueeze_1709 = None
        unsqueeze_1710: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg514_1, -1);  arg514_1 = None
        unsqueeze_1711: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1710, -1);  unsqueeze_1710 = None
        add_494: "f32[8, 1600, 14, 14]" = torch.ops.aten.add.Tensor(mul_641, unsqueeze_1711);  mul_641 = unsqueeze_1711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_213: "f32[8, 1600, 14, 14]" = torch.ops.aten.relu.default(add_494);  add_494 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_213: "f32[8, 1600, 7, 7]" = torch.ops.aten.convolution.default(relu_213, arg515_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 50);  relu_213 = arg515_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1712: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg516_1, -1);  arg516_1 = None
        unsqueeze_1713: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1712, -1);  unsqueeze_1712 = None
        sub_214: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_213, unsqueeze_1713);  convolution_213 = unsqueeze_1713 = None
        add_495: "f32[1600]" = torch.ops.aten.add.Tensor(arg517_1, 0.001);  arg517_1 = None
        sqrt_214: "f32[1600]" = torch.ops.aten.sqrt.default(add_495);  add_495 = None
        reciprocal_214: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_214);  sqrt_214 = None
        mul_642: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_214, 1);  reciprocal_214 = None
        unsqueeze_1714: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_642, -1);  mul_642 = None
        unsqueeze_1715: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1714, -1);  unsqueeze_1714 = None
        mul_643: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_1715);  sub_214 = unsqueeze_1715 = None
        unsqueeze_1716: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg518_1, -1);  arg518_1 = None
        unsqueeze_1717: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1716, -1);  unsqueeze_1716 = None
        mul_644: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(mul_643, unsqueeze_1717);  mul_643 = unsqueeze_1717 = None
        unsqueeze_1718: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg519_1, -1);  arg519_1 = None
        unsqueeze_1719: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1718, -1);  unsqueeze_1718 = None
        add_496: "f32[8, 1600, 7, 7]" = torch.ops.aten.add.Tensor(mul_644, unsqueeze_1719);  mul_644 = unsqueeze_1719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_214: "f32[8, 1600, 7, 7]" = torch.ops.aten.relu.default(add_496);  add_496 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_214: "f32[8, 2176, 7, 7]" = torch.ops.aten.convolution.default(relu_214, arg520_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_214 = arg520_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_602: "f32[8, 2048, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_214, 1, 0, 2048)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_497: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(slice_594, slice_602);  slice_594 = slice_602 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:134 in forward, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
        slice_598: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_211, 1, 2048, 9223372036854775807);  convolution_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_606: "f32[8, 128, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_214, 1, 2048, 9223372036854775807);  convolution_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_134: "f32[8, 384, 7, 7]" = torch.ops.aten.cat.default([slice_598, slice_606], 1);  slice_598 = slice_606 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_135: "f32[8, 2432, 7, 7]" = torch.ops.aten.cat.default([add_497, cat_134], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1720: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg521_1, -1);  arg521_1 = None
        unsqueeze_1721: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1720, -1);  unsqueeze_1720 = None
        sub_215: "f32[8, 2432, 7, 7]" = torch.ops.aten.sub.Tensor(cat_135, unsqueeze_1721);  cat_135 = unsqueeze_1721 = None
        add_498: "f32[2432]" = torch.ops.aten.add.Tensor(arg522_1, 0.001);  arg522_1 = None
        sqrt_215: "f32[2432]" = torch.ops.aten.sqrt.default(add_498);  add_498 = None
        reciprocal_215: "f32[2432]" = torch.ops.aten.reciprocal.default(sqrt_215);  sqrt_215 = None
        mul_645: "f32[2432]" = torch.ops.aten.mul.Tensor(reciprocal_215, 1);  reciprocal_215 = None
        unsqueeze_1722: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(mul_645, -1);  mul_645 = None
        unsqueeze_1723: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1722, -1);  unsqueeze_1722 = None
        mul_646: "f32[8, 2432, 7, 7]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_1723);  sub_215 = unsqueeze_1723 = None
        unsqueeze_1724: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg523_1, -1);  arg523_1 = None
        unsqueeze_1725: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1724, -1);  unsqueeze_1724 = None
        mul_647: "f32[8, 2432, 7, 7]" = torch.ops.aten.mul.Tensor(mul_646, unsqueeze_1725);  mul_646 = unsqueeze_1725 = None
        unsqueeze_1726: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg524_1, -1);  arg524_1 = None
        unsqueeze_1727: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1726, -1);  unsqueeze_1726 = None
        add_499: "f32[8, 2432, 7, 7]" = torch.ops.aten.add.Tensor(mul_647, unsqueeze_1727);  mul_647 = unsqueeze_1727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_215: "f32[8, 2432, 7, 7]" = torch.ops.aten.relu.default(add_499);  add_499 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_215: "f32[8, 1600, 7, 7]" = torch.ops.aten.convolution.default(relu_215, arg525_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_215 = arg525_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1728: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg526_1, -1);  arg526_1 = None
        unsqueeze_1729: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1728, -1);  unsqueeze_1728 = None
        sub_216: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_215, unsqueeze_1729);  convolution_215 = unsqueeze_1729 = None
        add_500: "f32[1600]" = torch.ops.aten.add.Tensor(arg527_1, 0.001);  arg527_1 = None
        sqrt_216: "f32[1600]" = torch.ops.aten.sqrt.default(add_500);  add_500 = None
        reciprocal_216: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_216);  sqrt_216 = None
        mul_648: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_216, 1);  reciprocal_216 = None
        unsqueeze_1730: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_648, -1);  mul_648 = None
        unsqueeze_1731: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1730, -1);  unsqueeze_1730 = None
        mul_649: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_1731);  sub_216 = unsqueeze_1731 = None
        unsqueeze_1732: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg528_1, -1);  arg528_1 = None
        unsqueeze_1733: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1732, -1);  unsqueeze_1732 = None
        mul_650: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(mul_649, unsqueeze_1733);  mul_649 = unsqueeze_1733 = None
        unsqueeze_1734: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg529_1, -1);  arg529_1 = None
        unsqueeze_1735: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1734, -1);  unsqueeze_1734 = None
        add_501: "f32[8, 1600, 7, 7]" = torch.ops.aten.add.Tensor(mul_650, unsqueeze_1735);  mul_650 = unsqueeze_1735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_216: "f32[8, 1600, 7, 7]" = torch.ops.aten.relu.default(add_501);  add_501 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_216: "f32[8, 1600, 7, 7]" = torch.ops.aten.convolution.default(relu_216, arg530_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_216 = arg530_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1736: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg531_1, -1);  arg531_1 = None
        unsqueeze_1737: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1736, -1);  unsqueeze_1736 = None
        sub_217: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_216, unsqueeze_1737);  convolution_216 = unsqueeze_1737 = None
        add_502: "f32[1600]" = torch.ops.aten.add.Tensor(arg532_1, 0.001);  arg532_1 = None
        sqrt_217: "f32[1600]" = torch.ops.aten.sqrt.default(add_502);  add_502 = None
        reciprocal_217: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_217);  sqrt_217 = None
        mul_651: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_217, 1);  reciprocal_217 = None
        unsqueeze_1738: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_651, -1);  mul_651 = None
        unsqueeze_1739: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1738, -1);  unsqueeze_1738 = None
        mul_652: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_1739);  sub_217 = unsqueeze_1739 = None
        unsqueeze_1740: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg533_1, -1);  arg533_1 = None
        unsqueeze_1741: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1740, -1);  unsqueeze_1740 = None
        mul_653: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(mul_652, unsqueeze_1741);  mul_652 = unsqueeze_1741 = None
        unsqueeze_1742: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg534_1, -1);  arg534_1 = None
        unsqueeze_1743: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1742, -1);  unsqueeze_1742 = None
        add_503: "f32[8, 1600, 7, 7]" = torch.ops.aten.add.Tensor(mul_653, unsqueeze_1743);  mul_653 = unsqueeze_1743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_217: "f32[8, 1600, 7, 7]" = torch.ops.aten.relu.default(add_503);  add_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_217: "f32[8, 2176, 7, 7]" = torch.ops.aten.convolution.default(relu_217, arg535_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_217 = arg535_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_610: "f32[8, 2048, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_217, 1, 0, 2048)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_504: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_497, slice_610);  add_497 = slice_610 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_614: "f32[8, 128, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_217, 1, 2048, 9223372036854775807);  convolution_217 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_136: "f32[8, 512, 7, 7]" = torch.ops.aten.cat.default([cat_134, slice_614], 1);  cat_134 = slice_614 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:118 in forward, code: x_in = torch.cat(x, dim=1)
        cat_137: "f32[8, 2560, 7, 7]" = torch.ops.aten.cat.default([add_504, cat_136], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1744: "f32[2560, 1]" = torch.ops.aten.unsqueeze.default(arg536_1, -1);  arg536_1 = None
        unsqueeze_1745: "f32[2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1744, -1);  unsqueeze_1744 = None
        sub_218: "f32[8, 2560, 7, 7]" = torch.ops.aten.sub.Tensor(cat_137, unsqueeze_1745);  cat_137 = unsqueeze_1745 = None
        add_505: "f32[2560]" = torch.ops.aten.add.Tensor(arg537_1, 0.001);  arg537_1 = None
        sqrt_218: "f32[2560]" = torch.ops.aten.sqrt.default(add_505);  add_505 = None
        reciprocal_218: "f32[2560]" = torch.ops.aten.reciprocal.default(sqrt_218);  sqrt_218 = None
        mul_654: "f32[2560]" = torch.ops.aten.mul.Tensor(reciprocal_218, 1);  reciprocal_218 = None
        unsqueeze_1746: "f32[2560, 1]" = torch.ops.aten.unsqueeze.default(mul_654, -1);  mul_654 = None
        unsqueeze_1747: "f32[2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1746, -1);  unsqueeze_1746 = None
        mul_655: "f32[8, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_1747);  sub_218 = unsqueeze_1747 = None
        unsqueeze_1748: "f32[2560, 1]" = torch.ops.aten.unsqueeze.default(arg538_1, -1);  arg538_1 = None
        unsqueeze_1749: "f32[2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1748, -1);  unsqueeze_1748 = None
        mul_656: "f32[8, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(mul_655, unsqueeze_1749);  mul_655 = unsqueeze_1749 = None
        unsqueeze_1750: "f32[2560, 1]" = torch.ops.aten.unsqueeze.default(arg539_1, -1);  arg539_1 = None
        unsqueeze_1751: "f32[2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1750, -1);  unsqueeze_1750 = None
        add_506: "f32[8, 2560, 7, 7]" = torch.ops.aten.add.Tensor(mul_656, unsqueeze_1751);  mul_656 = unsqueeze_1751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_218: "f32[8, 2560, 7, 7]" = torch.ops.aten.relu.default(add_506);  add_506 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_218: "f32[8, 1600, 7, 7]" = torch.ops.aten.convolution.default(relu_218, arg540_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_218 = arg540_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1752: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg541_1, -1);  arg541_1 = None
        unsqueeze_1753: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1752, -1);  unsqueeze_1752 = None
        sub_219: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_218, unsqueeze_1753);  convolution_218 = unsqueeze_1753 = None
        add_507: "f32[1600]" = torch.ops.aten.add.Tensor(arg542_1, 0.001);  arg542_1 = None
        sqrt_219: "f32[1600]" = torch.ops.aten.sqrt.default(add_507);  add_507 = None
        reciprocal_219: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_219);  sqrt_219 = None
        mul_657: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_219, 1);  reciprocal_219 = None
        unsqueeze_1754: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_657, -1);  mul_657 = None
        unsqueeze_1755: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1754, -1);  unsqueeze_1754 = None
        mul_658: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_1755);  sub_219 = unsqueeze_1755 = None
        unsqueeze_1756: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg543_1, -1);  arg543_1 = None
        unsqueeze_1757: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1756, -1);  unsqueeze_1756 = None
        mul_659: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_1757);  mul_658 = unsqueeze_1757 = None
        unsqueeze_1758: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg544_1, -1);  arg544_1 = None
        unsqueeze_1759: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1758, -1);  unsqueeze_1758 = None
        add_508: "f32[8, 1600, 7, 7]" = torch.ops.aten.add.Tensor(mul_659, unsqueeze_1759);  mul_659 = unsqueeze_1759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_219: "f32[8, 1600, 7, 7]" = torch.ops.aten.relu.default(add_508);  add_508 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_219: "f32[8, 1600, 7, 7]" = torch.ops.aten.convolution.default(relu_219, arg545_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_219 = arg545_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1760: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg546_1, -1);  arg546_1 = None
        unsqueeze_1761: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1760, -1);  unsqueeze_1760 = None
        sub_220: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_219, unsqueeze_1761);  convolution_219 = unsqueeze_1761 = None
        add_509: "f32[1600]" = torch.ops.aten.add.Tensor(arg547_1, 0.001);  arg547_1 = None
        sqrt_220: "f32[1600]" = torch.ops.aten.sqrt.default(add_509);  add_509 = None
        reciprocal_220: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_220);  sqrt_220 = None
        mul_660: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_220, 1);  reciprocal_220 = None
        unsqueeze_1762: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_660, -1);  mul_660 = None
        unsqueeze_1763: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1762, -1);  unsqueeze_1762 = None
        mul_661: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_1763);  sub_220 = unsqueeze_1763 = None
        unsqueeze_1764: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg548_1, -1);  arg548_1 = None
        unsqueeze_1765: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1764, -1);  unsqueeze_1764 = None
        mul_662: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(mul_661, unsqueeze_1765);  mul_661 = unsqueeze_1765 = None
        unsqueeze_1766: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg549_1, -1);  arg549_1 = None
        unsqueeze_1767: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1766, -1);  unsqueeze_1766 = None
        add_510: "f32[8, 1600, 7, 7]" = torch.ops.aten.add.Tensor(mul_662, unsqueeze_1767);  mul_662 = unsqueeze_1767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_220: "f32[8, 1600, 7, 7]" = torch.ops.aten.relu.default(add_510);  add_510 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:53 in forward, code: return self.conv(self.bn(x))
        convolution_220: "f32[8, 2176, 7, 7]" = torch.ops.aten.convolution.default(relu_220, arg550_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_220 = arg550_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:143 in forward, code: out1 = x_in[:, :self.num_1x1_c, :, :]
        slice_618: "f32[8, 2048, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_220, 1, 0, 2048)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:145 in forward, code: resid = x_s1 + out1
        add_511: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_504, slice_618);  add_504 = slice_618 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:144 in forward, code: out2 = x_in[:, self.num_1x1_c:, :, :]
        slice_622: "f32[8, 128, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_220, 1, 2048, 9223372036854775807);  convolution_220 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:146 in forward, code: dense = torch.cat([x_s2, out2], dim=1)
        cat_138: "f32[8, 640, 7, 7]" = torch.ops.aten.cat.default([cat_136, slice_622], 1);  cat_136 = slice_622 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:42 in forward, code: x = torch.cat(x, dim=1)
        cat_139: "f32[8, 2688, 7, 7]" = torch.ops.aten.cat.default([add_511, cat_138], 1);  add_511 = cat_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1768: "f32[2688, 1]" = torch.ops.aten.unsqueeze.default(arg551_1, -1);  arg551_1 = None
        unsqueeze_1769: "f32[2688, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1768, -1);  unsqueeze_1768 = None
        sub_221: "f32[8, 2688, 7, 7]" = torch.ops.aten.sub.Tensor(cat_139, unsqueeze_1769);  cat_139 = unsqueeze_1769 = None
        add_512: "f32[2688]" = torch.ops.aten.add.Tensor(arg552_1, 0.001);  arg552_1 = None
        sqrt_221: "f32[2688]" = torch.ops.aten.sqrt.default(add_512);  add_512 = None
        reciprocal_221: "f32[2688]" = torch.ops.aten.reciprocal.default(sqrt_221);  sqrt_221 = None
        mul_663: "f32[2688]" = torch.ops.aten.mul.Tensor(reciprocal_221, 1);  reciprocal_221 = None
        unsqueeze_1770: "f32[2688, 1]" = torch.ops.aten.unsqueeze.default(mul_663, -1);  mul_663 = None
        unsqueeze_1771: "f32[2688, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1770, -1);  unsqueeze_1770 = None
        mul_664: "f32[8, 2688, 7, 7]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_1771);  sub_221 = unsqueeze_1771 = None
        unsqueeze_1772: "f32[2688, 1]" = torch.ops.aten.unsqueeze.default(arg553_1, -1);  arg553_1 = None
        unsqueeze_1773: "f32[2688, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1772, -1);  unsqueeze_1772 = None
        mul_665: "f32[8, 2688, 7, 7]" = torch.ops.aten.mul.Tensor(mul_664, unsqueeze_1773);  mul_664 = unsqueeze_1773 = None
        unsqueeze_1774: "f32[2688, 1]" = torch.ops.aten.unsqueeze.default(arg554_1, -1);  arg554_1 = None
        unsqueeze_1775: "f32[2688, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1774, -1);  unsqueeze_1774 = None
        add_513: "f32[8, 2688, 7, 7]" = torch.ops.aten.add.Tensor(mul_665, unsqueeze_1775);  mul_665 = unsqueeze_1775 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_221: "f32[8, 2688, 7, 7]" = torch.ops.aten.relu.default(add_513);  add_513 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_1: "f32[8, 2688, 1, 1]" = torch.ops.aten.mean.dim(relu_221, [-1, -2], True);  relu_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:274 in forward_head, code: x = self.classifier(x)
        convolution_221: "f32[8, 1000, 1, 1]" = torch.ops.aten.convolution.default(mean_1, arg555_1, arg556_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg555_1 = arg556_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/dpn.py:275 in forward_head, code: return self.flatten(x)
        view_1: "f32[8, 1000]" = torch.ops.aten.reshape.default(convolution_221, [8, 1000]);  convolution_221 = None
        return (view_1,)
        