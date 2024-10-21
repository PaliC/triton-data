class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 3, 7, 7]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[104, 64, 1, 1]", arg7_1: "f32[104]", arg8_1: "f32[104]", arg9_1: "f32[104]", arg10_1: "f32[104]", arg11_1: "f32[26, 26, 3, 3]", arg12_1: "f32[26]", arg13_1: "f32[26]", arg14_1: "f32[26]", arg15_1: "f32[26]", arg16_1: "f32[26, 26, 3, 3]", arg17_1: "f32[26]", arg18_1: "f32[26]", arg19_1: "f32[26]", arg20_1: "f32[26]", arg21_1: "f32[26, 26, 3, 3]", arg22_1: "f32[26]", arg23_1: "f32[26]", arg24_1: "f32[26]", arg25_1: "f32[26]", arg26_1: "f32[256, 104, 1, 1]", arg27_1: "f32[256]", arg28_1: "f32[256]", arg29_1: "f32[256]", arg30_1: "f32[256]", arg31_1: "f32[256, 64, 1, 1]", arg32_1: "f32[256]", arg33_1: "f32[256]", arg34_1: "f32[256]", arg35_1: "f32[256]", arg36_1: "f32[104, 256, 1, 1]", arg37_1: "f32[104]", arg38_1: "f32[104]", arg39_1: "f32[104]", arg40_1: "f32[104]", arg41_1: "f32[26, 26, 3, 3]", arg42_1: "f32[26]", arg43_1: "f32[26]", arg44_1: "f32[26]", arg45_1: "f32[26]", arg46_1: "f32[26, 26, 3, 3]", arg47_1: "f32[26]", arg48_1: "f32[26]", arg49_1: "f32[26]", arg50_1: "f32[26]", arg51_1: "f32[26, 26, 3, 3]", arg52_1: "f32[26]", arg53_1: "f32[26]", arg54_1: "f32[26]", arg55_1: "f32[26]", arg56_1: "f32[256, 104, 1, 1]", arg57_1: "f32[256]", arg58_1: "f32[256]", arg59_1: "f32[256]", arg60_1: "f32[256]", arg61_1: "f32[104, 256, 1, 1]", arg62_1: "f32[104]", arg63_1: "f32[104]", arg64_1: "f32[104]", arg65_1: "f32[104]", arg66_1: "f32[26, 26, 3, 3]", arg67_1: "f32[26]", arg68_1: "f32[26]", arg69_1: "f32[26]", arg70_1: "f32[26]", arg71_1: "f32[26, 26, 3, 3]", arg72_1: "f32[26]", arg73_1: "f32[26]", arg74_1: "f32[26]", arg75_1: "f32[26]", arg76_1: "f32[26, 26, 3, 3]", arg77_1: "f32[26]", arg78_1: "f32[26]", arg79_1: "f32[26]", arg80_1: "f32[26]", arg81_1: "f32[256, 104, 1, 1]", arg82_1: "f32[256]", arg83_1: "f32[256]", arg84_1: "f32[256]", arg85_1: "f32[256]", arg86_1: "f32[208, 256, 1, 1]", arg87_1: "f32[208]", arg88_1: "f32[208]", arg89_1: "f32[208]", arg90_1: "f32[208]", arg91_1: "f32[52, 52, 3, 3]", arg92_1: "f32[52]", arg93_1: "f32[52]", arg94_1: "f32[52]", arg95_1: "f32[52]", arg96_1: "f32[52, 52, 3, 3]", arg97_1: "f32[52]", arg98_1: "f32[52]", arg99_1: "f32[52]", arg100_1: "f32[52]", arg101_1: "f32[52, 52, 3, 3]", arg102_1: "f32[52]", arg103_1: "f32[52]", arg104_1: "f32[52]", arg105_1: "f32[52]", arg106_1: "f32[512, 208, 1, 1]", arg107_1: "f32[512]", arg108_1: "f32[512]", arg109_1: "f32[512]", arg110_1: "f32[512]", arg111_1: "f32[512, 256, 1, 1]", arg112_1: "f32[512]", arg113_1: "f32[512]", arg114_1: "f32[512]", arg115_1: "f32[512]", arg116_1: "f32[208, 512, 1, 1]", arg117_1: "f32[208]", arg118_1: "f32[208]", arg119_1: "f32[208]", arg120_1: "f32[208]", arg121_1: "f32[52, 52, 3, 3]", arg122_1: "f32[52]", arg123_1: "f32[52]", arg124_1: "f32[52]", arg125_1: "f32[52]", arg126_1: "f32[52, 52, 3, 3]", arg127_1: "f32[52]", arg128_1: "f32[52]", arg129_1: "f32[52]", arg130_1: "f32[52]", arg131_1: "f32[52, 52, 3, 3]", arg132_1: "f32[52]", arg133_1: "f32[52]", arg134_1: "f32[52]", arg135_1: "f32[52]", arg136_1: "f32[512, 208, 1, 1]", arg137_1: "f32[512]", arg138_1: "f32[512]", arg139_1: "f32[512]", arg140_1: "f32[512]", arg141_1: "f32[208, 512, 1, 1]", arg142_1: "f32[208]", arg143_1: "f32[208]", arg144_1: "f32[208]", arg145_1: "f32[208]", arg146_1: "f32[52, 52, 3, 3]", arg147_1: "f32[52]", arg148_1: "f32[52]", arg149_1: "f32[52]", arg150_1: "f32[52]", arg151_1: "f32[52, 52, 3, 3]", arg152_1: "f32[52]", arg153_1: "f32[52]", arg154_1: "f32[52]", arg155_1: "f32[52]", arg156_1: "f32[52, 52, 3, 3]", arg157_1: "f32[52]", arg158_1: "f32[52]", arg159_1: "f32[52]", arg160_1: "f32[52]", arg161_1: "f32[512, 208, 1, 1]", arg162_1: "f32[512]", arg163_1: "f32[512]", arg164_1: "f32[512]", arg165_1: "f32[512]", arg166_1: "f32[208, 512, 1, 1]", arg167_1: "f32[208]", arg168_1: "f32[208]", arg169_1: "f32[208]", arg170_1: "f32[208]", arg171_1: "f32[52, 52, 3, 3]", arg172_1: "f32[52]", arg173_1: "f32[52]", arg174_1: "f32[52]", arg175_1: "f32[52]", arg176_1: "f32[52, 52, 3, 3]", arg177_1: "f32[52]", arg178_1: "f32[52]", arg179_1: "f32[52]", arg180_1: "f32[52]", arg181_1: "f32[52, 52, 3, 3]", arg182_1: "f32[52]", arg183_1: "f32[52]", arg184_1: "f32[52]", arg185_1: "f32[52]", arg186_1: "f32[512, 208, 1, 1]", arg187_1: "f32[512]", arg188_1: "f32[512]", arg189_1: "f32[512]", arg190_1: "f32[512]", arg191_1: "f32[416, 512, 1, 1]", arg192_1: "f32[416]", arg193_1: "f32[416]", arg194_1: "f32[416]", arg195_1: "f32[416]", arg196_1: "f32[104, 104, 3, 3]", arg197_1: "f32[104]", arg198_1: "f32[104]", arg199_1: "f32[104]", arg200_1: "f32[104]", arg201_1: "f32[104, 104, 3, 3]", arg202_1: "f32[104]", arg203_1: "f32[104]", arg204_1: "f32[104]", arg205_1: "f32[104]", arg206_1: "f32[104, 104, 3, 3]", arg207_1: "f32[104]", arg208_1: "f32[104]", arg209_1: "f32[104]", arg210_1: "f32[104]", arg211_1: "f32[1024, 416, 1, 1]", arg212_1: "f32[1024]", arg213_1: "f32[1024]", arg214_1: "f32[1024]", arg215_1: "f32[1024]", arg216_1: "f32[1024, 512, 1, 1]", arg217_1: "f32[1024]", arg218_1: "f32[1024]", arg219_1: "f32[1024]", arg220_1: "f32[1024]", arg221_1: "f32[416, 1024, 1, 1]", arg222_1: "f32[416]", arg223_1: "f32[416]", arg224_1: "f32[416]", arg225_1: "f32[416]", arg226_1: "f32[104, 104, 3, 3]", arg227_1: "f32[104]", arg228_1: "f32[104]", arg229_1: "f32[104]", arg230_1: "f32[104]", arg231_1: "f32[104, 104, 3, 3]", arg232_1: "f32[104]", arg233_1: "f32[104]", arg234_1: "f32[104]", arg235_1: "f32[104]", arg236_1: "f32[104, 104, 3, 3]", arg237_1: "f32[104]", arg238_1: "f32[104]", arg239_1: "f32[104]", arg240_1: "f32[104]", arg241_1: "f32[1024, 416, 1, 1]", arg242_1: "f32[1024]", arg243_1: "f32[1024]", arg244_1: "f32[1024]", arg245_1: "f32[1024]", arg246_1: "f32[416, 1024, 1, 1]", arg247_1: "f32[416]", arg248_1: "f32[416]", arg249_1: "f32[416]", arg250_1: "f32[416]", arg251_1: "f32[104, 104, 3, 3]", arg252_1: "f32[104]", arg253_1: "f32[104]", arg254_1: "f32[104]", arg255_1: "f32[104]", arg256_1: "f32[104, 104, 3, 3]", arg257_1: "f32[104]", arg258_1: "f32[104]", arg259_1: "f32[104]", arg260_1: "f32[104]", arg261_1: "f32[104, 104, 3, 3]", arg262_1: "f32[104]", arg263_1: "f32[104]", arg264_1: "f32[104]", arg265_1: "f32[104]", arg266_1: "f32[1024, 416, 1, 1]", arg267_1: "f32[1024]", arg268_1: "f32[1024]", arg269_1: "f32[1024]", arg270_1: "f32[1024]", arg271_1: "f32[416, 1024, 1, 1]", arg272_1: "f32[416]", arg273_1: "f32[416]", arg274_1: "f32[416]", arg275_1: "f32[416]", arg276_1: "f32[104, 104, 3, 3]", arg277_1: "f32[104]", arg278_1: "f32[104]", arg279_1: "f32[104]", arg280_1: "f32[104]", arg281_1: "f32[104, 104, 3, 3]", arg282_1: "f32[104]", arg283_1: "f32[104]", arg284_1: "f32[104]", arg285_1: "f32[104]", arg286_1: "f32[104, 104, 3, 3]", arg287_1: "f32[104]", arg288_1: "f32[104]", arg289_1: "f32[104]", arg290_1: "f32[104]", arg291_1: "f32[1024, 416, 1, 1]", arg292_1: "f32[1024]", arg293_1: "f32[1024]", arg294_1: "f32[1024]", arg295_1: "f32[1024]", arg296_1: "f32[416, 1024, 1, 1]", arg297_1: "f32[416]", arg298_1: "f32[416]", arg299_1: "f32[416]", arg300_1: "f32[416]", arg301_1: "f32[104, 104, 3, 3]", arg302_1: "f32[104]", arg303_1: "f32[104]", arg304_1: "f32[104]", arg305_1: "f32[104]", arg306_1: "f32[104, 104, 3, 3]", arg307_1: "f32[104]", arg308_1: "f32[104]", arg309_1: "f32[104]", arg310_1: "f32[104]", arg311_1: "f32[104, 104, 3, 3]", arg312_1: "f32[104]", arg313_1: "f32[104]", arg314_1: "f32[104]", arg315_1: "f32[104]", arg316_1: "f32[1024, 416, 1, 1]", arg317_1: "f32[1024]", arg318_1: "f32[1024]", arg319_1: "f32[1024]", arg320_1: "f32[1024]", arg321_1: "f32[416, 1024, 1, 1]", arg322_1: "f32[416]", arg323_1: "f32[416]", arg324_1: "f32[416]", arg325_1: "f32[416]", arg326_1: "f32[104, 104, 3, 3]", arg327_1: "f32[104]", arg328_1: "f32[104]", arg329_1: "f32[104]", arg330_1: "f32[104]", arg331_1: "f32[104, 104, 3, 3]", arg332_1: "f32[104]", arg333_1: "f32[104]", arg334_1: "f32[104]", arg335_1: "f32[104]", arg336_1: "f32[104, 104, 3, 3]", arg337_1: "f32[104]", arg338_1: "f32[104]", arg339_1: "f32[104]", arg340_1: "f32[104]", arg341_1: "f32[1024, 416, 1, 1]", arg342_1: "f32[1024]", arg343_1: "f32[1024]", arg344_1: "f32[1024]", arg345_1: "f32[1024]", arg346_1: "f32[416, 1024, 1, 1]", arg347_1: "f32[416]", arg348_1: "f32[416]", arg349_1: "f32[416]", arg350_1: "f32[416]", arg351_1: "f32[104, 104, 3, 3]", arg352_1: "f32[104]", arg353_1: "f32[104]", arg354_1: "f32[104]", arg355_1: "f32[104]", arg356_1: "f32[104, 104, 3, 3]", arg357_1: "f32[104]", arg358_1: "f32[104]", arg359_1: "f32[104]", arg360_1: "f32[104]", arg361_1: "f32[104, 104, 3, 3]", arg362_1: "f32[104]", arg363_1: "f32[104]", arg364_1: "f32[104]", arg365_1: "f32[104]", arg366_1: "f32[1024, 416, 1, 1]", arg367_1: "f32[1024]", arg368_1: "f32[1024]", arg369_1: "f32[1024]", arg370_1: "f32[1024]", arg371_1: "f32[416, 1024, 1, 1]", arg372_1: "f32[416]", arg373_1: "f32[416]", arg374_1: "f32[416]", arg375_1: "f32[416]", arg376_1: "f32[104, 104, 3, 3]", arg377_1: "f32[104]", arg378_1: "f32[104]", arg379_1: "f32[104]", arg380_1: "f32[104]", arg381_1: "f32[104, 104, 3, 3]", arg382_1: "f32[104]", arg383_1: "f32[104]", arg384_1: "f32[104]", arg385_1: "f32[104]", arg386_1: "f32[104, 104, 3, 3]", arg387_1: "f32[104]", arg388_1: "f32[104]", arg389_1: "f32[104]", arg390_1: "f32[104]", arg391_1: "f32[1024, 416, 1, 1]", arg392_1: "f32[1024]", arg393_1: "f32[1024]", arg394_1: "f32[1024]", arg395_1: "f32[1024]", arg396_1: "f32[416, 1024, 1, 1]", arg397_1: "f32[416]", arg398_1: "f32[416]", arg399_1: "f32[416]", arg400_1: "f32[416]", arg401_1: "f32[104, 104, 3, 3]", arg402_1: "f32[104]", arg403_1: "f32[104]", arg404_1: "f32[104]", arg405_1: "f32[104]", arg406_1: "f32[104, 104, 3, 3]", arg407_1: "f32[104]", arg408_1: "f32[104]", arg409_1: "f32[104]", arg410_1: "f32[104]", arg411_1: "f32[104, 104, 3, 3]", arg412_1: "f32[104]", arg413_1: "f32[104]", arg414_1: "f32[104]", arg415_1: "f32[104]", arg416_1: "f32[1024, 416, 1, 1]", arg417_1: "f32[1024]", arg418_1: "f32[1024]", arg419_1: "f32[1024]", arg420_1: "f32[1024]", arg421_1: "f32[416, 1024, 1, 1]", arg422_1: "f32[416]", arg423_1: "f32[416]", arg424_1: "f32[416]", arg425_1: "f32[416]", arg426_1: "f32[104, 104, 3, 3]", arg427_1: "f32[104]", arg428_1: "f32[104]", arg429_1: "f32[104]", arg430_1: "f32[104]", arg431_1: "f32[104, 104, 3, 3]", arg432_1: "f32[104]", arg433_1: "f32[104]", arg434_1: "f32[104]", arg435_1: "f32[104]", arg436_1: "f32[104, 104, 3, 3]", arg437_1: "f32[104]", arg438_1: "f32[104]", arg439_1: "f32[104]", arg440_1: "f32[104]", arg441_1: "f32[1024, 416, 1, 1]", arg442_1: "f32[1024]", arg443_1: "f32[1024]", arg444_1: "f32[1024]", arg445_1: "f32[1024]", arg446_1: "f32[416, 1024, 1, 1]", arg447_1: "f32[416]", arg448_1: "f32[416]", arg449_1: "f32[416]", arg450_1: "f32[416]", arg451_1: "f32[104, 104, 3, 3]", arg452_1: "f32[104]", arg453_1: "f32[104]", arg454_1: "f32[104]", arg455_1: "f32[104]", arg456_1: "f32[104, 104, 3, 3]", arg457_1: "f32[104]", arg458_1: "f32[104]", arg459_1: "f32[104]", arg460_1: "f32[104]", arg461_1: "f32[104, 104, 3, 3]", arg462_1: "f32[104]", arg463_1: "f32[104]", arg464_1: "f32[104]", arg465_1: "f32[104]", arg466_1: "f32[1024, 416, 1, 1]", arg467_1: "f32[1024]", arg468_1: "f32[1024]", arg469_1: "f32[1024]", arg470_1: "f32[1024]", arg471_1: "f32[416, 1024, 1, 1]", arg472_1: "f32[416]", arg473_1: "f32[416]", arg474_1: "f32[416]", arg475_1: "f32[416]", arg476_1: "f32[104, 104, 3, 3]", arg477_1: "f32[104]", arg478_1: "f32[104]", arg479_1: "f32[104]", arg480_1: "f32[104]", arg481_1: "f32[104, 104, 3, 3]", arg482_1: "f32[104]", arg483_1: "f32[104]", arg484_1: "f32[104]", arg485_1: "f32[104]", arg486_1: "f32[104, 104, 3, 3]", arg487_1: "f32[104]", arg488_1: "f32[104]", arg489_1: "f32[104]", arg490_1: "f32[104]", arg491_1: "f32[1024, 416, 1, 1]", arg492_1: "f32[1024]", arg493_1: "f32[1024]", arg494_1: "f32[1024]", arg495_1: "f32[1024]", arg496_1: "f32[416, 1024, 1, 1]", arg497_1: "f32[416]", arg498_1: "f32[416]", arg499_1: "f32[416]", arg500_1: "f32[416]", arg501_1: "f32[104, 104, 3, 3]", arg502_1: "f32[104]", arg503_1: "f32[104]", arg504_1: "f32[104]", arg505_1: "f32[104]", arg506_1: "f32[104, 104, 3, 3]", arg507_1: "f32[104]", arg508_1: "f32[104]", arg509_1: "f32[104]", arg510_1: "f32[104]", arg511_1: "f32[104, 104, 3, 3]", arg512_1: "f32[104]", arg513_1: "f32[104]", arg514_1: "f32[104]", arg515_1: "f32[104]", arg516_1: "f32[1024, 416, 1, 1]", arg517_1: "f32[1024]", arg518_1: "f32[1024]", arg519_1: "f32[1024]", arg520_1: "f32[1024]", arg521_1: "f32[416, 1024, 1, 1]", arg522_1: "f32[416]", arg523_1: "f32[416]", arg524_1: "f32[416]", arg525_1: "f32[416]", arg526_1: "f32[104, 104, 3, 3]", arg527_1: "f32[104]", arg528_1: "f32[104]", arg529_1: "f32[104]", arg530_1: "f32[104]", arg531_1: "f32[104, 104, 3, 3]", arg532_1: "f32[104]", arg533_1: "f32[104]", arg534_1: "f32[104]", arg535_1: "f32[104]", arg536_1: "f32[104, 104, 3, 3]", arg537_1: "f32[104]", arg538_1: "f32[104]", arg539_1: "f32[104]", arg540_1: "f32[104]", arg541_1: "f32[1024, 416, 1, 1]", arg542_1: "f32[1024]", arg543_1: "f32[1024]", arg544_1: "f32[1024]", arg545_1: "f32[1024]", arg546_1: "f32[416, 1024, 1, 1]", arg547_1: "f32[416]", arg548_1: "f32[416]", arg549_1: "f32[416]", arg550_1: "f32[416]", arg551_1: "f32[104, 104, 3, 3]", arg552_1: "f32[104]", arg553_1: "f32[104]", arg554_1: "f32[104]", arg555_1: "f32[104]", arg556_1: "f32[104, 104, 3, 3]", arg557_1: "f32[104]", arg558_1: "f32[104]", arg559_1: "f32[104]", arg560_1: "f32[104]", arg561_1: "f32[104, 104, 3, 3]", arg562_1: "f32[104]", arg563_1: "f32[104]", arg564_1: "f32[104]", arg565_1: "f32[104]", arg566_1: "f32[1024, 416, 1, 1]", arg567_1: "f32[1024]", arg568_1: "f32[1024]", arg569_1: "f32[1024]", arg570_1: "f32[1024]", arg571_1: "f32[416, 1024, 1, 1]", arg572_1: "f32[416]", arg573_1: "f32[416]", arg574_1: "f32[416]", arg575_1: "f32[416]", arg576_1: "f32[104, 104, 3, 3]", arg577_1: "f32[104]", arg578_1: "f32[104]", arg579_1: "f32[104]", arg580_1: "f32[104]", arg581_1: "f32[104, 104, 3, 3]", arg582_1: "f32[104]", arg583_1: "f32[104]", arg584_1: "f32[104]", arg585_1: "f32[104]", arg586_1: "f32[104, 104, 3, 3]", arg587_1: "f32[104]", arg588_1: "f32[104]", arg589_1: "f32[104]", arg590_1: "f32[104]", arg591_1: "f32[1024, 416, 1, 1]", arg592_1: "f32[1024]", arg593_1: "f32[1024]", arg594_1: "f32[1024]", arg595_1: "f32[1024]", arg596_1: "f32[416, 1024, 1, 1]", arg597_1: "f32[416]", arg598_1: "f32[416]", arg599_1: "f32[416]", arg600_1: "f32[416]", arg601_1: "f32[104, 104, 3, 3]", arg602_1: "f32[104]", arg603_1: "f32[104]", arg604_1: "f32[104]", arg605_1: "f32[104]", arg606_1: "f32[104, 104, 3, 3]", arg607_1: "f32[104]", arg608_1: "f32[104]", arg609_1: "f32[104]", arg610_1: "f32[104]", arg611_1: "f32[104, 104, 3, 3]", arg612_1: "f32[104]", arg613_1: "f32[104]", arg614_1: "f32[104]", arg615_1: "f32[104]", arg616_1: "f32[1024, 416, 1, 1]", arg617_1: "f32[1024]", arg618_1: "f32[1024]", arg619_1: "f32[1024]", arg620_1: "f32[1024]", arg621_1: "f32[416, 1024, 1, 1]", arg622_1: "f32[416]", arg623_1: "f32[416]", arg624_1: "f32[416]", arg625_1: "f32[416]", arg626_1: "f32[104, 104, 3, 3]", arg627_1: "f32[104]", arg628_1: "f32[104]", arg629_1: "f32[104]", arg630_1: "f32[104]", arg631_1: "f32[104, 104, 3, 3]", arg632_1: "f32[104]", arg633_1: "f32[104]", arg634_1: "f32[104]", arg635_1: "f32[104]", arg636_1: "f32[104, 104, 3, 3]", arg637_1: "f32[104]", arg638_1: "f32[104]", arg639_1: "f32[104]", arg640_1: "f32[104]", arg641_1: "f32[1024, 416, 1, 1]", arg642_1: "f32[1024]", arg643_1: "f32[1024]", arg644_1: "f32[1024]", arg645_1: "f32[1024]", arg646_1: "f32[416, 1024, 1, 1]", arg647_1: "f32[416]", arg648_1: "f32[416]", arg649_1: "f32[416]", arg650_1: "f32[416]", arg651_1: "f32[104, 104, 3, 3]", arg652_1: "f32[104]", arg653_1: "f32[104]", arg654_1: "f32[104]", arg655_1: "f32[104]", arg656_1: "f32[104, 104, 3, 3]", arg657_1: "f32[104]", arg658_1: "f32[104]", arg659_1: "f32[104]", arg660_1: "f32[104]", arg661_1: "f32[104, 104, 3, 3]", arg662_1: "f32[104]", arg663_1: "f32[104]", arg664_1: "f32[104]", arg665_1: "f32[104]", arg666_1: "f32[1024, 416, 1, 1]", arg667_1: "f32[1024]", arg668_1: "f32[1024]", arg669_1: "f32[1024]", arg670_1: "f32[1024]", arg671_1: "f32[416, 1024, 1, 1]", arg672_1: "f32[416]", arg673_1: "f32[416]", arg674_1: "f32[416]", arg675_1: "f32[416]", arg676_1: "f32[104, 104, 3, 3]", arg677_1: "f32[104]", arg678_1: "f32[104]", arg679_1: "f32[104]", arg680_1: "f32[104]", arg681_1: "f32[104, 104, 3, 3]", arg682_1: "f32[104]", arg683_1: "f32[104]", arg684_1: "f32[104]", arg685_1: "f32[104]", arg686_1: "f32[104, 104, 3, 3]", arg687_1: "f32[104]", arg688_1: "f32[104]", arg689_1: "f32[104]", arg690_1: "f32[104]", arg691_1: "f32[1024, 416, 1, 1]", arg692_1: "f32[1024]", arg693_1: "f32[1024]", arg694_1: "f32[1024]", arg695_1: "f32[1024]", arg696_1: "f32[416, 1024, 1, 1]", arg697_1: "f32[416]", arg698_1: "f32[416]", arg699_1: "f32[416]", arg700_1: "f32[416]", arg701_1: "f32[104, 104, 3, 3]", arg702_1: "f32[104]", arg703_1: "f32[104]", arg704_1: "f32[104]", arg705_1: "f32[104]", arg706_1: "f32[104, 104, 3, 3]", arg707_1: "f32[104]", arg708_1: "f32[104]", arg709_1: "f32[104]", arg710_1: "f32[104]", arg711_1: "f32[104, 104, 3, 3]", arg712_1: "f32[104]", arg713_1: "f32[104]", arg714_1: "f32[104]", arg715_1: "f32[104]", arg716_1: "f32[1024, 416, 1, 1]", arg717_1: "f32[1024]", arg718_1: "f32[1024]", arg719_1: "f32[1024]", arg720_1: "f32[1024]", arg721_1: "f32[416, 1024, 1, 1]", arg722_1: "f32[416]", arg723_1: "f32[416]", arg724_1: "f32[416]", arg725_1: "f32[416]", arg726_1: "f32[104, 104, 3, 3]", arg727_1: "f32[104]", arg728_1: "f32[104]", arg729_1: "f32[104]", arg730_1: "f32[104]", arg731_1: "f32[104, 104, 3, 3]", arg732_1: "f32[104]", arg733_1: "f32[104]", arg734_1: "f32[104]", arg735_1: "f32[104]", arg736_1: "f32[104, 104, 3, 3]", arg737_1: "f32[104]", arg738_1: "f32[104]", arg739_1: "f32[104]", arg740_1: "f32[104]", arg741_1: "f32[1024, 416, 1, 1]", arg742_1: "f32[1024]", arg743_1: "f32[1024]", arg744_1: "f32[1024]", arg745_1: "f32[1024]", arg746_1: "f32[416, 1024, 1, 1]", arg747_1: "f32[416]", arg748_1: "f32[416]", arg749_1: "f32[416]", arg750_1: "f32[416]", arg751_1: "f32[104, 104, 3, 3]", arg752_1: "f32[104]", arg753_1: "f32[104]", arg754_1: "f32[104]", arg755_1: "f32[104]", arg756_1: "f32[104, 104, 3, 3]", arg757_1: "f32[104]", arg758_1: "f32[104]", arg759_1: "f32[104]", arg760_1: "f32[104]", arg761_1: "f32[104, 104, 3, 3]", arg762_1: "f32[104]", arg763_1: "f32[104]", arg764_1: "f32[104]", arg765_1: "f32[104]", arg766_1: "f32[1024, 416, 1, 1]", arg767_1: "f32[1024]", arg768_1: "f32[1024]", arg769_1: "f32[1024]", arg770_1: "f32[1024]", arg771_1: "f32[832, 1024, 1, 1]", arg772_1: "f32[832]", arg773_1: "f32[832]", arg774_1: "f32[832]", arg775_1: "f32[832]", arg776_1: "f32[208, 208, 3, 3]", arg777_1: "f32[208]", arg778_1: "f32[208]", arg779_1: "f32[208]", arg780_1: "f32[208]", arg781_1: "f32[208, 208, 3, 3]", arg782_1: "f32[208]", arg783_1: "f32[208]", arg784_1: "f32[208]", arg785_1: "f32[208]", arg786_1: "f32[208, 208, 3, 3]", arg787_1: "f32[208]", arg788_1: "f32[208]", arg789_1: "f32[208]", arg790_1: "f32[208]", arg791_1: "f32[2048, 832, 1, 1]", arg792_1: "f32[2048]", arg793_1: "f32[2048]", arg794_1: "f32[2048]", arg795_1: "f32[2048]", arg796_1: "f32[2048, 1024, 1, 1]", arg797_1: "f32[2048]", arg798_1: "f32[2048]", arg799_1: "f32[2048]", arg800_1: "f32[2048]", arg801_1: "f32[832, 2048, 1, 1]", arg802_1: "f32[832]", arg803_1: "f32[832]", arg804_1: "f32[832]", arg805_1: "f32[832]", arg806_1: "f32[208, 208, 3, 3]", arg807_1: "f32[208]", arg808_1: "f32[208]", arg809_1: "f32[208]", arg810_1: "f32[208]", arg811_1: "f32[208, 208, 3, 3]", arg812_1: "f32[208]", arg813_1: "f32[208]", arg814_1: "f32[208]", arg815_1: "f32[208]", arg816_1: "f32[208, 208, 3, 3]", arg817_1: "f32[208]", arg818_1: "f32[208]", arg819_1: "f32[208]", arg820_1: "f32[208]", arg821_1: "f32[2048, 832, 1, 1]", arg822_1: "f32[2048]", arg823_1: "f32[2048]", arg824_1: "f32[2048]", arg825_1: "f32[2048]", arg826_1: "f32[832, 2048, 1, 1]", arg827_1: "f32[832]", arg828_1: "f32[832]", arg829_1: "f32[832]", arg830_1: "f32[832]", arg831_1: "f32[208, 208, 3, 3]", arg832_1: "f32[208]", arg833_1: "f32[208]", arg834_1: "f32[208]", arg835_1: "f32[208]", arg836_1: "f32[208, 208, 3, 3]", arg837_1: "f32[208]", arg838_1: "f32[208]", arg839_1: "f32[208]", arg840_1: "f32[208]", arg841_1: "f32[208, 208, 3, 3]", arg842_1: "f32[208]", arg843_1: "f32[208]", arg844_1: "f32[208]", arg845_1: "f32[208]", arg846_1: "f32[2048, 832, 1, 1]", arg847_1: "f32[2048]", arg848_1: "f32[2048]", arg849_1: "f32[2048]", arg850_1: "f32[2048]", arg851_1: "f32[1000, 2048]", arg852_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:615 in forward_features, code: x = self.conv1(x)
        convolution_170: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:616 in forward_features, code: x = self.bn1(x)
        unsqueeze_1360: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_1361: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, -1);  unsqueeze_1360 = None
        sub_170: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_170, unsqueeze_1361);  convolution_170 = unsqueeze_1361 = None
        add_431: "f32[64]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_170: "f32[64]" = torch.ops.aten.sqrt.default(add_431);  add_431 = None
        reciprocal_170: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_170);  sqrt_170 = None
        mul_510: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_170, 1);  reciprocal_170 = None
        unsqueeze_1362: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_510, -1);  mul_510 = None
        unsqueeze_1363: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, -1);  unsqueeze_1362 = None
        mul_511: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_1363);  sub_170 = unsqueeze_1363 = None
        unsqueeze_1364: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_1365: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, -1);  unsqueeze_1364 = None
        mul_512: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_511, unsqueeze_1365);  mul_511 = unsqueeze_1365 = None
        unsqueeze_1366: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_1367: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, -1);  unsqueeze_1366 = None
        add_432: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_512, unsqueeze_1367);  mul_512 = unsqueeze_1367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:617 in forward_features, code: x = self.act1(x)
        relu_166: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_432);  add_432 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:618 in forward_features, code: x = self.maxpool(x)
        _low_memory_max_pool2d_with_offsets_1 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_166, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu_166 = None
        getitem_662: "f32[8, 64, 56, 56]" = _low_memory_max_pool2d_with_offsets_1[0];  _low_memory_max_pool2d_with_offsets_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_171: "f32[8, 104, 56, 56]" = torch.ops.aten.convolution.default(getitem_662, arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1368: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_1369: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, -1);  unsqueeze_1368 = None
        sub_171: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_171, unsqueeze_1369);  convolution_171 = unsqueeze_1369 = None
        add_433: "f32[104]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_171: "f32[104]" = torch.ops.aten.sqrt.default(add_433);  add_433 = None
        reciprocal_171: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_171);  sqrt_171 = None
        mul_513: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_171, 1);  reciprocal_171 = None
        unsqueeze_1370: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_513, -1);  mul_513 = None
        unsqueeze_1371: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, -1);  unsqueeze_1370 = None
        mul_514: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_1371);  sub_171 = unsqueeze_1371 = None
        unsqueeze_1372: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_1373: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, -1);  unsqueeze_1372 = None
        mul_515: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(mul_514, unsqueeze_1373);  mul_514 = unsqueeze_1373 = None
        unsqueeze_1374: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_1375: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, -1);  unsqueeze_1374 = None
        add_434: "f32[8, 104, 56, 56]" = torch.ops.aten.add.Tensor(mul_515, unsqueeze_1375);  mul_515 = unsqueeze_1375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_167: "f32[8, 104, 56, 56]" = torch.ops.aten.relu.default(add_434);  add_434 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_166 = torch.ops.aten.split.Tensor(relu_167, 26, 1)
        getitem_668: "f32[8, 26, 56, 56]" = split_166[0];  split_166 = None
        split_167 = torch.ops.aten.split.Tensor(relu_167, 26, 1)
        getitem_673: "f32[8, 26, 56, 56]" = split_167[1];  split_167 = None
        split_168 = torch.ops.aten.split.Tensor(relu_167, 26, 1)
        getitem_678: "f32[8, 26, 56, 56]" = split_168[2];  split_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:99 in forward, code: spo.append(self.pool(spx[-1]))
        split_169 = torch.ops.aten.split.Tensor(relu_167, 26, 1);  relu_167 = None
        getitem_683: "f32[8, 26, 56, 56]" = split_169[3];  split_169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_172: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_668, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_668 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1376: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_1377: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, -1);  unsqueeze_1376 = None
        sub_172: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_172, unsqueeze_1377);  convolution_172 = unsqueeze_1377 = None
        add_435: "f32[26]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_172: "f32[26]" = torch.ops.aten.sqrt.default(add_435);  add_435 = None
        reciprocal_172: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_172);  sqrt_172 = None
        mul_516: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_172, 1);  reciprocal_172 = None
        unsqueeze_1378: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_516, -1);  mul_516 = None
        unsqueeze_1379: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, -1);  unsqueeze_1378 = None
        mul_517: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_1379);  sub_172 = unsqueeze_1379 = None
        unsqueeze_1380: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_1381: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, -1);  unsqueeze_1380 = None
        mul_518: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_517, unsqueeze_1381);  mul_517 = unsqueeze_1381 = None
        unsqueeze_1382: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_1383: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, -1);  unsqueeze_1382 = None
        add_436: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_518, unsqueeze_1383);  mul_518 = unsqueeze_1383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_168: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_436);  add_436 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_173: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_673, arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_673 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1384: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_1385: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, -1);  unsqueeze_1384 = None
        sub_173: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_173, unsqueeze_1385);  convolution_173 = unsqueeze_1385 = None
        add_437: "f32[26]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_173: "f32[26]" = torch.ops.aten.sqrt.default(add_437);  add_437 = None
        reciprocal_173: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_173);  sqrt_173 = None
        mul_519: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_173, 1);  reciprocal_173 = None
        unsqueeze_1386: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_519, -1);  mul_519 = None
        unsqueeze_1387: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, -1);  unsqueeze_1386 = None
        mul_520: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_1387);  sub_173 = unsqueeze_1387 = None
        unsqueeze_1388: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_1389: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, -1);  unsqueeze_1388 = None
        mul_521: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_520, unsqueeze_1389);  mul_520 = unsqueeze_1389 = None
        unsqueeze_1390: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_1391: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, -1);  unsqueeze_1390 = None
        add_438: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_521, unsqueeze_1391);  mul_521 = unsqueeze_1391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_169: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_438);  add_438 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_174: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_678, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_678 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1392: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_1393: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1392, -1);  unsqueeze_1392 = None
        sub_174: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_174, unsqueeze_1393);  convolution_174 = unsqueeze_1393 = None
        add_439: "f32[26]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_174: "f32[26]" = torch.ops.aten.sqrt.default(add_439);  add_439 = None
        reciprocal_174: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_174);  sqrt_174 = None
        mul_522: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_174, 1);  reciprocal_174 = None
        unsqueeze_1394: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_522, -1);  mul_522 = None
        unsqueeze_1395: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, -1);  unsqueeze_1394 = None
        mul_523: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_1395);  sub_174 = unsqueeze_1395 = None
        unsqueeze_1396: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_1397: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1396, -1);  unsqueeze_1396 = None
        mul_524: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_523, unsqueeze_1397);  mul_523 = unsqueeze_1397 = None
        unsqueeze_1398: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_1399: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, -1);  unsqueeze_1398 = None
        add_440: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_524, unsqueeze_1399);  mul_524 = unsqueeze_1399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_170: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_440);  add_440 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:99 in forward, code: spo.append(self.pool(spx[-1]))
        avg_pool2d_4: "f32[8, 26, 56, 56]" = torch.ops.aten.avg_pool2d.default(getitem_683, [3, 3], [1, 1], [1, 1]);  getitem_683 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_33: "f32[8, 104, 56, 56]" = torch.ops.aten.cat.default([relu_168, relu_169, relu_170, avg_pool2d_4], 1);  relu_168 = relu_169 = relu_170 = avg_pool2d_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_175: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_33, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_33 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1400: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_1401: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, -1);  unsqueeze_1400 = None
        sub_175: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_175, unsqueeze_1401);  convolution_175 = unsqueeze_1401 = None
        add_441: "f32[256]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_175: "f32[256]" = torch.ops.aten.sqrt.default(add_441);  add_441 = None
        reciprocal_175: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_175);  sqrt_175 = None
        mul_525: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_175, 1);  reciprocal_175 = None
        unsqueeze_1402: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_525, -1);  mul_525 = None
        unsqueeze_1403: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1402, -1);  unsqueeze_1402 = None
        mul_526: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_1403);  sub_175 = unsqueeze_1403 = None
        unsqueeze_1404: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_1405: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1404, -1);  unsqueeze_1404 = None
        mul_527: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_526, unsqueeze_1405);  mul_526 = unsqueeze_1405 = None
        unsqueeze_1406: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_1407: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, -1);  unsqueeze_1406 = None
        add_442: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_527, unsqueeze_1407);  mul_527 = unsqueeze_1407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:111 in forward, code: shortcut = self.downsample(x)
        convolution_176: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_662, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_662 = arg31_1 = None
        unsqueeze_1408: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_1409: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1408, -1);  unsqueeze_1408 = None
        sub_176: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_176, unsqueeze_1409);  convolution_176 = unsqueeze_1409 = None
        add_443: "f32[256]" = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_176: "f32[256]" = torch.ops.aten.sqrt.default(add_443);  add_443 = None
        reciprocal_176: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_176);  sqrt_176 = None
        mul_528: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_176, 1);  reciprocal_176 = None
        unsqueeze_1410: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_528, -1);  mul_528 = None
        unsqueeze_1411: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, -1);  unsqueeze_1410 = None
        mul_529: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_1411);  sub_176 = unsqueeze_1411 = None
        unsqueeze_1412: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_1413: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, -1);  unsqueeze_1412 = None
        mul_530: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_529, unsqueeze_1413);  mul_529 = unsqueeze_1413 = None
        unsqueeze_1414: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_1415: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1414, -1);  unsqueeze_1414 = None
        add_444: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_530, unsqueeze_1415);  mul_530 = unsqueeze_1415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_445: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_442, add_444);  add_442 = add_444 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_171: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_445);  add_445 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_177: "f32[8, 104, 56, 56]" = torch.ops.aten.convolution.default(relu_171, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1416: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_1417: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1416, -1);  unsqueeze_1416 = None
        sub_177: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_177, unsqueeze_1417);  convolution_177 = unsqueeze_1417 = None
        add_446: "f32[104]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_177: "f32[104]" = torch.ops.aten.sqrt.default(add_446);  add_446 = None
        reciprocal_177: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_177);  sqrt_177 = None
        mul_531: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_177, 1);  reciprocal_177 = None
        unsqueeze_1418: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_531, -1);  mul_531 = None
        unsqueeze_1419: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, -1);  unsqueeze_1418 = None
        mul_532: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_1419);  sub_177 = unsqueeze_1419 = None
        unsqueeze_1420: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_1421: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1420, -1);  unsqueeze_1420 = None
        mul_533: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_1421);  mul_532 = unsqueeze_1421 = None
        unsqueeze_1422: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_1423: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, -1);  unsqueeze_1422 = None
        add_447: "f32[8, 104, 56, 56]" = torch.ops.aten.add.Tensor(mul_533, unsqueeze_1423);  mul_533 = unsqueeze_1423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_172: "f32[8, 104, 56, 56]" = torch.ops.aten.relu.default(add_447);  add_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_171 = torch.ops.aten.split.Tensor(relu_172, 26, 1)
        getitem_688: "f32[8, 26, 56, 56]" = split_171[0];  split_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_172 = torch.ops.aten.split.Tensor(relu_172, 26, 1)
        getitem_693: "f32[8, 26, 56, 56]" = split_172[1];  split_172 = None
        split_173 = torch.ops.aten.split.Tensor(relu_172, 26, 1)
        getitem_698: "f32[8, 26, 56, 56]" = split_173[2];  split_173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_174 = torch.ops.aten.split.Tensor(relu_172, 26, 1);  relu_172 = None
        getitem_703: "f32[8, 26, 56, 56]" = split_174[3];  split_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_178: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_688, arg41_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_688 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1424: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_1425: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, -1);  unsqueeze_1424 = None
        sub_178: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_178, unsqueeze_1425);  convolution_178 = unsqueeze_1425 = None
        add_448: "f32[26]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_178: "f32[26]" = torch.ops.aten.sqrt.default(add_448);  add_448 = None
        reciprocal_178: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_178);  sqrt_178 = None
        mul_534: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_178, 1);  reciprocal_178 = None
        unsqueeze_1426: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_534, -1);  mul_534 = None
        unsqueeze_1427: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1426, -1);  unsqueeze_1426 = None
        mul_535: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_1427);  sub_178 = unsqueeze_1427 = None
        unsqueeze_1428: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_1429: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1428, -1);  unsqueeze_1428 = None
        mul_536: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_535, unsqueeze_1429);  mul_535 = unsqueeze_1429 = None
        unsqueeze_1430: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_1431: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1430, -1);  unsqueeze_1430 = None
        add_449: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_536, unsqueeze_1431);  mul_536 = unsqueeze_1431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_173: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_449);  add_449 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_450: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(relu_173, getitem_693);  getitem_693 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_179: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(add_450, arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_450 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1432: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_1433: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, -1);  unsqueeze_1432 = None
        sub_179: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_179, unsqueeze_1433);  convolution_179 = unsqueeze_1433 = None
        add_451: "f32[26]" = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_179: "f32[26]" = torch.ops.aten.sqrt.default(add_451);  add_451 = None
        reciprocal_179: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_179);  sqrt_179 = None
        mul_537: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_179, 1);  reciprocal_179 = None
        unsqueeze_1434: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_537, -1);  mul_537 = None
        unsqueeze_1435: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, -1);  unsqueeze_1434 = None
        mul_538: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_1435);  sub_179 = unsqueeze_1435 = None
        unsqueeze_1436: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_1437: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1436, -1);  unsqueeze_1436 = None
        mul_539: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_538, unsqueeze_1437);  mul_538 = unsqueeze_1437 = None
        unsqueeze_1438: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_1439: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, -1);  unsqueeze_1438 = None
        add_452: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_539, unsqueeze_1439);  mul_539 = unsqueeze_1439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_174: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_452);  add_452 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_453: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(relu_174, getitem_698);  getitem_698 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_180: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(add_453, arg51_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_453 = arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1440: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_1441: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1440, -1);  unsqueeze_1440 = None
        sub_180: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_180, unsqueeze_1441);  convolution_180 = unsqueeze_1441 = None
        add_454: "f32[26]" = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_180: "f32[26]" = torch.ops.aten.sqrt.default(add_454);  add_454 = None
        reciprocal_180: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_180);  sqrt_180 = None
        mul_540: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_180, 1);  reciprocal_180 = None
        unsqueeze_1442: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_540, -1);  mul_540 = None
        unsqueeze_1443: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1442, -1);  unsqueeze_1442 = None
        mul_541: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_1443);  sub_180 = unsqueeze_1443 = None
        unsqueeze_1444: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_1445: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, -1);  unsqueeze_1444 = None
        mul_542: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_541, unsqueeze_1445);  mul_541 = unsqueeze_1445 = None
        unsqueeze_1446: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_1447: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, -1);  unsqueeze_1446 = None
        add_455: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_542, unsqueeze_1447);  mul_542 = unsqueeze_1447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_175: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_455);  add_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_34: "f32[8, 104, 56, 56]" = torch.ops.aten.cat.default([relu_173, relu_174, relu_175, getitem_703], 1);  relu_173 = relu_174 = relu_175 = getitem_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_181: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_34, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_34 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1448: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_1449: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, -1);  unsqueeze_1448 = None
        sub_181: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_181, unsqueeze_1449);  convolution_181 = unsqueeze_1449 = None
        add_456: "f32[256]" = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_181: "f32[256]" = torch.ops.aten.sqrt.default(add_456);  add_456 = None
        reciprocal_181: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_181);  sqrt_181 = None
        mul_543: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_181, 1);  reciprocal_181 = None
        unsqueeze_1450: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_543, -1);  mul_543 = None
        unsqueeze_1451: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, -1);  unsqueeze_1450 = None
        mul_544: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_1451);  sub_181 = unsqueeze_1451 = None
        unsqueeze_1452: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_1453: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1452, -1);  unsqueeze_1452 = None
        mul_545: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_544, unsqueeze_1453);  mul_544 = unsqueeze_1453 = None
        unsqueeze_1454: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_1455: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, -1);  unsqueeze_1454 = None
        add_457: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_1455);  mul_545 = unsqueeze_1455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_458: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_457, relu_171);  add_457 = relu_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_176: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_458);  add_458 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_182: "f32[8, 104, 56, 56]" = torch.ops.aten.convolution.default(relu_176, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1456: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_1457: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, -1);  unsqueeze_1456 = None
        sub_182: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_182, unsqueeze_1457);  convolution_182 = unsqueeze_1457 = None
        add_459: "f32[104]" = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_182: "f32[104]" = torch.ops.aten.sqrt.default(add_459);  add_459 = None
        reciprocal_182: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_182);  sqrt_182 = None
        mul_546: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_182, 1);  reciprocal_182 = None
        unsqueeze_1458: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_546, -1);  mul_546 = None
        unsqueeze_1459: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, -1);  unsqueeze_1458 = None
        mul_547: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_1459);  sub_182 = unsqueeze_1459 = None
        unsqueeze_1460: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_1461: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, -1);  unsqueeze_1460 = None
        mul_548: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(mul_547, unsqueeze_1461);  mul_547 = unsqueeze_1461 = None
        unsqueeze_1462: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_1463: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, -1);  unsqueeze_1462 = None
        add_460: "f32[8, 104, 56, 56]" = torch.ops.aten.add.Tensor(mul_548, unsqueeze_1463);  mul_548 = unsqueeze_1463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_177: "f32[8, 104, 56, 56]" = torch.ops.aten.relu.default(add_460);  add_460 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_176 = torch.ops.aten.split.Tensor(relu_177, 26, 1)
        getitem_708: "f32[8, 26, 56, 56]" = split_176[0];  split_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_177 = torch.ops.aten.split.Tensor(relu_177, 26, 1)
        getitem_713: "f32[8, 26, 56, 56]" = split_177[1];  split_177 = None
        split_178 = torch.ops.aten.split.Tensor(relu_177, 26, 1)
        getitem_718: "f32[8, 26, 56, 56]" = split_178[2];  split_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_179 = torch.ops.aten.split.Tensor(relu_177, 26, 1);  relu_177 = None
        getitem_723: "f32[8, 26, 56, 56]" = split_179[3];  split_179 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_183: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_708, arg66_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_708 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1464: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_1465: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1464, -1);  unsqueeze_1464 = None
        sub_183: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_183, unsqueeze_1465);  convolution_183 = unsqueeze_1465 = None
        add_461: "f32[26]" = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_183: "f32[26]" = torch.ops.aten.sqrt.default(add_461);  add_461 = None
        reciprocal_183: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_183);  sqrt_183 = None
        mul_549: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_183, 1);  reciprocal_183 = None
        unsqueeze_1466: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_549, -1);  mul_549 = None
        unsqueeze_1467: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, -1);  unsqueeze_1466 = None
        mul_550: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_1467);  sub_183 = unsqueeze_1467 = None
        unsqueeze_1468: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_1469: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, -1);  unsqueeze_1468 = None
        mul_551: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_550, unsqueeze_1469);  mul_550 = unsqueeze_1469 = None
        unsqueeze_1470: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_1471: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, -1);  unsqueeze_1470 = None
        add_462: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_551, unsqueeze_1471);  mul_551 = unsqueeze_1471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_178: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_462);  add_462 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_463: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(relu_178, getitem_713);  getitem_713 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_184: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(add_463, arg71_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_463 = arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1472: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_1473: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, -1);  unsqueeze_1472 = None
        sub_184: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_184, unsqueeze_1473);  convolution_184 = unsqueeze_1473 = None
        add_464: "f32[26]" = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_184: "f32[26]" = torch.ops.aten.sqrt.default(add_464);  add_464 = None
        reciprocal_184: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_184);  sqrt_184 = None
        mul_552: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_184, 1);  reciprocal_184 = None
        unsqueeze_1474: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_552, -1);  mul_552 = None
        unsqueeze_1475: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, -1);  unsqueeze_1474 = None
        mul_553: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_1475);  sub_184 = unsqueeze_1475 = None
        unsqueeze_1476: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_1477: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1476, -1);  unsqueeze_1476 = None
        mul_554: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_1477);  mul_553 = unsqueeze_1477 = None
        unsqueeze_1478: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_1479: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, -1);  unsqueeze_1478 = None
        add_465: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_554, unsqueeze_1479);  mul_554 = unsqueeze_1479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_179: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_465);  add_465 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_466: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(relu_179, getitem_718);  getitem_718 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_185: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(add_466, arg76_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_466 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1480: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_1481: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1480, -1);  unsqueeze_1480 = None
        sub_185: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_185, unsqueeze_1481);  convolution_185 = unsqueeze_1481 = None
        add_467: "f32[26]" = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_185: "f32[26]" = torch.ops.aten.sqrt.default(add_467);  add_467 = None
        reciprocal_185: "f32[26]" = torch.ops.aten.reciprocal.default(sqrt_185);  sqrt_185 = None
        mul_555: "f32[26]" = torch.ops.aten.mul.Tensor(reciprocal_185, 1);  reciprocal_185 = None
        unsqueeze_1482: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(mul_555, -1);  mul_555 = None
        unsqueeze_1483: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, -1);  unsqueeze_1482 = None
        mul_556: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_1483);  sub_185 = unsqueeze_1483 = None
        unsqueeze_1484: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_1485: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, -1);  unsqueeze_1484 = None
        mul_557: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_556, unsqueeze_1485);  mul_556 = unsqueeze_1485 = None
        unsqueeze_1486: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_1487: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1486, -1);  unsqueeze_1486 = None
        add_468: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_557, unsqueeze_1487);  mul_557 = unsqueeze_1487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_180: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_468);  add_468 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_35: "f32[8, 104, 56, 56]" = torch.ops.aten.cat.default([relu_178, relu_179, relu_180, getitem_723], 1);  relu_178 = relu_179 = relu_180 = getitem_723 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_186: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_35, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_35 = arg81_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1488: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_1489: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1488, -1);  unsqueeze_1488 = None
        sub_186: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_186, unsqueeze_1489);  convolution_186 = unsqueeze_1489 = None
        add_469: "f32[256]" = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_186: "f32[256]" = torch.ops.aten.sqrt.default(add_469);  add_469 = None
        reciprocal_186: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_186);  sqrt_186 = None
        mul_558: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_186, 1);  reciprocal_186 = None
        unsqueeze_1490: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_558, -1);  mul_558 = None
        unsqueeze_1491: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, -1);  unsqueeze_1490 = None
        mul_559: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_1491);  sub_186 = unsqueeze_1491 = None
        unsqueeze_1492: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_1493: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1492, -1);  unsqueeze_1492 = None
        mul_560: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_559, unsqueeze_1493);  mul_559 = unsqueeze_1493 = None
        unsqueeze_1494: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_1495: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, -1);  unsqueeze_1494 = None
        add_470: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_560, unsqueeze_1495);  mul_560 = unsqueeze_1495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_471: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_470, relu_176);  add_470 = relu_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_181: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_471);  add_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_187: "f32[8, 208, 56, 56]" = torch.ops.aten.convolution.default(relu_181, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1496: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_1497: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1496, -1);  unsqueeze_1496 = None
        sub_187: "f32[8, 208, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_187, unsqueeze_1497);  convolution_187 = unsqueeze_1497 = None
        add_472: "f32[208]" = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_187: "f32[208]" = torch.ops.aten.sqrt.default(add_472);  add_472 = None
        reciprocal_187: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_187);  sqrt_187 = None
        mul_561: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_187, 1);  reciprocal_187 = None
        unsqueeze_1498: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_561, -1);  mul_561 = None
        unsqueeze_1499: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, -1);  unsqueeze_1498 = None
        mul_562: "f32[8, 208, 56, 56]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_1499);  sub_187 = unsqueeze_1499 = None
        unsqueeze_1500: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_1501: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1500, -1);  unsqueeze_1500 = None
        mul_563: "f32[8, 208, 56, 56]" = torch.ops.aten.mul.Tensor(mul_562, unsqueeze_1501);  mul_562 = unsqueeze_1501 = None
        unsqueeze_1502: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_1503: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1502, -1);  unsqueeze_1502 = None
        add_473: "f32[8, 208, 56, 56]" = torch.ops.aten.add.Tensor(mul_563, unsqueeze_1503);  mul_563 = unsqueeze_1503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_182: "f32[8, 208, 56, 56]" = torch.ops.aten.relu.default(add_473);  add_473 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_181 = torch.ops.aten.split.Tensor(relu_182, 52, 1)
        getitem_728: "f32[8, 52, 56, 56]" = split_181[0];  split_181 = None
        split_182 = torch.ops.aten.split.Tensor(relu_182, 52, 1)
        getitem_733: "f32[8, 52, 56, 56]" = split_182[1];  split_182 = None
        split_183 = torch.ops.aten.split.Tensor(relu_182, 52, 1)
        getitem_738: "f32[8, 52, 56, 56]" = split_183[2];  split_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:99 in forward, code: spo.append(self.pool(spx[-1]))
        split_184 = torch.ops.aten.split.Tensor(relu_182, 52, 1);  relu_182 = None
        getitem_743: "f32[8, 52, 56, 56]" = split_184[3];  split_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_188: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_728, arg91_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_728 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1504: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_1505: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1504, -1);  unsqueeze_1504 = None
        sub_188: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_188, unsqueeze_1505);  convolution_188 = unsqueeze_1505 = None
        add_474: "f32[52]" = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_188: "f32[52]" = torch.ops.aten.sqrt.default(add_474);  add_474 = None
        reciprocal_188: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_188);  sqrt_188 = None
        mul_564: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_188, 1);  reciprocal_188 = None
        unsqueeze_1506: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_564, -1);  mul_564 = None
        unsqueeze_1507: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, -1);  unsqueeze_1506 = None
        mul_565: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_1507);  sub_188 = unsqueeze_1507 = None
        unsqueeze_1508: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_1509: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1508, -1);  unsqueeze_1508 = None
        mul_566: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_565, unsqueeze_1509);  mul_565 = unsqueeze_1509 = None
        unsqueeze_1510: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_1511: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1510, -1);  unsqueeze_1510 = None
        add_475: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_1511);  mul_566 = unsqueeze_1511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_183: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_475);  add_475 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_189: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_733, arg96_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_733 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1512: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_1513: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1512, -1);  unsqueeze_1512 = None
        sub_189: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_189, unsqueeze_1513);  convolution_189 = unsqueeze_1513 = None
        add_476: "f32[52]" = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_189: "f32[52]" = torch.ops.aten.sqrt.default(add_476);  add_476 = None
        reciprocal_189: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_189);  sqrt_189 = None
        mul_567: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_189, 1);  reciprocal_189 = None
        unsqueeze_1514: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_567, -1);  mul_567 = None
        unsqueeze_1515: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1514, -1);  unsqueeze_1514 = None
        mul_568: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_1515);  sub_189 = unsqueeze_1515 = None
        unsqueeze_1516: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_1517: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1516, -1);  unsqueeze_1516 = None
        mul_569: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_568, unsqueeze_1517);  mul_568 = unsqueeze_1517 = None
        unsqueeze_1518: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_1519: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, -1);  unsqueeze_1518 = None
        add_477: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_569, unsqueeze_1519);  mul_569 = unsqueeze_1519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_184: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_477);  add_477 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_190: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_738, arg101_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_738 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1520: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_1521: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1520, -1);  unsqueeze_1520 = None
        sub_190: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_190, unsqueeze_1521);  convolution_190 = unsqueeze_1521 = None
        add_478: "f32[52]" = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_190: "f32[52]" = torch.ops.aten.sqrt.default(add_478);  add_478 = None
        reciprocal_190: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_190);  sqrt_190 = None
        mul_570: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_190, 1);  reciprocal_190 = None
        unsqueeze_1522: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_570, -1);  mul_570 = None
        unsqueeze_1523: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1522, -1);  unsqueeze_1522 = None
        mul_571: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_1523);  sub_190 = unsqueeze_1523 = None
        unsqueeze_1524: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_1525: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1524, -1);  unsqueeze_1524 = None
        mul_572: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_571, unsqueeze_1525);  mul_571 = unsqueeze_1525 = None
        unsqueeze_1526: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_1527: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1526, -1);  unsqueeze_1526 = None
        add_479: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_572, unsqueeze_1527);  mul_572 = unsqueeze_1527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_185: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_479);  add_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:99 in forward, code: spo.append(self.pool(spx[-1]))
        avg_pool2d_5: "f32[8, 52, 28, 28]" = torch.ops.aten.avg_pool2d.default(getitem_743, [3, 3], [2, 2], [1, 1]);  getitem_743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_36: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([relu_183, relu_184, relu_185, avg_pool2d_5], 1);  relu_183 = relu_184 = relu_185 = avg_pool2d_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_191: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_36, arg106_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_36 = arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1528: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_1529: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1528, -1);  unsqueeze_1528 = None
        sub_191: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_191, unsqueeze_1529);  convolution_191 = unsqueeze_1529 = None
        add_480: "f32[512]" = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_191: "f32[512]" = torch.ops.aten.sqrt.default(add_480);  add_480 = None
        reciprocal_191: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_191);  sqrt_191 = None
        mul_573: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_191, 1);  reciprocal_191 = None
        unsqueeze_1530: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_573, -1);  mul_573 = None
        unsqueeze_1531: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, -1);  unsqueeze_1530 = None
        mul_574: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_1531);  sub_191 = unsqueeze_1531 = None
        unsqueeze_1532: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_1533: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1532, -1);  unsqueeze_1532 = None
        mul_575: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_1533);  mul_574 = unsqueeze_1533 = None
        unsqueeze_1534: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_1535: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1534, -1);  unsqueeze_1534 = None
        add_481: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_575, unsqueeze_1535);  mul_575 = unsqueeze_1535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:111 in forward, code: shortcut = self.downsample(x)
        convolution_192: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_181, arg111_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_181 = arg111_1 = None
        unsqueeze_1536: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_1537: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1536, -1);  unsqueeze_1536 = None
        sub_192: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_192, unsqueeze_1537);  convolution_192 = unsqueeze_1537 = None
        add_482: "f32[512]" = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_192: "f32[512]" = torch.ops.aten.sqrt.default(add_482);  add_482 = None
        reciprocal_192: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_192);  sqrt_192 = None
        mul_576: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_192, 1);  reciprocal_192 = None
        unsqueeze_1538: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_576, -1);  mul_576 = None
        unsqueeze_1539: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1538, -1);  unsqueeze_1538 = None
        mul_577: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_1539);  sub_192 = unsqueeze_1539 = None
        unsqueeze_1540: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_1541: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1540, -1);  unsqueeze_1540 = None
        mul_578: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_577, unsqueeze_1541);  mul_577 = unsqueeze_1541 = None
        unsqueeze_1542: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_1543: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, -1);  unsqueeze_1542 = None
        add_483: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_578, unsqueeze_1543);  mul_578 = unsqueeze_1543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_484: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_481, add_483);  add_481 = add_483 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_186: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_484);  add_484 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_193: "f32[8, 208, 28, 28]" = torch.ops.aten.convolution.default(relu_186, arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1544: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_1545: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1544, -1);  unsqueeze_1544 = None
        sub_193: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_193, unsqueeze_1545);  convolution_193 = unsqueeze_1545 = None
        add_485: "f32[208]" = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
        sqrt_193: "f32[208]" = torch.ops.aten.sqrt.default(add_485);  add_485 = None
        reciprocal_193: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_193);  sqrt_193 = None
        mul_579: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_193, 1);  reciprocal_193 = None
        unsqueeze_1546: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_579, -1);  mul_579 = None
        unsqueeze_1547: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1546, -1);  unsqueeze_1546 = None
        mul_580: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_1547);  sub_193 = unsqueeze_1547 = None
        unsqueeze_1548: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_1549: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1548, -1);  unsqueeze_1548 = None
        mul_581: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(mul_580, unsqueeze_1549);  mul_580 = unsqueeze_1549 = None
        unsqueeze_1550: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_1551: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1550, -1);  unsqueeze_1550 = None
        add_486: "f32[8, 208, 28, 28]" = torch.ops.aten.add.Tensor(mul_581, unsqueeze_1551);  mul_581 = unsqueeze_1551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_187: "f32[8, 208, 28, 28]" = torch.ops.aten.relu.default(add_486);  add_486 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_186 = torch.ops.aten.split.Tensor(relu_187, 52, 1)
        getitem_748: "f32[8, 52, 28, 28]" = split_186[0];  split_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_187 = torch.ops.aten.split.Tensor(relu_187, 52, 1)
        getitem_753: "f32[8, 52, 28, 28]" = split_187[1];  split_187 = None
        split_188 = torch.ops.aten.split.Tensor(relu_187, 52, 1)
        getitem_758: "f32[8, 52, 28, 28]" = split_188[2];  split_188 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_189 = torch.ops.aten.split.Tensor(relu_187, 52, 1);  relu_187 = None
        getitem_763: "f32[8, 52, 28, 28]" = split_189[3];  split_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_194: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_748, arg121_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_748 = arg121_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1552: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_1553: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1552, -1);  unsqueeze_1552 = None
        sub_194: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_194, unsqueeze_1553);  convolution_194 = unsqueeze_1553 = None
        add_487: "f32[52]" = torch.ops.aten.add.Tensor(arg123_1, 1e-05);  arg123_1 = None
        sqrt_194: "f32[52]" = torch.ops.aten.sqrt.default(add_487);  add_487 = None
        reciprocal_194: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_194);  sqrt_194 = None
        mul_582: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_194, 1);  reciprocal_194 = None
        unsqueeze_1554: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_582, -1);  mul_582 = None
        unsqueeze_1555: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, -1);  unsqueeze_1554 = None
        mul_583: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_1555);  sub_194 = unsqueeze_1555 = None
        unsqueeze_1556: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_1557: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1556, -1);  unsqueeze_1556 = None
        mul_584: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_583, unsqueeze_1557);  mul_583 = unsqueeze_1557 = None
        unsqueeze_1558: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_1559: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1558, -1);  unsqueeze_1558 = None
        add_488: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_584, unsqueeze_1559);  mul_584 = unsqueeze_1559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_188: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_488);  add_488 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_489: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_188, getitem_753);  getitem_753 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_195: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_489, arg126_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_489 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1560: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_1561: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1560, -1);  unsqueeze_1560 = None
        sub_195: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_195, unsqueeze_1561);  convolution_195 = unsqueeze_1561 = None
        add_490: "f32[52]" = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_195: "f32[52]" = torch.ops.aten.sqrt.default(add_490);  add_490 = None
        reciprocal_195: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_195);  sqrt_195 = None
        mul_585: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_195, 1);  reciprocal_195 = None
        unsqueeze_1562: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_585, -1);  mul_585 = None
        unsqueeze_1563: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1562, -1);  unsqueeze_1562 = None
        mul_586: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_1563);  sub_195 = unsqueeze_1563 = None
        unsqueeze_1564: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_1565: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1564, -1);  unsqueeze_1564 = None
        mul_587: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_586, unsqueeze_1565);  mul_586 = unsqueeze_1565 = None
        unsqueeze_1566: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_1567: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, -1);  unsqueeze_1566 = None
        add_491: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_1567);  mul_587 = unsqueeze_1567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_189: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_491);  add_491 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_492: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_189, getitem_758);  getitem_758 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_196: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_492, arg131_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_492 = arg131_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1568: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_1569: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1568, -1);  unsqueeze_1568 = None
        sub_196: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_196, unsqueeze_1569);  convolution_196 = unsqueeze_1569 = None
        add_493: "f32[52]" = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_196: "f32[52]" = torch.ops.aten.sqrt.default(add_493);  add_493 = None
        reciprocal_196: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_196);  sqrt_196 = None
        mul_588: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_196, 1);  reciprocal_196 = None
        unsqueeze_1570: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_588, -1);  mul_588 = None
        unsqueeze_1571: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1570, -1);  unsqueeze_1570 = None
        mul_589: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_1571);  sub_196 = unsqueeze_1571 = None
        unsqueeze_1572: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_1573: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1572, -1);  unsqueeze_1572 = None
        mul_590: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_589, unsqueeze_1573);  mul_589 = unsqueeze_1573 = None
        unsqueeze_1574: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_1575: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1574, -1);  unsqueeze_1574 = None
        add_494: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_590, unsqueeze_1575);  mul_590 = unsqueeze_1575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_190: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_494);  add_494 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_37: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([relu_188, relu_189, relu_190, getitem_763], 1);  relu_188 = relu_189 = relu_190 = getitem_763 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_197: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_37, arg136_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_37 = arg136_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1576: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_1577: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1576, -1);  unsqueeze_1576 = None
        sub_197: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_197, unsqueeze_1577);  convolution_197 = unsqueeze_1577 = None
        add_495: "f32[512]" = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
        sqrt_197: "f32[512]" = torch.ops.aten.sqrt.default(add_495);  add_495 = None
        reciprocal_197: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_197);  sqrt_197 = None
        mul_591: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_197, 1);  reciprocal_197 = None
        unsqueeze_1578: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_591, -1);  mul_591 = None
        unsqueeze_1579: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, -1);  unsqueeze_1578 = None
        mul_592: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_1579);  sub_197 = unsqueeze_1579 = None
        unsqueeze_1580: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_1581: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1580, -1);  unsqueeze_1580 = None
        mul_593: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_592, unsqueeze_1581);  mul_592 = unsqueeze_1581 = None
        unsqueeze_1582: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_1583: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1582, -1);  unsqueeze_1582 = None
        add_496: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_593, unsqueeze_1583);  mul_593 = unsqueeze_1583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_497: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_496, relu_186);  add_496 = relu_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_191: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_497);  add_497 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_198: "f32[8, 208, 28, 28]" = torch.ops.aten.convolution.default(relu_191, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg141_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1584: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_1585: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1584, -1);  unsqueeze_1584 = None
        sub_198: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_198, unsqueeze_1585);  convolution_198 = unsqueeze_1585 = None
        add_498: "f32[208]" = torch.ops.aten.add.Tensor(arg143_1, 1e-05);  arg143_1 = None
        sqrt_198: "f32[208]" = torch.ops.aten.sqrt.default(add_498);  add_498 = None
        reciprocal_198: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_198);  sqrt_198 = None
        mul_594: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_198, 1);  reciprocal_198 = None
        unsqueeze_1586: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_594, -1);  mul_594 = None
        unsqueeze_1587: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1586, -1);  unsqueeze_1586 = None
        mul_595: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_1587);  sub_198 = unsqueeze_1587 = None
        unsqueeze_1588: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_1589: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1588, -1);  unsqueeze_1588 = None
        mul_596: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_1589);  mul_595 = unsqueeze_1589 = None
        unsqueeze_1590: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_1591: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, -1);  unsqueeze_1590 = None
        add_499: "f32[8, 208, 28, 28]" = torch.ops.aten.add.Tensor(mul_596, unsqueeze_1591);  mul_596 = unsqueeze_1591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_192: "f32[8, 208, 28, 28]" = torch.ops.aten.relu.default(add_499);  add_499 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_191 = torch.ops.aten.split.Tensor(relu_192, 52, 1)
        getitem_768: "f32[8, 52, 28, 28]" = split_191[0];  split_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_192 = torch.ops.aten.split.Tensor(relu_192, 52, 1)
        getitem_773: "f32[8, 52, 28, 28]" = split_192[1];  split_192 = None
        split_193 = torch.ops.aten.split.Tensor(relu_192, 52, 1)
        getitem_778: "f32[8, 52, 28, 28]" = split_193[2];  split_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_194 = torch.ops.aten.split.Tensor(relu_192, 52, 1);  relu_192 = None
        getitem_783: "f32[8, 52, 28, 28]" = split_194[3];  split_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_199: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_768, arg146_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_768 = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1592: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_1593: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1592, -1);  unsqueeze_1592 = None
        sub_199: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_199, unsqueeze_1593);  convolution_199 = unsqueeze_1593 = None
        add_500: "f32[52]" = torch.ops.aten.add.Tensor(arg148_1, 1e-05);  arg148_1 = None
        sqrt_199: "f32[52]" = torch.ops.aten.sqrt.default(add_500);  add_500 = None
        reciprocal_199: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_199);  sqrt_199 = None
        mul_597: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_199, 1);  reciprocal_199 = None
        unsqueeze_1594: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_597, -1);  mul_597 = None
        unsqueeze_1595: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1594, -1);  unsqueeze_1594 = None
        mul_598: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_1595);  sub_199 = unsqueeze_1595 = None
        unsqueeze_1596: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_1597: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1596, -1);  unsqueeze_1596 = None
        mul_599: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_598, unsqueeze_1597);  mul_598 = unsqueeze_1597 = None
        unsqueeze_1598: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_1599: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1598, -1);  unsqueeze_1598 = None
        add_501: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_599, unsqueeze_1599);  mul_599 = unsqueeze_1599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_193: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_501);  add_501 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_502: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_193, getitem_773);  getitem_773 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_200: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_502, arg151_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_502 = arg151_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1600: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_1601: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1600, -1);  unsqueeze_1600 = None
        sub_200: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_200, unsqueeze_1601);  convolution_200 = unsqueeze_1601 = None
        add_503: "f32[52]" = torch.ops.aten.add.Tensor(arg153_1, 1e-05);  arg153_1 = None
        sqrt_200: "f32[52]" = torch.ops.aten.sqrt.default(add_503);  add_503 = None
        reciprocal_200: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_200);  sqrt_200 = None
        mul_600: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_200, 1);  reciprocal_200 = None
        unsqueeze_1602: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_600, -1);  mul_600 = None
        unsqueeze_1603: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, -1);  unsqueeze_1602 = None
        mul_601: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_1603);  sub_200 = unsqueeze_1603 = None
        unsqueeze_1604: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_1605: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1604, -1);  unsqueeze_1604 = None
        mul_602: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_601, unsqueeze_1605);  mul_601 = unsqueeze_1605 = None
        unsqueeze_1606: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_1607: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1606, -1);  unsqueeze_1606 = None
        add_504: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_602, unsqueeze_1607);  mul_602 = unsqueeze_1607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_194: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_504);  add_504 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_505: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_194, getitem_778);  getitem_778 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_201: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_505, arg156_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_505 = arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1608: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_1609: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1608, -1);  unsqueeze_1608 = None
        sub_201: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_201, unsqueeze_1609);  convolution_201 = unsqueeze_1609 = None
        add_506: "f32[52]" = torch.ops.aten.add.Tensor(arg158_1, 1e-05);  arg158_1 = None
        sqrt_201: "f32[52]" = torch.ops.aten.sqrt.default(add_506);  add_506 = None
        reciprocal_201: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_201);  sqrt_201 = None
        mul_603: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_201, 1);  reciprocal_201 = None
        unsqueeze_1610: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_603, -1);  mul_603 = None
        unsqueeze_1611: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1610, -1);  unsqueeze_1610 = None
        mul_604: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_1611);  sub_201 = unsqueeze_1611 = None
        unsqueeze_1612: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_1613: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1612, -1);  unsqueeze_1612 = None
        mul_605: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_604, unsqueeze_1613);  mul_604 = unsqueeze_1613 = None
        unsqueeze_1614: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_1615: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, -1);  unsqueeze_1614 = None
        add_507: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_605, unsqueeze_1615);  mul_605 = unsqueeze_1615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_195: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_507);  add_507 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_38: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([relu_193, relu_194, relu_195, getitem_783], 1);  relu_193 = relu_194 = relu_195 = getitem_783 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_202: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_38, arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_38 = arg161_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1616: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_1617: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1616, -1);  unsqueeze_1616 = None
        sub_202: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_202, unsqueeze_1617);  convolution_202 = unsqueeze_1617 = None
        add_508: "f32[512]" = torch.ops.aten.add.Tensor(arg163_1, 1e-05);  arg163_1 = None
        sqrt_202: "f32[512]" = torch.ops.aten.sqrt.default(add_508);  add_508 = None
        reciprocal_202: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_202);  sqrt_202 = None
        mul_606: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_202, 1);  reciprocal_202 = None
        unsqueeze_1618: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_606, -1);  mul_606 = None
        unsqueeze_1619: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1618, -1);  unsqueeze_1618 = None
        mul_607: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_1619);  sub_202 = unsqueeze_1619 = None
        unsqueeze_1620: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_1621: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1620, -1);  unsqueeze_1620 = None
        mul_608: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_607, unsqueeze_1621);  mul_607 = unsqueeze_1621 = None
        unsqueeze_1622: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_1623: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1622, -1);  unsqueeze_1622 = None
        add_509: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_608, unsqueeze_1623);  mul_608 = unsqueeze_1623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_510: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_509, relu_191);  add_509 = relu_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_196: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_510);  add_510 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_203: "f32[8, 208, 28, 28]" = torch.ops.aten.convolution.default(relu_196, arg166_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1624: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_1625: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1624, -1);  unsqueeze_1624 = None
        sub_203: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_203, unsqueeze_1625);  convolution_203 = unsqueeze_1625 = None
        add_511: "f32[208]" = torch.ops.aten.add.Tensor(arg168_1, 1e-05);  arg168_1 = None
        sqrt_203: "f32[208]" = torch.ops.aten.sqrt.default(add_511);  add_511 = None
        reciprocal_203: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_203);  sqrt_203 = None
        mul_609: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_203, 1);  reciprocal_203 = None
        unsqueeze_1626: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_609, -1);  mul_609 = None
        unsqueeze_1627: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, -1);  unsqueeze_1626 = None
        mul_610: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_1627);  sub_203 = unsqueeze_1627 = None
        unsqueeze_1628: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_1629: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1628, -1);  unsqueeze_1628 = None
        mul_611: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(mul_610, unsqueeze_1629);  mul_610 = unsqueeze_1629 = None
        unsqueeze_1630: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_1631: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1630, -1);  unsqueeze_1630 = None
        add_512: "f32[8, 208, 28, 28]" = torch.ops.aten.add.Tensor(mul_611, unsqueeze_1631);  mul_611 = unsqueeze_1631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_197: "f32[8, 208, 28, 28]" = torch.ops.aten.relu.default(add_512);  add_512 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_196 = torch.ops.aten.split.Tensor(relu_197, 52, 1)
        getitem_788: "f32[8, 52, 28, 28]" = split_196[0];  split_196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_197 = torch.ops.aten.split.Tensor(relu_197, 52, 1)
        getitem_793: "f32[8, 52, 28, 28]" = split_197[1];  split_197 = None
        split_198 = torch.ops.aten.split.Tensor(relu_197, 52, 1)
        getitem_798: "f32[8, 52, 28, 28]" = split_198[2];  split_198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_199 = torch.ops.aten.split.Tensor(relu_197, 52, 1);  relu_197 = None
        getitem_803: "f32[8, 52, 28, 28]" = split_199[3];  split_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_204: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_788, arg171_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_788 = arg171_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1632: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_1633: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1632, -1);  unsqueeze_1632 = None
        sub_204: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_204, unsqueeze_1633);  convolution_204 = unsqueeze_1633 = None
        add_513: "f32[52]" = torch.ops.aten.add.Tensor(arg173_1, 1e-05);  arg173_1 = None
        sqrt_204: "f32[52]" = torch.ops.aten.sqrt.default(add_513);  add_513 = None
        reciprocal_204: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_204);  sqrt_204 = None
        mul_612: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_204, 1);  reciprocal_204 = None
        unsqueeze_1634: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_612, -1);  mul_612 = None
        unsqueeze_1635: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1634, -1);  unsqueeze_1634 = None
        mul_613: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_1635);  sub_204 = unsqueeze_1635 = None
        unsqueeze_1636: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_1637: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1636, -1);  unsqueeze_1636 = None
        mul_614: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_613, unsqueeze_1637);  mul_613 = unsqueeze_1637 = None
        unsqueeze_1638: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_1639: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, -1);  unsqueeze_1638 = None
        add_514: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_614, unsqueeze_1639);  mul_614 = unsqueeze_1639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_198: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_514);  add_514 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_515: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_198, getitem_793);  getitem_793 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_205: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_515, arg176_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_515 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1640: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_1641: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1640, -1);  unsqueeze_1640 = None
        sub_205: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_205, unsqueeze_1641);  convolution_205 = unsqueeze_1641 = None
        add_516: "f32[52]" = torch.ops.aten.add.Tensor(arg178_1, 1e-05);  arg178_1 = None
        sqrt_205: "f32[52]" = torch.ops.aten.sqrt.default(add_516);  add_516 = None
        reciprocal_205: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_205);  sqrt_205 = None
        mul_615: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_205, 1);  reciprocal_205 = None
        unsqueeze_1642: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_615, -1);  mul_615 = None
        unsqueeze_1643: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1642, -1);  unsqueeze_1642 = None
        mul_616: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_1643);  sub_205 = unsqueeze_1643 = None
        unsqueeze_1644: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_1645: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1644, -1);  unsqueeze_1644 = None
        mul_617: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_1645);  mul_616 = unsqueeze_1645 = None
        unsqueeze_1646: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_1647: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1646, -1);  unsqueeze_1646 = None
        add_517: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_617, unsqueeze_1647);  mul_617 = unsqueeze_1647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_199: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_517);  add_517 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_518: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_199, getitem_798);  getitem_798 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_206: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_518, arg181_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_518 = arg181_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1648: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_1649: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1648, -1);  unsqueeze_1648 = None
        sub_206: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_206, unsqueeze_1649);  convolution_206 = unsqueeze_1649 = None
        add_519: "f32[52]" = torch.ops.aten.add.Tensor(arg183_1, 1e-05);  arg183_1 = None
        sqrt_206: "f32[52]" = torch.ops.aten.sqrt.default(add_519);  add_519 = None
        reciprocal_206: "f32[52]" = torch.ops.aten.reciprocal.default(sqrt_206);  sqrt_206 = None
        mul_618: "f32[52]" = torch.ops.aten.mul.Tensor(reciprocal_206, 1);  reciprocal_206 = None
        unsqueeze_1650: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(mul_618, -1);  mul_618 = None
        unsqueeze_1651: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, -1);  unsqueeze_1650 = None
        mul_619: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_1651);  sub_206 = unsqueeze_1651 = None
        unsqueeze_1652: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_1653: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1652, -1);  unsqueeze_1652 = None
        mul_620: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_619, unsqueeze_1653);  mul_619 = unsqueeze_1653 = None
        unsqueeze_1654: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_1655: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1654, -1);  unsqueeze_1654 = None
        add_520: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_620, unsqueeze_1655);  mul_620 = unsqueeze_1655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_200: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_520);  add_520 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_39: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([relu_198, relu_199, relu_200, getitem_803], 1);  relu_198 = relu_199 = relu_200 = getitem_803 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_207: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_39, arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_39 = arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1656: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_1657: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1656, -1);  unsqueeze_1656 = None
        sub_207: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_207, unsqueeze_1657);  convolution_207 = unsqueeze_1657 = None
        add_521: "f32[512]" = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_207: "f32[512]" = torch.ops.aten.sqrt.default(add_521);  add_521 = None
        reciprocal_207: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_207);  sqrt_207 = None
        mul_621: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_207, 1);  reciprocal_207 = None
        unsqueeze_1658: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_621, -1);  mul_621 = None
        unsqueeze_1659: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1658, -1);  unsqueeze_1658 = None
        mul_622: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_1659);  sub_207 = unsqueeze_1659 = None
        unsqueeze_1660: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_1661: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1660, -1);  unsqueeze_1660 = None
        mul_623: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_622, unsqueeze_1661);  mul_622 = unsqueeze_1661 = None
        unsqueeze_1662: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_1663: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, -1);  unsqueeze_1662 = None
        add_522: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_623, unsqueeze_1663);  mul_623 = unsqueeze_1663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_523: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_522, relu_196);  add_522 = relu_196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_201: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_523);  add_523 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_208: "f32[8, 416, 28, 28]" = torch.ops.aten.convolution.default(relu_201, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg191_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1664: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_1665: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1664, -1);  unsqueeze_1664 = None
        sub_208: "f32[8, 416, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_208, unsqueeze_1665);  convolution_208 = unsqueeze_1665 = None
        add_524: "f32[416]" = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_208: "f32[416]" = torch.ops.aten.sqrt.default(add_524);  add_524 = None
        reciprocal_208: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_208);  sqrt_208 = None
        mul_624: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_208, 1);  reciprocal_208 = None
        unsqueeze_1666: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_624, -1);  mul_624 = None
        unsqueeze_1667: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1666, -1);  unsqueeze_1666 = None
        mul_625: "f32[8, 416, 28, 28]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_1667);  sub_208 = unsqueeze_1667 = None
        unsqueeze_1668: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_1669: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1668, -1);  unsqueeze_1668 = None
        mul_626: "f32[8, 416, 28, 28]" = torch.ops.aten.mul.Tensor(mul_625, unsqueeze_1669);  mul_625 = unsqueeze_1669 = None
        unsqueeze_1670: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_1671: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1670, -1);  unsqueeze_1670 = None
        add_525: "f32[8, 416, 28, 28]" = torch.ops.aten.add.Tensor(mul_626, unsqueeze_1671);  mul_626 = unsqueeze_1671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_202: "f32[8, 416, 28, 28]" = torch.ops.aten.relu.default(add_525);  add_525 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_201 = torch.ops.aten.split.Tensor(relu_202, 104, 1)
        getitem_808: "f32[8, 104, 28, 28]" = split_201[0];  split_201 = None
        split_202 = torch.ops.aten.split.Tensor(relu_202, 104, 1)
        getitem_813: "f32[8, 104, 28, 28]" = split_202[1];  split_202 = None
        split_203 = torch.ops.aten.split.Tensor(relu_202, 104, 1)
        getitem_818: "f32[8, 104, 28, 28]" = split_203[2];  split_203 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:99 in forward, code: spo.append(self.pool(spx[-1]))
        split_204 = torch.ops.aten.split.Tensor(relu_202, 104, 1);  relu_202 = None
        getitem_823: "f32[8, 104, 28, 28]" = split_204[3];  split_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_209: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_808, arg196_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_808 = arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1672: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_1673: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1672, -1);  unsqueeze_1672 = None
        sub_209: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_209, unsqueeze_1673);  convolution_209 = unsqueeze_1673 = None
        add_526: "f32[104]" = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_209: "f32[104]" = torch.ops.aten.sqrt.default(add_526);  add_526 = None
        reciprocal_209: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_209);  sqrt_209 = None
        mul_627: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_209, 1);  reciprocal_209 = None
        unsqueeze_1674: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_627, -1);  mul_627 = None
        unsqueeze_1675: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1674, -1);  unsqueeze_1674 = None
        mul_628: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_1675);  sub_209 = unsqueeze_1675 = None
        unsqueeze_1676: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_1677: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1676, -1);  unsqueeze_1676 = None
        mul_629: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_628, unsqueeze_1677);  mul_628 = unsqueeze_1677 = None
        unsqueeze_1678: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_1679: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1678, -1);  unsqueeze_1678 = None
        add_527: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_629, unsqueeze_1679);  mul_629 = unsqueeze_1679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_203: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_527);  add_527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_210: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_813, arg201_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_813 = arg201_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1680: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_1681: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1680, -1);  unsqueeze_1680 = None
        sub_210: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_210, unsqueeze_1681);  convolution_210 = unsqueeze_1681 = None
        add_528: "f32[104]" = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_210: "f32[104]" = torch.ops.aten.sqrt.default(add_528);  add_528 = None
        reciprocal_210: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_210);  sqrt_210 = None
        mul_630: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_210, 1);  reciprocal_210 = None
        unsqueeze_1682: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_630, -1);  mul_630 = None
        unsqueeze_1683: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1682, -1);  unsqueeze_1682 = None
        mul_631: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_1683);  sub_210 = unsqueeze_1683 = None
        unsqueeze_1684: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_1685: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1684, -1);  unsqueeze_1684 = None
        mul_632: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_631, unsqueeze_1685);  mul_631 = unsqueeze_1685 = None
        unsqueeze_1686: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_1687: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1686, -1);  unsqueeze_1686 = None
        add_529: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_632, unsqueeze_1687);  mul_632 = unsqueeze_1687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_204: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_529);  add_529 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_211: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_818, arg206_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_818 = arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1688: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_1689: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1688, -1);  unsqueeze_1688 = None
        sub_211: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_211, unsqueeze_1689);  convolution_211 = unsqueeze_1689 = None
        add_530: "f32[104]" = torch.ops.aten.add.Tensor(arg208_1, 1e-05);  arg208_1 = None
        sqrt_211: "f32[104]" = torch.ops.aten.sqrt.default(add_530);  add_530 = None
        reciprocal_211: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_211);  sqrt_211 = None
        mul_633: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_211, 1);  reciprocal_211 = None
        unsqueeze_1690: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_633, -1);  mul_633 = None
        unsqueeze_1691: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1690, -1);  unsqueeze_1690 = None
        mul_634: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_1691);  sub_211 = unsqueeze_1691 = None
        unsqueeze_1692: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_1693: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1692, -1);  unsqueeze_1692 = None
        mul_635: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_634, unsqueeze_1693);  mul_634 = unsqueeze_1693 = None
        unsqueeze_1694: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_1695: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1694, -1);  unsqueeze_1694 = None
        add_531: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_635, unsqueeze_1695);  mul_635 = unsqueeze_1695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_205: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_531);  add_531 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:99 in forward, code: spo.append(self.pool(spx[-1]))
        avg_pool2d_6: "f32[8, 104, 14, 14]" = torch.ops.aten.avg_pool2d.default(getitem_823, [3, 3], [2, 2], [1, 1]);  getitem_823 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_40: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_203, relu_204, relu_205, avg_pool2d_6], 1);  relu_203 = relu_204 = relu_205 = avg_pool2d_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_212: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_40, arg211_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_40 = arg211_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1696: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_1697: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1696, -1);  unsqueeze_1696 = None
        sub_212: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_212, unsqueeze_1697);  convolution_212 = unsqueeze_1697 = None
        add_532: "f32[1024]" = torch.ops.aten.add.Tensor(arg213_1, 1e-05);  arg213_1 = None
        sqrt_212: "f32[1024]" = torch.ops.aten.sqrt.default(add_532);  add_532 = None
        reciprocal_212: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_212);  sqrt_212 = None
        mul_636: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_212, 1);  reciprocal_212 = None
        unsqueeze_1698: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_636, -1);  mul_636 = None
        unsqueeze_1699: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1698, -1);  unsqueeze_1698 = None
        mul_637: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_1699);  sub_212 = unsqueeze_1699 = None
        unsqueeze_1700: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_1701: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1700, -1);  unsqueeze_1700 = None
        mul_638: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_1701);  mul_637 = unsqueeze_1701 = None
        unsqueeze_1702: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_1703: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1702, -1);  unsqueeze_1702 = None
        add_533: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_638, unsqueeze_1703);  mul_638 = unsqueeze_1703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:111 in forward, code: shortcut = self.downsample(x)
        convolution_213: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_201, arg216_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_201 = arg216_1 = None
        unsqueeze_1704: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_1705: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1704, -1);  unsqueeze_1704 = None
        sub_213: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_213, unsqueeze_1705);  convolution_213 = unsqueeze_1705 = None
        add_534: "f32[1024]" = torch.ops.aten.add.Tensor(arg218_1, 1e-05);  arg218_1 = None
        sqrt_213: "f32[1024]" = torch.ops.aten.sqrt.default(add_534);  add_534 = None
        reciprocal_213: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_213);  sqrt_213 = None
        mul_639: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_213, 1);  reciprocal_213 = None
        unsqueeze_1706: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_639, -1);  mul_639 = None
        unsqueeze_1707: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1706, -1);  unsqueeze_1706 = None
        mul_640: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_1707);  sub_213 = unsqueeze_1707 = None
        unsqueeze_1708: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_1709: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1708, -1);  unsqueeze_1708 = None
        mul_641: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_640, unsqueeze_1709);  mul_640 = unsqueeze_1709 = None
        unsqueeze_1710: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_1711: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1710, -1);  unsqueeze_1710 = None
        add_535: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_641, unsqueeze_1711);  mul_641 = unsqueeze_1711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_536: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_533, add_535);  add_533 = add_535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_206: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_536);  add_536 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_214: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_206, arg221_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg221_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1712: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_1713: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1712, -1);  unsqueeze_1712 = None
        sub_214: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_214, unsqueeze_1713);  convolution_214 = unsqueeze_1713 = None
        add_537: "f32[416]" = torch.ops.aten.add.Tensor(arg223_1, 1e-05);  arg223_1 = None
        sqrt_214: "f32[416]" = torch.ops.aten.sqrt.default(add_537);  add_537 = None
        reciprocal_214: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_214);  sqrt_214 = None
        mul_642: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_214, 1);  reciprocal_214 = None
        unsqueeze_1714: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_642, -1);  mul_642 = None
        unsqueeze_1715: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1714, -1);  unsqueeze_1714 = None
        mul_643: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_1715);  sub_214 = unsqueeze_1715 = None
        unsqueeze_1716: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_1717: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1716, -1);  unsqueeze_1716 = None
        mul_644: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_643, unsqueeze_1717);  mul_643 = unsqueeze_1717 = None
        unsqueeze_1718: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_1719: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1718, -1);  unsqueeze_1718 = None
        add_538: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_644, unsqueeze_1719);  mul_644 = unsqueeze_1719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_207: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_538);  add_538 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_206 = torch.ops.aten.split.Tensor(relu_207, 104, 1)
        getitem_828: "f32[8, 104, 14, 14]" = split_206[0];  split_206 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_207 = torch.ops.aten.split.Tensor(relu_207, 104, 1)
        getitem_833: "f32[8, 104, 14, 14]" = split_207[1];  split_207 = None
        split_208 = torch.ops.aten.split.Tensor(relu_207, 104, 1)
        getitem_838: "f32[8, 104, 14, 14]" = split_208[2];  split_208 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_209 = torch.ops.aten.split.Tensor(relu_207, 104, 1);  relu_207 = None
        getitem_843: "f32[8, 104, 14, 14]" = split_209[3];  split_209 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_215: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_828, arg226_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_828 = arg226_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1720: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_1721: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1720, -1);  unsqueeze_1720 = None
        sub_215: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_215, unsqueeze_1721);  convolution_215 = unsqueeze_1721 = None
        add_539: "f32[104]" = torch.ops.aten.add.Tensor(arg228_1, 1e-05);  arg228_1 = None
        sqrt_215: "f32[104]" = torch.ops.aten.sqrt.default(add_539);  add_539 = None
        reciprocal_215: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_215);  sqrt_215 = None
        mul_645: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_215, 1);  reciprocal_215 = None
        unsqueeze_1722: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_645, -1);  mul_645 = None
        unsqueeze_1723: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1722, -1);  unsqueeze_1722 = None
        mul_646: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_1723);  sub_215 = unsqueeze_1723 = None
        unsqueeze_1724: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_1725: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1724, -1);  unsqueeze_1724 = None
        mul_647: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_646, unsqueeze_1725);  mul_646 = unsqueeze_1725 = None
        unsqueeze_1726: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_1727: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1726, -1);  unsqueeze_1726 = None
        add_540: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_647, unsqueeze_1727);  mul_647 = unsqueeze_1727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_208: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_540);  add_540 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_541: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_208, getitem_833);  getitem_833 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_216: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_541, arg231_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_541 = arg231_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1728: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_1729: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1728, -1);  unsqueeze_1728 = None
        sub_216: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_216, unsqueeze_1729);  convolution_216 = unsqueeze_1729 = None
        add_542: "f32[104]" = torch.ops.aten.add.Tensor(arg233_1, 1e-05);  arg233_1 = None
        sqrt_216: "f32[104]" = torch.ops.aten.sqrt.default(add_542);  add_542 = None
        reciprocal_216: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_216);  sqrt_216 = None
        mul_648: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_216, 1);  reciprocal_216 = None
        unsqueeze_1730: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_648, -1);  mul_648 = None
        unsqueeze_1731: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1730, -1);  unsqueeze_1730 = None
        mul_649: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_1731);  sub_216 = unsqueeze_1731 = None
        unsqueeze_1732: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_1733: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1732, -1);  unsqueeze_1732 = None
        mul_650: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_649, unsqueeze_1733);  mul_649 = unsqueeze_1733 = None
        unsqueeze_1734: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_1735: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1734, -1);  unsqueeze_1734 = None
        add_543: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_650, unsqueeze_1735);  mul_650 = unsqueeze_1735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_209: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_543);  add_543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_544: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_209, getitem_838);  getitem_838 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_217: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_544, arg236_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_544 = arg236_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1736: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_1737: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1736, -1);  unsqueeze_1736 = None
        sub_217: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_217, unsqueeze_1737);  convolution_217 = unsqueeze_1737 = None
        add_545: "f32[104]" = torch.ops.aten.add.Tensor(arg238_1, 1e-05);  arg238_1 = None
        sqrt_217: "f32[104]" = torch.ops.aten.sqrt.default(add_545);  add_545 = None
        reciprocal_217: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_217);  sqrt_217 = None
        mul_651: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_217, 1);  reciprocal_217 = None
        unsqueeze_1738: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_651, -1);  mul_651 = None
        unsqueeze_1739: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1738, -1);  unsqueeze_1738 = None
        mul_652: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_1739);  sub_217 = unsqueeze_1739 = None
        unsqueeze_1740: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_1741: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1740, -1);  unsqueeze_1740 = None
        mul_653: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_652, unsqueeze_1741);  mul_652 = unsqueeze_1741 = None
        unsqueeze_1742: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_1743: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1742, -1);  unsqueeze_1742 = None
        add_546: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_653, unsqueeze_1743);  mul_653 = unsqueeze_1743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_210: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_546);  add_546 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_41: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_208, relu_209, relu_210, getitem_843], 1);  relu_208 = relu_209 = relu_210 = getitem_843 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_218: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_41, arg241_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_41 = arg241_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1744: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_1745: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1744, -1);  unsqueeze_1744 = None
        sub_218: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_218, unsqueeze_1745);  convolution_218 = unsqueeze_1745 = None
        add_547: "f32[1024]" = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_218: "f32[1024]" = torch.ops.aten.sqrt.default(add_547);  add_547 = None
        reciprocal_218: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_218);  sqrt_218 = None
        mul_654: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_218, 1);  reciprocal_218 = None
        unsqueeze_1746: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_654, -1);  mul_654 = None
        unsqueeze_1747: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1746, -1);  unsqueeze_1746 = None
        mul_655: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_1747);  sub_218 = unsqueeze_1747 = None
        unsqueeze_1748: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_1749: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1748, -1);  unsqueeze_1748 = None
        mul_656: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_655, unsqueeze_1749);  mul_655 = unsqueeze_1749 = None
        unsqueeze_1750: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_1751: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1750, -1);  unsqueeze_1750 = None
        add_548: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_656, unsqueeze_1751);  mul_656 = unsqueeze_1751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_549: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_548, relu_206);  add_548 = relu_206 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_211: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_549);  add_549 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_219: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_211, arg246_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg246_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1752: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_1753: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1752, -1);  unsqueeze_1752 = None
        sub_219: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_219, unsqueeze_1753);  convolution_219 = unsqueeze_1753 = None
        add_550: "f32[416]" = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
        sqrt_219: "f32[416]" = torch.ops.aten.sqrt.default(add_550);  add_550 = None
        reciprocal_219: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_219);  sqrt_219 = None
        mul_657: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_219, 1);  reciprocal_219 = None
        unsqueeze_1754: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_657, -1);  mul_657 = None
        unsqueeze_1755: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1754, -1);  unsqueeze_1754 = None
        mul_658: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_1755);  sub_219 = unsqueeze_1755 = None
        unsqueeze_1756: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_1757: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1756, -1);  unsqueeze_1756 = None
        mul_659: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_1757);  mul_658 = unsqueeze_1757 = None
        unsqueeze_1758: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_1759: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1758, -1);  unsqueeze_1758 = None
        add_551: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_659, unsqueeze_1759);  mul_659 = unsqueeze_1759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_212: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_551);  add_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_211 = torch.ops.aten.split.Tensor(relu_212, 104, 1)
        getitem_848: "f32[8, 104, 14, 14]" = split_211[0];  split_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_212 = torch.ops.aten.split.Tensor(relu_212, 104, 1)
        getitem_853: "f32[8, 104, 14, 14]" = split_212[1];  split_212 = None
        split_213 = torch.ops.aten.split.Tensor(relu_212, 104, 1)
        getitem_858: "f32[8, 104, 14, 14]" = split_213[2];  split_213 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_214 = torch.ops.aten.split.Tensor(relu_212, 104, 1);  relu_212 = None
        getitem_863: "f32[8, 104, 14, 14]" = split_214[3];  split_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_220: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_848, arg251_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_848 = arg251_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1760: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_1761: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1760, -1);  unsqueeze_1760 = None
        sub_220: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_220, unsqueeze_1761);  convolution_220 = unsqueeze_1761 = None
        add_552: "f32[104]" = torch.ops.aten.add.Tensor(arg253_1, 1e-05);  arg253_1 = None
        sqrt_220: "f32[104]" = torch.ops.aten.sqrt.default(add_552);  add_552 = None
        reciprocal_220: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_220);  sqrt_220 = None
        mul_660: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_220, 1);  reciprocal_220 = None
        unsqueeze_1762: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_660, -1);  mul_660 = None
        unsqueeze_1763: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1762, -1);  unsqueeze_1762 = None
        mul_661: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_1763);  sub_220 = unsqueeze_1763 = None
        unsqueeze_1764: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_1765: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1764, -1);  unsqueeze_1764 = None
        mul_662: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_661, unsqueeze_1765);  mul_661 = unsqueeze_1765 = None
        unsqueeze_1766: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_1767: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1766, -1);  unsqueeze_1766 = None
        add_553: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_662, unsqueeze_1767);  mul_662 = unsqueeze_1767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_213: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_553);  add_553 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_554: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_213, getitem_853);  getitem_853 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_221: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_554, arg256_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_554 = arg256_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1768: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_1769: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1768, -1);  unsqueeze_1768 = None
        sub_221: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_221, unsqueeze_1769);  convolution_221 = unsqueeze_1769 = None
        add_555: "f32[104]" = torch.ops.aten.add.Tensor(arg258_1, 1e-05);  arg258_1 = None
        sqrt_221: "f32[104]" = torch.ops.aten.sqrt.default(add_555);  add_555 = None
        reciprocal_221: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_221);  sqrt_221 = None
        mul_663: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_221, 1);  reciprocal_221 = None
        unsqueeze_1770: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_663, -1);  mul_663 = None
        unsqueeze_1771: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1770, -1);  unsqueeze_1770 = None
        mul_664: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_1771);  sub_221 = unsqueeze_1771 = None
        unsqueeze_1772: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_1773: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1772, -1);  unsqueeze_1772 = None
        mul_665: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_664, unsqueeze_1773);  mul_664 = unsqueeze_1773 = None
        unsqueeze_1774: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_1775: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1774, -1);  unsqueeze_1774 = None
        add_556: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_665, unsqueeze_1775);  mul_665 = unsqueeze_1775 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_214: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_556);  add_556 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_557: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_214, getitem_858);  getitem_858 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_222: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_557, arg261_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_557 = arg261_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1776: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_1777: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1776, -1);  unsqueeze_1776 = None
        sub_222: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_222, unsqueeze_1777);  convolution_222 = unsqueeze_1777 = None
        add_558: "f32[104]" = torch.ops.aten.add.Tensor(arg263_1, 1e-05);  arg263_1 = None
        sqrt_222: "f32[104]" = torch.ops.aten.sqrt.default(add_558);  add_558 = None
        reciprocal_222: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_222);  sqrt_222 = None
        mul_666: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_222, 1);  reciprocal_222 = None
        unsqueeze_1778: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_666, -1);  mul_666 = None
        unsqueeze_1779: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1778, -1);  unsqueeze_1778 = None
        mul_667: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_1779);  sub_222 = unsqueeze_1779 = None
        unsqueeze_1780: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_1781: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1780, -1);  unsqueeze_1780 = None
        mul_668: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_667, unsqueeze_1781);  mul_667 = unsqueeze_1781 = None
        unsqueeze_1782: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_1783: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1782, -1);  unsqueeze_1782 = None
        add_559: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_668, unsqueeze_1783);  mul_668 = unsqueeze_1783 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_215: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_559);  add_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_42: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_213, relu_214, relu_215, getitem_863], 1);  relu_213 = relu_214 = relu_215 = getitem_863 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_223: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_42, arg266_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_42 = arg266_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1784: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_1785: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1784, -1);  unsqueeze_1784 = None
        sub_223: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_223, unsqueeze_1785);  convolution_223 = unsqueeze_1785 = None
        add_560: "f32[1024]" = torch.ops.aten.add.Tensor(arg268_1, 1e-05);  arg268_1 = None
        sqrt_223: "f32[1024]" = torch.ops.aten.sqrt.default(add_560);  add_560 = None
        reciprocal_223: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_223);  sqrt_223 = None
        mul_669: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_223, 1);  reciprocal_223 = None
        unsqueeze_1786: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_669, -1);  mul_669 = None
        unsqueeze_1787: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1786, -1);  unsqueeze_1786 = None
        mul_670: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_1787);  sub_223 = unsqueeze_1787 = None
        unsqueeze_1788: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_1789: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1788, -1);  unsqueeze_1788 = None
        mul_671: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_670, unsqueeze_1789);  mul_670 = unsqueeze_1789 = None
        unsqueeze_1790: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_1791: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1790, -1);  unsqueeze_1790 = None
        add_561: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_671, unsqueeze_1791);  mul_671 = unsqueeze_1791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_562: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_561, relu_211);  add_561 = relu_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_216: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_562);  add_562 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_224: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_216, arg271_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg271_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1792: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_1793: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1792, -1);  unsqueeze_1792 = None
        sub_224: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_224, unsqueeze_1793);  convolution_224 = unsqueeze_1793 = None
        add_563: "f32[416]" = torch.ops.aten.add.Tensor(arg273_1, 1e-05);  arg273_1 = None
        sqrt_224: "f32[416]" = torch.ops.aten.sqrt.default(add_563);  add_563 = None
        reciprocal_224: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_224);  sqrt_224 = None
        mul_672: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_224, 1);  reciprocal_224 = None
        unsqueeze_1794: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_672, -1);  mul_672 = None
        unsqueeze_1795: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1794, -1);  unsqueeze_1794 = None
        mul_673: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_1795);  sub_224 = unsqueeze_1795 = None
        unsqueeze_1796: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_1797: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1796, -1);  unsqueeze_1796 = None
        mul_674: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_673, unsqueeze_1797);  mul_673 = unsqueeze_1797 = None
        unsqueeze_1798: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_1799: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1798, -1);  unsqueeze_1798 = None
        add_564: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_674, unsqueeze_1799);  mul_674 = unsqueeze_1799 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_217: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_564);  add_564 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_216 = torch.ops.aten.split.Tensor(relu_217, 104, 1)
        getitem_868: "f32[8, 104, 14, 14]" = split_216[0];  split_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_217 = torch.ops.aten.split.Tensor(relu_217, 104, 1)
        getitem_873: "f32[8, 104, 14, 14]" = split_217[1];  split_217 = None
        split_218 = torch.ops.aten.split.Tensor(relu_217, 104, 1)
        getitem_878: "f32[8, 104, 14, 14]" = split_218[2];  split_218 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_219 = torch.ops.aten.split.Tensor(relu_217, 104, 1);  relu_217 = None
        getitem_883: "f32[8, 104, 14, 14]" = split_219[3];  split_219 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_225: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_868, arg276_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_868 = arg276_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1800: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_1801: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1800, -1);  unsqueeze_1800 = None
        sub_225: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_225, unsqueeze_1801);  convolution_225 = unsqueeze_1801 = None
        add_565: "f32[104]" = torch.ops.aten.add.Tensor(arg278_1, 1e-05);  arg278_1 = None
        sqrt_225: "f32[104]" = torch.ops.aten.sqrt.default(add_565);  add_565 = None
        reciprocal_225: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_225);  sqrt_225 = None
        mul_675: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_225, 1);  reciprocal_225 = None
        unsqueeze_1802: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_675, -1);  mul_675 = None
        unsqueeze_1803: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1802, -1);  unsqueeze_1802 = None
        mul_676: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_1803);  sub_225 = unsqueeze_1803 = None
        unsqueeze_1804: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_1805: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1804, -1);  unsqueeze_1804 = None
        mul_677: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_676, unsqueeze_1805);  mul_676 = unsqueeze_1805 = None
        unsqueeze_1806: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_1807: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1806, -1);  unsqueeze_1806 = None
        add_566: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_677, unsqueeze_1807);  mul_677 = unsqueeze_1807 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_218: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_566);  add_566 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_567: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_218, getitem_873);  getitem_873 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_226: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_567, arg281_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_567 = arg281_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1808: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_1809: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1808, -1);  unsqueeze_1808 = None
        sub_226: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_226, unsqueeze_1809);  convolution_226 = unsqueeze_1809 = None
        add_568: "f32[104]" = torch.ops.aten.add.Tensor(arg283_1, 1e-05);  arg283_1 = None
        sqrt_226: "f32[104]" = torch.ops.aten.sqrt.default(add_568);  add_568 = None
        reciprocal_226: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_226);  sqrt_226 = None
        mul_678: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_226, 1);  reciprocal_226 = None
        unsqueeze_1810: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_678, -1);  mul_678 = None
        unsqueeze_1811: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1810, -1);  unsqueeze_1810 = None
        mul_679: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_1811);  sub_226 = unsqueeze_1811 = None
        unsqueeze_1812: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_1813: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1812, -1);  unsqueeze_1812 = None
        mul_680: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_679, unsqueeze_1813);  mul_679 = unsqueeze_1813 = None
        unsqueeze_1814: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_1815: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1814, -1);  unsqueeze_1814 = None
        add_569: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_680, unsqueeze_1815);  mul_680 = unsqueeze_1815 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_219: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_569);  add_569 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_570: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_219, getitem_878);  getitem_878 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_227: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_570, arg286_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_570 = arg286_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1816: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_1817: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1816, -1);  unsqueeze_1816 = None
        sub_227: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_227, unsqueeze_1817);  convolution_227 = unsqueeze_1817 = None
        add_571: "f32[104]" = torch.ops.aten.add.Tensor(arg288_1, 1e-05);  arg288_1 = None
        sqrt_227: "f32[104]" = torch.ops.aten.sqrt.default(add_571);  add_571 = None
        reciprocal_227: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_227);  sqrt_227 = None
        mul_681: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_227, 1);  reciprocal_227 = None
        unsqueeze_1818: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_681, -1);  mul_681 = None
        unsqueeze_1819: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1818, -1);  unsqueeze_1818 = None
        mul_682: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_1819);  sub_227 = unsqueeze_1819 = None
        unsqueeze_1820: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_1821: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1820, -1);  unsqueeze_1820 = None
        mul_683: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_682, unsqueeze_1821);  mul_682 = unsqueeze_1821 = None
        unsqueeze_1822: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_1823: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1822, -1);  unsqueeze_1822 = None
        add_572: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_683, unsqueeze_1823);  mul_683 = unsqueeze_1823 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_220: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_572);  add_572 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_43: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_218, relu_219, relu_220, getitem_883], 1);  relu_218 = relu_219 = relu_220 = getitem_883 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_228: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_43, arg291_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_43 = arg291_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1824: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_1825: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1824, -1);  unsqueeze_1824 = None
        sub_228: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_228, unsqueeze_1825);  convolution_228 = unsqueeze_1825 = None
        add_573: "f32[1024]" = torch.ops.aten.add.Tensor(arg293_1, 1e-05);  arg293_1 = None
        sqrt_228: "f32[1024]" = torch.ops.aten.sqrt.default(add_573);  add_573 = None
        reciprocal_228: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_228);  sqrt_228 = None
        mul_684: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_228, 1);  reciprocal_228 = None
        unsqueeze_1826: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_684, -1);  mul_684 = None
        unsqueeze_1827: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1826, -1);  unsqueeze_1826 = None
        mul_685: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_1827);  sub_228 = unsqueeze_1827 = None
        unsqueeze_1828: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_1829: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1828, -1);  unsqueeze_1828 = None
        mul_686: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_685, unsqueeze_1829);  mul_685 = unsqueeze_1829 = None
        unsqueeze_1830: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_1831: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1830, -1);  unsqueeze_1830 = None
        add_574: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_686, unsqueeze_1831);  mul_686 = unsqueeze_1831 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_575: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_574, relu_216);  add_574 = relu_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_221: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_575);  add_575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_229: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_221, arg296_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg296_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1832: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg297_1, -1);  arg297_1 = None
        unsqueeze_1833: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1832, -1);  unsqueeze_1832 = None
        sub_229: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_229, unsqueeze_1833);  convolution_229 = unsqueeze_1833 = None
        add_576: "f32[416]" = torch.ops.aten.add.Tensor(arg298_1, 1e-05);  arg298_1 = None
        sqrt_229: "f32[416]" = torch.ops.aten.sqrt.default(add_576);  add_576 = None
        reciprocal_229: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_229);  sqrt_229 = None
        mul_687: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_229, 1);  reciprocal_229 = None
        unsqueeze_1834: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_687, -1);  mul_687 = None
        unsqueeze_1835: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1834, -1);  unsqueeze_1834 = None
        mul_688: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_1835);  sub_229 = unsqueeze_1835 = None
        unsqueeze_1836: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_1837: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1836, -1);  unsqueeze_1836 = None
        mul_689: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_688, unsqueeze_1837);  mul_688 = unsqueeze_1837 = None
        unsqueeze_1838: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_1839: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1838, -1);  unsqueeze_1838 = None
        add_577: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_689, unsqueeze_1839);  mul_689 = unsqueeze_1839 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_222: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_577);  add_577 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_221 = torch.ops.aten.split.Tensor(relu_222, 104, 1)
        getitem_888: "f32[8, 104, 14, 14]" = split_221[0];  split_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_222 = torch.ops.aten.split.Tensor(relu_222, 104, 1)
        getitem_893: "f32[8, 104, 14, 14]" = split_222[1];  split_222 = None
        split_223 = torch.ops.aten.split.Tensor(relu_222, 104, 1)
        getitem_898: "f32[8, 104, 14, 14]" = split_223[2];  split_223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_224 = torch.ops.aten.split.Tensor(relu_222, 104, 1);  relu_222 = None
        getitem_903: "f32[8, 104, 14, 14]" = split_224[3];  split_224 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_230: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_888, arg301_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_888 = arg301_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1840: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_1841: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1840, -1);  unsqueeze_1840 = None
        sub_230: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_230, unsqueeze_1841);  convolution_230 = unsqueeze_1841 = None
        add_578: "f32[104]" = torch.ops.aten.add.Tensor(arg303_1, 1e-05);  arg303_1 = None
        sqrt_230: "f32[104]" = torch.ops.aten.sqrt.default(add_578);  add_578 = None
        reciprocal_230: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_230);  sqrt_230 = None
        mul_690: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_230, 1);  reciprocal_230 = None
        unsqueeze_1842: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_690, -1);  mul_690 = None
        unsqueeze_1843: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1842, -1);  unsqueeze_1842 = None
        mul_691: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_1843);  sub_230 = unsqueeze_1843 = None
        unsqueeze_1844: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_1845: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1844, -1);  unsqueeze_1844 = None
        mul_692: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_691, unsqueeze_1845);  mul_691 = unsqueeze_1845 = None
        unsqueeze_1846: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_1847: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1846, -1);  unsqueeze_1846 = None
        add_579: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_692, unsqueeze_1847);  mul_692 = unsqueeze_1847 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_223: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_579);  add_579 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_580: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_223, getitem_893);  getitem_893 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_231: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_580, arg306_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_580 = arg306_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1848: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg307_1, -1);  arg307_1 = None
        unsqueeze_1849: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1848, -1);  unsqueeze_1848 = None
        sub_231: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_231, unsqueeze_1849);  convolution_231 = unsqueeze_1849 = None
        add_581: "f32[104]" = torch.ops.aten.add.Tensor(arg308_1, 1e-05);  arg308_1 = None
        sqrt_231: "f32[104]" = torch.ops.aten.sqrt.default(add_581);  add_581 = None
        reciprocal_231: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_231);  sqrt_231 = None
        mul_693: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_231, 1);  reciprocal_231 = None
        unsqueeze_1850: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_693, -1);  mul_693 = None
        unsqueeze_1851: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1850, -1);  unsqueeze_1850 = None
        mul_694: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_1851);  sub_231 = unsqueeze_1851 = None
        unsqueeze_1852: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_1853: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1852, -1);  unsqueeze_1852 = None
        mul_695: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_694, unsqueeze_1853);  mul_694 = unsqueeze_1853 = None
        unsqueeze_1854: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
        unsqueeze_1855: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1854, -1);  unsqueeze_1854 = None
        add_582: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_695, unsqueeze_1855);  mul_695 = unsqueeze_1855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_224: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_582);  add_582 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_583: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_224, getitem_898);  getitem_898 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_232: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_583, arg311_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_583 = arg311_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1856: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_1857: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1856, -1);  unsqueeze_1856 = None
        sub_232: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_232, unsqueeze_1857);  convolution_232 = unsqueeze_1857 = None
        add_584: "f32[104]" = torch.ops.aten.add.Tensor(arg313_1, 1e-05);  arg313_1 = None
        sqrt_232: "f32[104]" = torch.ops.aten.sqrt.default(add_584);  add_584 = None
        reciprocal_232: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_232);  sqrt_232 = None
        mul_696: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_232, 1);  reciprocal_232 = None
        unsqueeze_1858: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_696, -1);  mul_696 = None
        unsqueeze_1859: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1858, -1);  unsqueeze_1858 = None
        mul_697: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_1859);  sub_232 = unsqueeze_1859 = None
        unsqueeze_1860: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_1861: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1860, -1);  unsqueeze_1860 = None
        mul_698: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_697, unsqueeze_1861);  mul_697 = unsqueeze_1861 = None
        unsqueeze_1862: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg315_1, -1);  arg315_1 = None
        unsqueeze_1863: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1862, -1);  unsqueeze_1862 = None
        add_585: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_698, unsqueeze_1863);  mul_698 = unsqueeze_1863 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_225: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_585);  add_585 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_44: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_223, relu_224, relu_225, getitem_903], 1);  relu_223 = relu_224 = relu_225 = getitem_903 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_233: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_44, arg316_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_44 = arg316_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1864: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
        unsqueeze_1865: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1864, -1);  unsqueeze_1864 = None
        sub_233: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_233, unsqueeze_1865);  convolution_233 = unsqueeze_1865 = None
        add_586: "f32[1024]" = torch.ops.aten.add.Tensor(arg318_1, 1e-05);  arg318_1 = None
        sqrt_233: "f32[1024]" = torch.ops.aten.sqrt.default(add_586);  add_586 = None
        reciprocal_233: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_233);  sqrt_233 = None
        mul_699: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_233, 1);  reciprocal_233 = None
        unsqueeze_1866: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_699, -1);  mul_699 = None
        unsqueeze_1867: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1866, -1);  unsqueeze_1866 = None
        mul_700: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_1867);  sub_233 = unsqueeze_1867 = None
        unsqueeze_1868: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_1869: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1868, -1);  unsqueeze_1868 = None
        mul_701: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_700, unsqueeze_1869);  mul_700 = unsqueeze_1869 = None
        unsqueeze_1870: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_1871: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1870, -1);  unsqueeze_1870 = None
        add_587: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_701, unsqueeze_1871);  mul_701 = unsqueeze_1871 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_588: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_587, relu_221);  add_587 = relu_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_226: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_588);  add_588 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_234: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_226, arg321_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg321_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1872: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_1873: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1872, -1);  unsqueeze_1872 = None
        sub_234: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_234, unsqueeze_1873);  convolution_234 = unsqueeze_1873 = None
        add_589: "f32[416]" = torch.ops.aten.add.Tensor(arg323_1, 1e-05);  arg323_1 = None
        sqrt_234: "f32[416]" = torch.ops.aten.sqrt.default(add_589);  add_589 = None
        reciprocal_234: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_234);  sqrt_234 = None
        mul_702: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_234, 1);  reciprocal_234 = None
        unsqueeze_1874: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_702, -1);  mul_702 = None
        unsqueeze_1875: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1874, -1);  unsqueeze_1874 = None
        mul_703: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_1875);  sub_234 = unsqueeze_1875 = None
        unsqueeze_1876: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_1877: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1876, -1);  unsqueeze_1876 = None
        mul_704: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_703, unsqueeze_1877);  mul_703 = unsqueeze_1877 = None
        unsqueeze_1878: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_1879: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1878, -1);  unsqueeze_1878 = None
        add_590: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_704, unsqueeze_1879);  mul_704 = unsqueeze_1879 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_227: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_590);  add_590 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_226 = torch.ops.aten.split.Tensor(relu_227, 104, 1)
        getitem_908: "f32[8, 104, 14, 14]" = split_226[0];  split_226 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_227 = torch.ops.aten.split.Tensor(relu_227, 104, 1)
        getitem_913: "f32[8, 104, 14, 14]" = split_227[1];  split_227 = None
        split_228 = torch.ops.aten.split.Tensor(relu_227, 104, 1)
        getitem_918: "f32[8, 104, 14, 14]" = split_228[2];  split_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_229 = torch.ops.aten.split.Tensor(relu_227, 104, 1);  relu_227 = None
        getitem_923: "f32[8, 104, 14, 14]" = split_229[3];  split_229 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_235: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_908, arg326_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_908 = arg326_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1880: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg327_1, -1);  arg327_1 = None
        unsqueeze_1881: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1880, -1);  unsqueeze_1880 = None
        sub_235: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_235, unsqueeze_1881);  convolution_235 = unsqueeze_1881 = None
        add_591: "f32[104]" = torch.ops.aten.add.Tensor(arg328_1, 1e-05);  arg328_1 = None
        sqrt_235: "f32[104]" = torch.ops.aten.sqrt.default(add_591);  add_591 = None
        reciprocal_235: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_235);  sqrt_235 = None
        mul_705: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_235, 1);  reciprocal_235 = None
        unsqueeze_1882: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_705, -1);  mul_705 = None
        unsqueeze_1883: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1882, -1);  unsqueeze_1882 = None
        mul_706: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_1883);  sub_235 = unsqueeze_1883 = None
        unsqueeze_1884: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_1885: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1884, -1);  unsqueeze_1884 = None
        mul_707: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_706, unsqueeze_1885);  mul_706 = unsqueeze_1885 = None
        unsqueeze_1886: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_1887: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1886, -1);  unsqueeze_1886 = None
        add_592: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_707, unsqueeze_1887);  mul_707 = unsqueeze_1887 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_228: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_592);  add_592 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_593: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_228, getitem_913);  getitem_913 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_236: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_593, arg331_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_593 = arg331_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1888: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg332_1, -1);  arg332_1 = None
        unsqueeze_1889: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1888, -1);  unsqueeze_1888 = None
        sub_236: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_236, unsqueeze_1889);  convolution_236 = unsqueeze_1889 = None
        add_594: "f32[104]" = torch.ops.aten.add.Tensor(arg333_1, 1e-05);  arg333_1 = None
        sqrt_236: "f32[104]" = torch.ops.aten.sqrt.default(add_594);  add_594 = None
        reciprocal_236: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_236);  sqrt_236 = None
        mul_708: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_236, 1);  reciprocal_236 = None
        unsqueeze_1890: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_708, -1);  mul_708 = None
        unsqueeze_1891: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1890, -1);  unsqueeze_1890 = None
        mul_709: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_1891);  sub_236 = unsqueeze_1891 = None
        unsqueeze_1892: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_1893: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1892, -1);  unsqueeze_1892 = None
        mul_710: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_709, unsqueeze_1893);  mul_709 = unsqueeze_1893 = None
        unsqueeze_1894: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
        unsqueeze_1895: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1894, -1);  unsqueeze_1894 = None
        add_595: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_710, unsqueeze_1895);  mul_710 = unsqueeze_1895 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_229: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_595);  add_595 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_596: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_229, getitem_918);  getitem_918 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_237: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_596, arg336_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_596 = arg336_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1896: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg337_1, -1);  arg337_1 = None
        unsqueeze_1897: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1896, -1);  unsqueeze_1896 = None
        sub_237: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_237, unsqueeze_1897);  convolution_237 = unsqueeze_1897 = None
        add_597: "f32[104]" = torch.ops.aten.add.Tensor(arg338_1, 1e-05);  arg338_1 = None
        sqrt_237: "f32[104]" = torch.ops.aten.sqrt.default(add_597);  add_597 = None
        reciprocal_237: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_237);  sqrt_237 = None
        mul_711: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_237, 1);  reciprocal_237 = None
        unsqueeze_1898: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_711, -1);  mul_711 = None
        unsqueeze_1899: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1898, -1);  unsqueeze_1898 = None
        mul_712: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_1899);  sub_237 = unsqueeze_1899 = None
        unsqueeze_1900: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_1901: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1900, -1);  unsqueeze_1900 = None
        mul_713: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_712, unsqueeze_1901);  mul_712 = unsqueeze_1901 = None
        unsqueeze_1902: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg340_1, -1);  arg340_1 = None
        unsqueeze_1903: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1902, -1);  unsqueeze_1902 = None
        add_598: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_713, unsqueeze_1903);  mul_713 = unsqueeze_1903 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_230: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_598);  add_598 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_45: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_228, relu_229, relu_230, getitem_923], 1);  relu_228 = relu_229 = relu_230 = getitem_923 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_238: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_45, arg341_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_45 = arg341_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1904: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_1905: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1904, -1);  unsqueeze_1904 = None
        sub_238: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_238, unsqueeze_1905);  convolution_238 = unsqueeze_1905 = None
        add_599: "f32[1024]" = torch.ops.aten.add.Tensor(arg343_1, 1e-05);  arg343_1 = None
        sqrt_238: "f32[1024]" = torch.ops.aten.sqrt.default(add_599);  add_599 = None
        reciprocal_238: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_238);  sqrt_238 = None
        mul_714: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_238, 1);  reciprocal_238 = None
        unsqueeze_1906: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_714, -1);  mul_714 = None
        unsqueeze_1907: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1906, -1);  unsqueeze_1906 = None
        mul_715: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_1907);  sub_238 = unsqueeze_1907 = None
        unsqueeze_1908: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_1909: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1908, -1);  unsqueeze_1908 = None
        mul_716: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_715, unsqueeze_1909);  mul_715 = unsqueeze_1909 = None
        unsqueeze_1910: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_1911: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1910, -1);  unsqueeze_1910 = None
        add_600: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_716, unsqueeze_1911);  mul_716 = unsqueeze_1911 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_601: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_600, relu_226);  add_600 = relu_226 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_231: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_601);  add_601 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_239: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_231, arg346_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg346_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1912: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
        unsqueeze_1913: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1912, -1);  unsqueeze_1912 = None
        sub_239: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_239, unsqueeze_1913);  convolution_239 = unsqueeze_1913 = None
        add_602: "f32[416]" = torch.ops.aten.add.Tensor(arg348_1, 1e-05);  arg348_1 = None
        sqrt_239: "f32[416]" = torch.ops.aten.sqrt.default(add_602);  add_602 = None
        reciprocal_239: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_239);  sqrt_239 = None
        mul_717: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_239, 1);  reciprocal_239 = None
        unsqueeze_1914: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_717, -1);  mul_717 = None
        unsqueeze_1915: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1914, -1);  unsqueeze_1914 = None
        mul_718: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_1915);  sub_239 = unsqueeze_1915 = None
        unsqueeze_1916: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_1917: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1916, -1);  unsqueeze_1916 = None
        mul_719: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_718, unsqueeze_1917);  mul_718 = unsqueeze_1917 = None
        unsqueeze_1918: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
        unsqueeze_1919: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1918, -1);  unsqueeze_1918 = None
        add_603: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_719, unsqueeze_1919);  mul_719 = unsqueeze_1919 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_232: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_603);  add_603 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_231 = torch.ops.aten.split.Tensor(relu_232, 104, 1)
        getitem_928: "f32[8, 104, 14, 14]" = split_231[0];  split_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_232 = torch.ops.aten.split.Tensor(relu_232, 104, 1)
        getitem_933: "f32[8, 104, 14, 14]" = split_232[1];  split_232 = None
        split_233 = torch.ops.aten.split.Tensor(relu_232, 104, 1)
        getitem_938: "f32[8, 104, 14, 14]" = split_233[2];  split_233 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_234 = torch.ops.aten.split.Tensor(relu_232, 104, 1);  relu_232 = None
        getitem_943: "f32[8, 104, 14, 14]" = split_234[3];  split_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_240: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_928, arg351_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_928 = arg351_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1920: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
        unsqueeze_1921: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1920, -1);  unsqueeze_1920 = None
        sub_240: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_240, unsqueeze_1921);  convolution_240 = unsqueeze_1921 = None
        add_604: "f32[104]" = torch.ops.aten.add.Tensor(arg353_1, 1e-05);  arg353_1 = None
        sqrt_240: "f32[104]" = torch.ops.aten.sqrt.default(add_604);  add_604 = None
        reciprocal_240: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_240);  sqrt_240 = None
        mul_720: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_240, 1);  reciprocal_240 = None
        unsqueeze_1922: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_720, -1);  mul_720 = None
        unsqueeze_1923: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1922, -1);  unsqueeze_1922 = None
        mul_721: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_1923);  sub_240 = unsqueeze_1923 = None
        unsqueeze_1924: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_1925: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1924, -1);  unsqueeze_1924 = None
        mul_722: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_721, unsqueeze_1925);  mul_721 = unsqueeze_1925 = None
        unsqueeze_1926: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
        unsqueeze_1927: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1926, -1);  unsqueeze_1926 = None
        add_605: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_722, unsqueeze_1927);  mul_722 = unsqueeze_1927 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_233: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_605);  add_605 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_606: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_233, getitem_933);  getitem_933 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_241: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_606, arg356_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_606 = arg356_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1928: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg357_1, -1);  arg357_1 = None
        unsqueeze_1929: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1928, -1);  unsqueeze_1928 = None
        sub_241: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_241, unsqueeze_1929);  convolution_241 = unsqueeze_1929 = None
        add_607: "f32[104]" = torch.ops.aten.add.Tensor(arg358_1, 1e-05);  arg358_1 = None
        sqrt_241: "f32[104]" = torch.ops.aten.sqrt.default(add_607);  add_607 = None
        reciprocal_241: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_241);  sqrt_241 = None
        mul_723: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_241, 1);  reciprocal_241 = None
        unsqueeze_1930: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_723, -1);  mul_723 = None
        unsqueeze_1931: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1930, -1);  unsqueeze_1930 = None
        mul_724: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_1931);  sub_241 = unsqueeze_1931 = None
        unsqueeze_1932: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_1933: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1932, -1);  unsqueeze_1932 = None
        mul_725: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_724, unsqueeze_1933);  mul_724 = unsqueeze_1933 = None
        unsqueeze_1934: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_1935: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1934, -1);  unsqueeze_1934 = None
        add_608: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_725, unsqueeze_1935);  mul_725 = unsqueeze_1935 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_234: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_608);  add_608 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_609: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_234, getitem_938);  getitem_938 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_242: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_609, arg361_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_609 = arg361_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1936: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg362_1, -1);  arg362_1 = None
        unsqueeze_1937: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1936, -1);  unsqueeze_1936 = None
        sub_242: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_242, unsqueeze_1937);  convolution_242 = unsqueeze_1937 = None
        add_610: "f32[104]" = torch.ops.aten.add.Tensor(arg363_1, 1e-05);  arg363_1 = None
        sqrt_242: "f32[104]" = torch.ops.aten.sqrt.default(add_610);  add_610 = None
        reciprocal_242: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_242);  sqrt_242 = None
        mul_726: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_242, 1);  reciprocal_242 = None
        unsqueeze_1938: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_726, -1);  mul_726 = None
        unsqueeze_1939: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1938, -1);  unsqueeze_1938 = None
        mul_727: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_1939);  sub_242 = unsqueeze_1939 = None
        unsqueeze_1940: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
        unsqueeze_1941: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1940, -1);  unsqueeze_1940 = None
        mul_728: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_727, unsqueeze_1941);  mul_727 = unsqueeze_1941 = None
        unsqueeze_1942: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
        unsqueeze_1943: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1942, -1);  unsqueeze_1942 = None
        add_611: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_728, unsqueeze_1943);  mul_728 = unsqueeze_1943 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_235: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_611);  add_611 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_46: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_233, relu_234, relu_235, getitem_943], 1);  relu_233 = relu_234 = relu_235 = getitem_943 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_243: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_46, arg366_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_46 = arg366_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1944: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg367_1, -1);  arg367_1 = None
        unsqueeze_1945: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1944, -1);  unsqueeze_1944 = None
        sub_243: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_243, unsqueeze_1945);  convolution_243 = unsqueeze_1945 = None
        add_612: "f32[1024]" = torch.ops.aten.add.Tensor(arg368_1, 1e-05);  arg368_1 = None
        sqrt_243: "f32[1024]" = torch.ops.aten.sqrt.default(add_612);  add_612 = None
        reciprocal_243: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_243);  sqrt_243 = None
        mul_729: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_243, 1);  reciprocal_243 = None
        unsqueeze_1946: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_729, -1);  mul_729 = None
        unsqueeze_1947: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1946, -1);  unsqueeze_1946 = None
        mul_730: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_1947);  sub_243 = unsqueeze_1947 = None
        unsqueeze_1948: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg369_1, -1);  arg369_1 = None
        unsqueeze_1949: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1948, -1);  unsqueeze_1948 = None
        mul_731: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_730, unsqueeze_1949);  mul_730 = unsqueeze_1949 = None
        unsqueeze_1950: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg370_1, -1);  arg370_1 = None
        unsqueeze_1951: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1950, -1);  unsqueeze_1950 = None
        add_613: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_731, unsqueeze_1951);  mul_731 = unsqueeze_1951 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_614: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_613, relu_231);  add_613 = relu_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_236: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_614);  add_614 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_244: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_236, arg371_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg371_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1952: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg372_1, -1);  arg372_1 = None
        unsqueeze_1953: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1952, -1);  unsqueeze_1952 = None
        sub_244: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_244, unsqueeze_1953);  convolution_244 = unsqueeze_1953 = None
        add_615: "f32[416]" = torch.ops.aten.add.Tensor(arg373_1, 1e-05);  arg373_1 = None
        sqrt_244: "f32[416]" = torch.ops.aten.sqrt.default(add_615);  add_615 = None
        reciprocal_244: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_244);  sqrt_244 = None
        mul_732: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_244, 1);  reciprocal_244 = None
        unsqueeze_1954: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_732, -1);  mul_732 = None
        unsqueeze_1955: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1954, -1);  unsqueeze_1954 = None
        mul_733: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_1955);  sub_244 = unsqueeze_1955 = None
        unsqueeze_1956: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
        unsqueeze_1957: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1956, -1);  unsqueeze_1956 = None
        mul_734: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_733, unsqueeze_1957);  mul_733 = unsqueeze_1957 = None
        unsqueeze_1958: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg375_1, -1);  arg375_1 = None
        unsqueeze_1959: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1958, -1);  unsqueeze_1958 = None
        add_616: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_734, unsqueeze_1959);  mul_734 = unsqueeze_1959 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_237: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_616);  add_616 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_236 = torch.ops.aten.split.Tensor(relu_237, 104, 1)
        getitem_948: "f32[8, 104, 14, 14]" = split_236[0];  split_236 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_237 = torch.ops.aten.split.Tensor(relu_237, 104, 1)
        getitem_953: "f32[8, 104, 14, 14]" = split_237[1];  split_237 = None
        split_238 = torch.ops.aten.split.Tensor(relu_237, 104, 1)
        getitem_958: "f32[8, 104, 14, 14]" = split_238[2];  split_238 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_239 = torch.ops.aten.split.Tensor(relu_237, 104, 1);  relu_237 = None
        getitem_963: "f32[8, 104, 14, 14]" = split_239[3];  split_239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_245: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_948, arg376_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_948 = arg376_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1960: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg377_1, -1);  arg377_1 = None
        unsqueeze_1961: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1960, -1);  unsqueeze_1960 = None
        sub_245: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_245, unsqueeze_1961);  convolution_245 = unsqueeze_1961 = None
        add_617: "f32[104]" = torch.ops.aten.add.Tensor(arg378_1, 1e-05);  arg378_1 = None
        sqrt_245: "f32[104]" = torch.ops.aten.sqrt.default(add_617);  add_617 = None
        reciprocal_245: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_245);  sqrt_245 = None
        mul_735: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_245, 1);  reciprocal_245 = None
        unsqueeze_1962: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_735, -1);  mul_735 = None
        unsqueeze_1963: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1962, -1);  unsqueeze_1962 = None
        mul_736: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_1963);  sub_245 = unsqueeze_1963 = None
        unsqueeze_1964: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_1965: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1964, -1);  unsqueeze_1964 = None
        mul_737: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_736, unsqueeze_1965);  mul_736 = unsqueeze_1965 = None
        unsqueeze_1966: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
        unsqueeze_1967: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1966, -1);  unsqueeze_1966 = None
        add_618: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_737, unsqueeze_1967);  mul_737 = unsqueeze_1967 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_238: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_618);  add_618 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_619: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_238, getitem_953);  getitem_953 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_246: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_619, arg381_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_619 = arg381_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1968: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg382_1, -1);  arg382_1 = None
        unsqueeze_1969: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1968, -1);  unsqueeze_1968 = None
        sub_246: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_246, unsqueeze_1969);  convolution_246 = unsqueeze_1969 = None
        add_620: "f32[104]" = torch.ops.aten.add.Tensor(arg383_1, 1e-05);  arg383_1 = None
        sqrt_246: "f32[104]" = torch.ops.aten.sqrt.default(add_620);  add_620 = None
        reciprocal_246: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_246);  sqrt_246 = None
        mul_738: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_246, 1);  reciprocal_246 = None
        unsqueeze_1970: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_738, -1);  mul_738 = None
        unsqueeze_1971: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1970, -1);  unsqueeze_1970 = None
        mul_739: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_1971);  sub_246 = unsqueeze_1971 = None
        unsqueeze_1972: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg384_1, -1);  arg384_1 = None
        unsqueeze_1973: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1972, -1);  unsqueeze_1972 = None
        mul_740: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_739, unsqueeze_1973);  mul_739 = unsqueeze_1973 = None
        unsqueeze_1974: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
        unsqueeze_1975: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1974, -1);  unsqueeze_1974 = None
        add_621: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_740, unsqueeze_1975);  mul_740 = unsqueeze_1975 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_239: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_621);  add_621 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_622: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_239, getitem_958);  getitem_958 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_247: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_622, arg386_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_622 = arg386_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_1976: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg387_1, -1);  arg387_1 = None
        unsqueeze_1977: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1976, -1);  unsqueeze_1976 = None
        sub_247: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_247, unsqueeze_1977);  convolution_247 = unsqueeze_1977 = None
        add_623: "f32[104]" = torch.ops.aten.add.Tensor(arg388_1, 1e-05);  arg388_1 = None
        sqrt_247: "f32[104]" = torch.ops.aten.sqrt.default(add_623);  add_623 = None
        reciprocal_247: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_247);  sqrt_247 = None
        mul_741: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_247, 1);  reciprocal_247 = None
        unsqueeze_1978: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_741, -1);  mul_741 = None
        unsqueeze_1979: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1978, -1);  unsqueeze_1978 = None
        mul_742: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_1979);  sub_247 = unsqueeze_1979 = None
        unsqueeze_1980: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg389_1, -1);  arg389_1 = None
        unsqueeze_1981: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1980, -1);  unsqueeze_1980 = None
        mul_743: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_742, unsqueeze_1981);  mul_742 = unsqueeze_1981 = None
        unsqueeze_1982: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg390_1, -1);  arg390_1 = None
        unsqueeze_1983: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1982, -1);  unsqueeze_1982 = None
        add_624: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_743, unsqueeze_1983);  mul_743 = unsqueeze_1983 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_240: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_624);  add_624 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_47: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_238, relu_239, relu_240, getitem_963], 1);  relu_238 = relu_239 = relu_240 = getitem_963 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_248: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_47, arg391_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_47 = arg391_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_1984: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg392_1, -1);  arg392_1 = None
        unsqueeze_1985: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1984, -1);  unsqueeze_1984 = None
        sub_248: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_248, unsqueeze_1985);  convolution_248 = unsqueeze_1985 = None
        add_625: "f32[1024]" = torch.ops.aten.add.Tensor(arg393_1, 1e-05);  arg393_1 = None
        sqrt_248: "f32[1024]" = torch.ops.aten.sqrt.default(add_625);  add_625 = None
        reciprocal_248: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_248);  sqrt_248 = None
        mul_744: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_248, 1);  reciprocal_248 = None
        unsqueeze_1986: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_744, -1);  mul_744 = None
        unsqueeze_1987: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1986, -1);  unsqueeze_1986 = None
        mul_745: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_1987);  sub_248 = unsqueeze_1987 = None
        unsqueeze_1988: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
        unsqueeze_1989: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1988, -1);  unsqueeze_1988 = None
        mul_746: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_745, unsqueeze_1989);  mul_745 = unsqueeze_1989 = None
        unsqueeze_1990: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
        unsqueeze_1991: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1990, -1);  unsqueeze_1990 = None
        add_626: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_746, unsqueeze_1991);  mul_746 = unsqueeze_1991 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_627: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_626, relu_236);  add_626 = relu_236 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_241: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_627);  add_627 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_249: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_241, arg396_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg396_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_1992: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
        unsqueeze_1993: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1992, -1);  unsqueeze_1992 = None
        sub_249: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_249, unsqueeze_1993);  convolution_249 = unsqueeze_1993 = None
        add_628: "f32[416]" = torch.ops.aten.add.Tensor(arg398_1, 1e-05);  arg398_1 = None
        sqrt_249: "f32[416]" = torch.ops.aten.sqrt.default(add_628);  add_628 = None
        reciprocal_249: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_249);  sqrt_249 = None
        mul_747: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_249, 1);  reciprocal_249 = None
        unsqueeze_1994: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_747, -1);  mul_747 = None
        unsqueeze_1995: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1994, -1);  unsqueeze_1994 = None
        mul_748: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_1995);  sub_249 = unsqueeze_1995 = None
        unsqueeze_1996: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg399_1, -1);  arg399_1 = None
        unsqueeze_1997: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1996, -1);  unsqueeze_1996 = None
        mul_749: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_748, unsqueeze_1997);  mul_748 = unsqueeze_1997 = None
        unsqueeze_1998: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg400_1, -1);  arg400_1 = None
        unsqueeze_1999: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1998, -1);  unsqueeze_1998 = None
        add_629: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_749, unsqueeze_1999);  mul_749 = unsqueeze_1999 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_242: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_629);  add_629 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_241 = torch.ops.aten.split.Tensor(relu_242, 104, 1)
        getitem_968: "f32[8, 104, 14, 14]" = split_241[0];  split_241 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_242 = torch.ops.aten.split.Tensor(relu_242, 104, 1)
        getitem_973: "f32[8, 104, 14, 14]" = split_242[1];  split_242 = None
        split_243 = torch.ops.aten.split.Tensor(relu_242, 104, 1)
        getitem_978: "f32[8, 104, 14, 14]" = split_243[2];  split_243 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_244 = torch.ops.aten.split.Tensor(relu_242, 104, 1);  relu_242 = None
        getitem_983: "f32[8, 104, 14, 14]" = split_244[3];  split_244 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_250: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_968, arg401_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_968 = arg401_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2000: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg402_1, -1);  arg402_1 = None
        unsqueeze_2001: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2000, -1);  unsqueeze_2000 = None
        sub_250: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_250, unsqueeze_2001);  convolution_250 = unsqueeze_2001 = None
        add_630: "f32[104]" = torch.ops.aten.add.Tensor(arg403_1, 1e-05);  arg403_1 = None
        sqrt_250: "f32[104]" = torch.ops.aten.sqrt.default(add_630);  add_630 = None
        reciprocal_250: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_250);  sqrt_250 = None
        mul_750: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_250, 1);  reciprocal_250 = None
        unsqueeze_2002: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_750, -1);  mul_750 = None
        unsqueeze_2003: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2002, -1);  unsqueeze_2002 = None
        mul_751: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_2003);  sub_250 = unsqueeze_2003 = None
        unsqueeze_2004: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
        unsqueeze_2005: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2004, -1);  unsqueeze_2004 = None
        mul_752: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_751, unsqueeze_2005);  mul_751 = unsqueeze_2005 = None
        unsqueeze_2006: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg405_1, -1);  arg405_1 = None
        unsqueeze_2007: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2006, -1);  unsqueeze_2006 = None
        add_631: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_752, unsqueeze_2007);  mul_752 = unsqueeze_2007 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_243: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_631);  add_631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_632: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_243, getitem_973);  getitem_973 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_251: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_632, arg406_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_632 = arg406_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2008: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg407_1, -1);  arg407_1 = None
        unsqueeze_2009: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2008, -1);  unsqueeze_2008 = None
        sub_251: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_251, unsqueeze_2009);  convolution_251 = unsqueeze_2009 = None
        add_633: "f32[104]" = torch.ops.aten.add.Tensor(arg408_1, 1e-05);  arg408_1 = None
        sqrt_251: "f32[104]" = torch.ops.aten.sqrt.default(add_633);  add_633 = None
        reciprocal_251: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_251);  sqrt_251 = None
        mul_753: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_251, 1);  reciprocal_251 = None
        unsqueeze_2010: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_753, -1);  mul_753 = None
        unsqueeze_2011: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2010, -1);  unsqueeze_2010 = None
        mul_754: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_2011);  sub_251 = unsqueeze_2011 = None
        unsqueeze_2012: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
        unsqueeze_2013: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2012, -1);  unsqueeze_2012 = None
        mul_755: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_754, unsqueeze_2013);  mul_754 = unsqueeze_2013 = None
        unsqueeze_2014: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg410_1, -1);  arg410_1 = None
        unsqueeze_2015: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2014, -1);  unsqueeze_2014 = None
        add_634: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_755, unsqueeze_2015);  mul_755 = unsqueeze_2015 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_244: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_634);  add_634 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_635: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_244, getitem_978);  getitem_978 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_252: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_635, arg411_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_635 = arg411_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2016: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg412_1, -1);  arg412_1 = None
        unsqueeze_2017: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2016, -1);  unsqueeze_2016 = None
        sub_252: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_252, unsqueeze_2017);  convolution_252 = unsqueeze_2017 = None
        add_636: "f32[104]" = torch.ops.aten.add.Tensor(arg413_1, 1e-05);  arg413_1 = None
        sqrt_252: "f32[104]" = torch.ops.aten.sqrt.default(add_636);  add_636 = None
        reciprocal_252: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_252);  sqrt_252 = None
        mul_756: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_252, 1);  reciprocal_252 = None
        unsqueeze_2018: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_756, -1);  mul_756 = None
        unsqueeze_2019: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2018, -1);  unsqueeze_2018 = None
        mul_757: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_2019);  sub_252 = unsqueeze_2019 = None
        unsqueeze_2020: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_2021: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2020, -1);  unsqueeze_2020 = None
        mul_758: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_757, unsqueeze_2021);  mul_757 = unsqueeze_2021 = None
        unsqueeze_2022: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg415_1, -1);  arg415_1 = None
        unsqueeze_2023: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2022, -1);  unsqueeze_2022 = None
        add_637: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_758, unsqueeze_2023);  mul_758 = unsqueeze_2023 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_245: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_637);  add_637 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_48: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_243, relu_244, relu_245, getitem_983], 1);  relu_243 = relu_244 = relu_245 = getitem_983 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_253: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_48, arg416_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_48 = arg416_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2024: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg417_1, -1);  arg417_1 = None
        unsqueeze_2025: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2024, -1);  unsqueeze_2024 = None
        sub_253: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_253, unsqueeze_2025);  convolution_253 = unsqueeze_2025 = None
        add_638: "f32[1024]" = torch.ops.aten.add.Tensor(arg418_1, 1e-05);  arg418_1 = None
        sqrt_253: "f32[1024]" = torch.ops.aten.sqrt.default(add_638);  add_638 = None
        reciprocal_253: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_253);  sqrt_253 = None
        mul_759: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_253, 1);  reciprocal_253 = None
        unsqueeze_2026: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_759, -1);  mul_759 = None
        unsqueeze_2027: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2026, -1);  unsqueeze_2026 = None
        mul_760: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_2027);  sub_253 = unsqueeze_2027 = None
        unsqueeze_2028: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg419_1, -1);  arg419_1 = None
        unsqueeze_2029: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2028, -1);  unsqueeze_2028 = None
        mul_761: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_760, unsqueeze_2029);  mul_760 = unsqueeze_2029 = None
        unsqueeze_2030: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg420_1, -1);  arg420_1 = None
        unsqueeze_2031: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2030, -1);  unsqueeze_2030 = None
        add_639: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_761, unsqueeze_2031);  mul_761 = unsqueeze_2031 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_640: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_639, relu_241);  add_639 = relu_241 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_246: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_640);  add_640 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_254: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_246, arg421_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg421_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2032: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg422_1, -1);  arg422_1 = None
        unsqueeze_2033: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2032, -1);  unsqueeze_2032 = None
        sub_254: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_254, unsqueeze_2033);  convolution_254 = unsqueeze_2033 = None
        add_641: "f32[416]" = torch.ops.aten.add.Tensor(arg423_1, 1e-05);  arg423_1 = None
        sqrt_254: "f32[416]" = torch.ops.aten.sqrt.default(add_641);  add_641 = None
        reciprocal_254: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_254);  sqrt_254 = None
        mul_762: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_254, 1);  reciprocal_254 = None
        unsqueeze_2034: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_762, -1);  mul_762 = None
        unsqueeze_2035: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2034, -1);  unsqueeze_2034 = None
        mul_763: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_2035);  sub_254 = unsqueeze_2035 = None
        unsqueeze_2036: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg424_1, -1);  arg424_1 = None
        unsqueeze_2037: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2036, -1);  unsqueeze_2036 = None
        mul_764: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_763, unsqueeze_2037);  mul_763 = unsqueeze_2037 = None
        unsqueeze_2038: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg425_1, -1);  arg425_1 = None
        unsqueeze_2039: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2038, -1);  unsqueeze_2038 = None
        add_642: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_764, unsqueeze_2039);  mul_764 = unsqueeze_2039 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_247: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_642);  add_642 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_246 = torch.ops.aten.split.Tensor(relu_247, 104, 1)
        getitem_988: "f32[8, 104, 14, 14]" = split_246[0];  split_246 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_247 = torch.ops.aten.split.Tensor(relu_247, 104, 1)
        getitem_993: "f32[8, 104, 14, 14]" = split_247[1];  split_247 = None
        split_248 = torch.ops.aten.split.Tensor(relu_247, 104, 1)
        getitem_998: "f32[8, 104, 14, 14]" = split_248[2];  split_248 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_249 = torch.ops.aten.split.Tensor(relu_247, 104, 1);  relu_247 = None
        getitem_1003: "f32[8, 104, 14, 14]" = split_249[3];  split_249 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_255: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_988, arg426_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_988 = arg426_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2040: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg427_1, -1);  arg427_1 = None
        unsqueeze_2041: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2040, -1);  unsqueeze_2040 = None
        sub_255: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_255, unsqueeze_2041);  convolution_255 = unsqueeze_2041 = None
        add_643: "f32[104]" = torch.ops.aten.add.Tensor(arg428_1, 1e-05);  arg428_1 = None
        sqrt_255: "f32[104]" = torch.ops.aten.sqrt.default(add_643);  add_643 = None
        reciprocal_255: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_255);  sqrt_255 = None
        mul_765: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_255, 1);  reciprocal_255 = None
        unsqueeze_2042: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_765, -1);  mul_765 = None
        unsqueeze_2043: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2042, -1);  unsqueeze_2042 = None
        mul_766: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_2043);  sub_255 = unsqueeze_2043 = None
        unsqueeze_2044: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg429_1, -1);  arg429_1 = None
        unsqueeze_2045: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2044, -1);  unsqueeze_2044 = None
        mul_767: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_766, unsqueeze_2045);  mul_766 = unsqueeze_2045 = None
        unsqueeze_2046: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
        unsqueeze_2047: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2046, -1);  unsqueeze_2046 = None
        add_644: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_767, unsqueeze_2047);  mul_767 = unsqueeze_2047 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_248: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_644);  add_644 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_645: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_248, getitem_993);  getitem_993 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_256: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_645, arg431_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_645 = arg431_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2048: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg432_1, -1);  arg432_1 = None
        unsqueeze_2049: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2048, -1);  unsqueeze_2048 = None
        sub_256: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_256, unsqueeze_2049);  convolution_256 = unsqueeze_2049 = None
        add_646: "f32[104]" = torch.ops.aten.add.Tensor(arg433_1, 1e-05);  arg433_1 = None
        sqrt_256: "f32[104]" = torch.ops.aten.sqrt.default(add_646);  add_646 = None
        reciprocal_256: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_256);  sqrt_256 = None
        mul_768: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_256, 1);  reciprocal_256 = None
        unsqueeze_2050: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_768, -1);  mul_768 = None
        unsqueeze_2051: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2050, -1);  unsqueeze_2050 = None
        mul_769: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_2051);  sub_256 = unsqueeze_2051 = None
        unsqueeze_2052: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg434_1, -1);  arg434_1 = None
        unsqueeze_2053: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2052, -1);  unsqueeze_2052 = None
        mul_770: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_769, unsqueeze_2053);  mul_769 = unsqueeze_2053 = None
        unsqueeze_2054: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg435_1, -1);  arg435_1 = None
        unsqueeze_2055: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2054, -1);  unsqueeze_2054 = None
        add_647: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_770, unsqueeze_2055);  mul_770 = unsqueeze_2055 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_249: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_647);  add_647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_648: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_249, getitem_998);  getitem_998 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_257: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_648, arg436_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_648 = arg436_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2056: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg437_1, -1);  arg437_1 = None
        unsqueeze_2057: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2056, -1);  unsqueeze_2056 = None
        sub_257: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_257, unsqueeze_2057);  convolution_257 = unsqueeze_2057 = None
        add_649: "f32[104]" = torch.ops.aten.add.Tensor(arg438_1, 1e-05);  arg438_1 = None
        sqrt_257: "f32[104]" = torch.ops.aten.sqrt.default(add_649);  add_649 = None
        reciprocal_257: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_257);  sqrt_257 = None
        mul_771: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_257, 1);  reciprocal_257 = None
        unsqueeze_2058: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_771, -1);  mul_771 = None
        unsqueeze_2059: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2058, -1);  unsqueeze_2058 = None
        mul_772: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_2059);  sub_257 = unsqueeze_2059 = None
        unsqueeze_2060: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg439_1, -1);  arg439_1 = None
        unsqueeze_2061: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2060, -1);  unsqueeze_2060 = None
        mul_773: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_772, unsqueeze_2061);  mul_772 = unsqueeze_2061 = None
        unsqueeze_2062: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg440_1, -1);  arg440_1 = None
        unsqueeze_2063: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2062, -1);  unsqueeze_2062 = None
        add_650: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_773, unsqueeze_2063);  mul_773 = unsqueeze_2063 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_250: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_650);  add_650 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_49: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_248, relu_249, relu_250, getitem_1003], 1);  relu_248 = relu_249 = relu_250 = getitem_1003 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_258: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_49, arg441_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_49 = arg441_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2064: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg442_1, -1);  arg442_1 = None
        unsqueeze_2065: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2064, -1);  unsqueeze_2064 = None
        sub_258: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_258, unsqueeze_2065);  convolution_258 = unsqueeze_2065 = None
        add_651: "f32[1024]" = torch.ops.aten.add.Tensor(arg443_1, 1e-05);  arg443_1 = None
        sqrt_258: "f32[1024]" = torch.ops.aten.sqrt.default(add_651);  add_651 = None
        reciprocal_258: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_258);  sqrt_258 = None
        mul_774: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_258, 1);  reciprocal_258 = None
        unsqueeze_2066: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_774, -1);  mul_774 = None
        unsqueeze_2067: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2066, -1);  unsqueeze_2066 = None
        mul_775: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_2067);  sub_258 = unsqueeze_2067 = None
        unsqueeze_2068: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg444_1, -1);  arg444_1 = None
        unsqueeze_2069: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2068, -1);  unsqueeze_2068 = None
        mul_776: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_775, unsqueeze_2069);  mul_775 = unsqueeze_2069 = None
        unsqueeze_2070: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg445_1, -1);  arg445_1 = None
        unsqueeze_2071: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2070, -1);  unsqueeze_2070 = None
        add_652: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_776, unsqueeze_2071);  mul_776 = unsqueeze_2071 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_653: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_652, relu_246);  add_652 = relu_246 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_251: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_653);  add_653 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_259: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_251, arg446_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg446_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2072: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg447_1, -1);  arg447_1 = None
        unsqueeze_2073: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2072, -1);  unsqueeze_2072 = None
        sub_259: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_259, unsqueeze_2073);  convolution_259 = unsqueeze_2073 = None
        add_654: "f32[416]" = torch.ops.aten.add.Tensor(arg448_1, 1e-05);  arg448_1 = None
        sqrt_259: "f32[416]" = torch.ops.aten.sqrt.default(add_654);  add_654 = None
        reciprocal_259: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_259);  sqrt_259 = None
        mul_777: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_259, 1);  reciprocal_259 = None
        unsqueeze_2074: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_777, -1);  mul_777 = None
        unsqueeze_2075: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2074, -1);  unsqueeze_2074 = None
        mul_778: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_2075);  sub_259 = unsqueeze_2075 = None
        unsqueeze_2076: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
        unsqueeze_2077: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2076, -1);  unsqueeze_2076 = None
        mul_779: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_778, unsqueeze_2077);  mul_778 = unsqueeze_2077 = None
        unsqueeze_2078: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg450_1, -1);  arg450_1 = None
        unsqueeze_2079: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2078, -1);  unsqueeze_2078 = None
        add_655: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_779, unsqueeze_2079);  mul_779 = unsqueeze_2079 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_252: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_655);  add_655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_251 = torch.ops.aten.split.Tensor(relu_252, 104, 1)
        getitem_1008: "f32[8, 104, 14, 14]" = split_251[0];  split_251 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_252 = torch.ops.aten.split.Tensor(relu_252, 104, 1)
        getitem_1013: "f32[8, 104, 14, 14]" = split_252[1];  split_252 = None
        split_253 = torch.ops.aten.split.Tensor(relu_252, 104, 1)
        getitem_1018: "f32[8, 104, 14, 14]" = split_253[2];  split_253 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_254 = torch.ops.aten.split.Tensor(relu_252, 104, 1);  relu_252 = None
        getitem_1023: "f32[8, 104, 14, 14]" = split_254[3];  split_254 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_260: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_1008, arg451_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1008 = arg451_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2080: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg452_1, -1);  arg452_1 = None
        unsqueeze_2081: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2080, -1);  unsqueeze_2080 = None
        sub_260: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_260, unsqueeze_2081);  convolution_260 = unsqueeze_2081 = None
        add_656: "f32[104]" = torch.ops.aten.add.Tensor(arg453_1, 1e-05);  arg453_1 = None
        sqrt_260: "f32[104]" = torch.ops.aten.sqrt.default(add_656);  add_656 = None
        reciprocal_260: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_260);  sqrt_260 = None
        mul_780: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_260, 1);  reciprocal_260 = None
        unsqueeze_2082: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_780, -1);  mul_780 = None
        unsqueeze_2083: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2082, -1);  unsqueeze_2082 = None
        mul_781: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_2083);  sub_260 = unsqueeze_2083 = None
        unsqueeze_2084: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
        unsqueeze_2085: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2084, -1);  unsqueeze_2084 = None
        mul_782: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_781, unsqueeze_2085);  mul_781 = unsqueeze_2085 = None
        unsqueeze_2086: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg455_1, -1);  arg455_1 = None
        unsqueeze_2087: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2086, -1);  unsqueeze_2086 = None
        add_657: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_782, unsqueeze_2087);  mul_782 = unsqueeze_2087 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_253: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_657);  add_657 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_658: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_253, getitem_1013);  getitem_1013 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_261: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_658, arg456_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_658 = arg456_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2088: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg457_1, -1);  arg457_1 = None
        unsqueeze_2089: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2088, -1);  unsqueeze_2088 = None
        sub_261: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_261, unsqueeze_2089);  convolution_261 = unsqueeze_2089 = None
        add_659: "f32[104]" = torch.ops.aten.add.Tensor(arg458_1, 1e-05);  arg458_1 = None
        sqrt_261: "f32[104]" = torch.ops.aten.sqrt.default(add_659);  add_659 = None
        reciprocal_261: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_261);  sqrt_261 = None
        mul_783: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_261, 1);  reciprocal_261 = None
        unsqueeze_2090: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_783, -1);  mul_783 = None
        unsqueeze_2091: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2090, -1);  unsqueeze_2090 = None
        mul_784: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_2091);  sub_261 = unsqueeze_2091 = None
        unsqueeze_2092: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg459_1, -1);  arg459_1 = None
        unsqueeze_2093: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2092, -1);  unsqueeze_2092 = None
        mul_785: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_784, unsqueeze_2093);  mul_784 = unsqueeze_2093 = None
        unsqueeze_2094: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg460_1, -1);  arg460_1 = None
        unsqueeze_2095: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2094, -1);  unsqueeze_2094 = None
        add_660: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_785, unsqueeze_2095);  mul_785 = unsqueeze_2095 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_254: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_660);  add_660 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_661: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_254, getitem_1018);  getitem_1018 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_262: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_661, arg461_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_661 = arg461_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2096: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg462_1, -1);  arg462_1 = None
        unsqueeze_2097: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2096, -1);  unsqueeze_2096 = None
        sub_262: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_262, unsqueeze_2097);  convolution_262 = unsqueeze_2097 = None
        add_662: "f32[104]" = torch.ops.aten.add.Tensor(arg463_1, 1e-05);  arg463_1 = None
        sqrt_262: "f32[104]" = torch.ops.aten.sqrt.default(add_662);  add_662 = None
        reciprocal_262: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_262);  sqrt_262 = None
        mul_786: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_262, 1);  reciprocal_262 = None
        unsqueeze_2098: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_786, -1);  mul_786 = None
        unsqueeze_2099: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2098, -1);  unsqueeze_2098 = None
        mul_787: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_2099);  sub_262 = unsqueeze_2099 = None
        unsqueeze_2100: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
        unsqueeze_2101: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2100, -1);  unsqueeze_2100 = None
        mul_788: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_787, unsqueeze_2101);  mul_787 = unsqueeze_2101 = None
        unsqueeze_2102: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg465_1, -1);  arg465_1 = None
        unsqueeze_2103: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2102, -1);  unsqueeze_2102 = None
        add_663: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_788, unsqueeze_2103);  mul_788 = unsqueeze_2103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_255: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_663);  add_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_50: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_253, relu_254, relu_255, getitem_1023], 1);  relu_253 = relu_254 = relu_255 = getitem_1023 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_263: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_50, arg466_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_50 = arg466_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2104: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg467_1, -1);  arg467_1 = None
        unsqueeze_2105: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2104, -1);  unsqueeze_2104 = None
        sub_263: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_263, unsqueeze_2105);  convolution_263 = unsqueeze_2105 = None
        add_664: "f32[1024]" = torch.ops.aten.add.Tensor(arg468_1, 1e-05);  arg468_1 = None
        sqrt_263: "f32[1024]" = torch.ops.aten.sqrt.default(add_664);  add_664 = None
        reciprocal_263: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_263);  sqrt_263 = None
        mul_789: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_263, 1);  reciprocal_263 = None
        unsqueeze_2106: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_789, -1);  mul_789 = None
        unsqueeze_2107: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2106, -1);  unsqueeze_2106 = None
        mul_790: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_2107);  sub_263 = unsqueeze_2107 = None
        unsqueeze_2108: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
        unsqueeze_2109: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2108, -1);  unsqueeze_2108 = None
        mul_791: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_790, unsqueeze_2109);  mul_790 = unsqueeze_2109 = None
        unsqueeze_2110: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg470_1, -1);  arg470_1 = None
        unsqueeze_2111: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2110, -1);  unsqueeze_2110 = None
        add_665: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_791, unsqueeze_2111);  mul_791 = unsqueeze_2111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_666: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_665, relu_251);  add_665 = relu_251 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_256: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_666);  add_666 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_264: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_256, arg471_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg471_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2112: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg472_1, -1);  arg472_1 = None
        unsqueeze_2113: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2112, -1);  unsqueeze_2112 = None
        sub_264: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_264, unsqueeze_2113);  convolution_264 = unsqueeze_2113 = None
        add_667: "f32[416]" = torch.ops.aten.add.Tensor(arg473_1, 1e-05);  arg473_1 = None
        sqrt_264: "f32[416]" = torch.ops.aten.sqrt.default(add_667);  add_667 = None
        reciprocal_264: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_264);  sqrt_264 = None
        mul_792: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_264, 1);  reciprocal_264 = None
        unsqueeze_2114: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_792, -1);  mul_792 = None
        unsqueeze_2115: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2114, -1);  unsqueeze_2114 = None
        mul_793: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_2115);  sub_264 = unsqueeze_2115 = None
        unsqueeze_2116: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg474_1, -1);  arg474_1 = None
        unsqueeze_2117: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2116, -1);  unsqueeze_2116 = None
        mul_794: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_793, unsqueeze_2117);  mul_793 = unsqueeze_2117 = None
        unsqueeze_2118: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg475_1, -1);  arg475_1 = None
        unsqueeze_2119: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2118, -1);  unsqueeze_2118 = None
        add_668: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_794, unsqueeze_2119);  mul_794 = unsqueeze_2119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_257: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_668);  add_668 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_256 = torch.ops.aten.split.Tensor(relu_257, 104, 1)
        getitem_1028: "f32[8, 104, 14, 14]" = split_256[0];  split_256 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_257 = torch.ops.aten.split.Tensor(relu_257, 104, 1)
        getitem_1033: "f32[8, 104, 14, 14]" = split_257[1];  split_257 = None
        split_258 = torch.ops.aten.split.Tensor(relu_257, 104, 1)
        getitem_1038: "f32[8, 104, 14, 14]" = split_258[2];  split_258 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_259 = torch.ops.aten.split.Tensor(relu_257, 104, 1);  relu_257 = None
        getitem_1043: "f32[8, 104, 14, 14]" = split_259[3];  split_259 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_265: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_1028, arg476_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1028 = arg476_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2120: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg477_1, -1);  arg477_1 = None
        unsqueeze_2121: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2120, -1);  unsqueeze_2120 = None
        sub_265: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_265, unsqueeze_2121);  convolution_265 = unsqueeze_2121 = None
        add_669: "f32[104]" = torch.ops.aten.add.Tensor(arg478_1, 1e-05);  arg478_1 = None
        sqrt_265: "f32[104]" = torch.ops.aten.sqrt.default(add_669);  add_669 = None
        reciprocal_265: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_265);  sqrt_265 = None
        mul_795: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_265, 1);  reciprocal_265 = None
        unsqueeze_2122: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_795, -1);  mul_795 = None
        unsqueeze_2123: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2122, -1);  unsqueeze_2122 = None
        mul_796: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_2123);  sub_265 = unsqueeze_2123 = None
        unsqueeze_2124: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg479_1, -1);  arg479_1 = None
        unsqueeze_2125: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2124, -1);  unsqueeze_2124 = None
        mul_797: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_796, unsqueeze_2125);  mul_796 = unsqueeze_2125 = None
        unsqueeze_2126: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg480_1, -1);  arg480_1 = None
        unsqueeze_2127: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2126, -1);  unsqueeze_2126 = None
        add_670: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_797, unsqueeze_2127);  mul_797 = unsqueeze_2127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_258: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_670);  add_670 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_671: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_258, getitem_1033);  getitem_1033 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_266: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_671, arg481_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_671 = arg481_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2128: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg482_1, -1);  arg482_1 = None
        unsqueeze_2129: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2128, -1);  unsqueeze_2128 = None
        sub_266: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_266, unsqueeze_2129);  convolution_266 = unsqueeze_2129 = None
        add_672: "f32[104]" = torch.ops.aten.add.Tensor(arg483_1, 1e-05);  arg483_1 = None
        sqrt_266: "f32[104]" = torch.ops.aten.sqrt.default(add_672);  add_672 = None
        reciprocal_266: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_266);  sqrt_266 = None
        mul_798: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_266, 1);  reciprocal_266 = None
        unsqueeze_2130: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_798, -1);  mul_798 = None
        unsqueeze_2131: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2130, -1);  unsqueeze_2130 = None
        mul_799: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_2131);  sub_266 = unsqueeze_2131 = None
        unsqueeze_2132: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg484_1, -1);  arg484_1 = None
        unsqueeze_2133: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2132, -1);  unsqueeze_2132 = None
        mul_800: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_799, unsqueeze_2133);  mul_799 = unsqueeze_2133 = None
        unsqueeze_2134: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg485_1, -1);  arg485_1 = None
        unsqueeze_2135: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2134, -1);  unsqueeze_2134 = None
        add_673: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_800, unsqueeze_2135);  mul_800 = unsqueeze_2135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_259: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_673);  add_673 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_674: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_259, getitem_1038);  getitem_1038 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_267: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_674, arg486_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_674 = arg486_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2136: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg487_1, -1);  arg487_1 = None
        unsqueeze_2137: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2136, -1);  unsqueeze_2136 = None
        sub_267: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_267, unsqueeze_2137);  convolution_267 = unsqueeze_2137 = None
        add_675: "f32[104]" = torch.ops.aten.add.Tensor(arg488_1, 1e-05);  arg488_1 = None
        sqrt_267: "f32[104]" = torch.ops.aten.sqrt.default(add_675);  add_675 = None
        reciprocal_267: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_267);  sqrt_267 = None
        mul_801: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_267, 1);  reciprocal_267 = None
        unsqueeze_2138: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_801, -1);  mul_801 = None
        unsqueeze_2139: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2138, -1);  unsqueeze_2138 = None
        mul_802: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_2139);  sub_267 = unsqueeze_2139 = None
        unsqueeze_2140: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg489_1, -1);  arg489_1 = None
        unsqueeze_2141: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2140, -1);  unsqueeze_2140 = None
        mul_803: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_802, unsqueeze_2141);  mul_802 = unsqueeze_2141 = None
        unsqueeze_2142: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg490_1, -1);  arg490_1 = None
        unsqueeze_2143: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2142, -1);  unsqueeze_2142 = None
        add_676: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_803, unsqueeze_2143);  mul_803 = unsqueeze_2143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_260: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_676);  add_676 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_51: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_258, relu_259, relu_260, getitem_1043], 1);  relu_258 = relu_259 = relu_260 = getitem_1043 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_268: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_51, arg491_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_51 = arg491_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2144: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg492_1, -1);  arg492_1 = None
        unsqueeze_2145: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2144, -1);  unsqueeze_2144 = None
        sub_268: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_268, unsqueeze_2145);  convolution_268 = unsqueeze_2145 = None
        add_677: "f32[1024]" = torch.ops.aten.add.Tensor(arg493_1, 1e-05);  arg493_1 = None
        sqrt_268: "f32[1024]" = torch.ops.aten.sqrt.default(add_677);  add_677 = None
        reciprocal_268: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_268);  sqrt_268 = None
        mul_804: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_268, 1);  reciprocal_268 = None
        unsqueeze_2146: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_804, -1);  mul_804 = None
        unsqueeze_2147: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2146, -1);  unsqueeze_2146 = None
        mul_805: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_2147);  sub_268 = unsqueeze_2147 = None
        unsqueeze_2148: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg494_1, -1);  arg494_1 = None
        unsqueeze_2149: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2148, -1);  unsqueeze_2148 = None
        mul_806: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_805, unsqueeze_2149);  mul_805 = unsqueeze_2149 = None
        unsqueeze_2150: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg495_1, -1);  arg495_1 = None
        unsqueeze_2151: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2150, -1);  unsqueeze_2150 = None
        add_678: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_806, unsqueeze_2151);  mul_806 = unsqueeze_2151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_679: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_678, relu_256);  add_678 = relu_256 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_261: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_679);  add_679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_269: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_261, arg496_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg496_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2152: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg497_1, -1);  arg497_1 = None
        unsqueeze_2153: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2152, -1);  unsqueeze_2152 = None
        sub_269: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_269, unsqueeze_2153);  convolution_269 = unsqueeze_2153 = None
        add_680: "f32[416]" = torch.ops.aten.add.Tensor(arg498_1, 1e-05);  arg498_1 = None
        sqrt_269: "f32[416]" = torch.ops.aten.sqrt.default(add_680);  add_680 = None
        reciprocal_269: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_269);  sqrt_269 = None
        mul_807: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_269, 1);  reciprocal_269 = None
        unsqueeze_2154: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_807, -1);  mul_807 = None
        unsqueeze_2155: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2154, -1);  unsqueeze_2154 = None
        mul_808: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_2155);  sub_269 = unsqueeze_2155 = None
        unsqueeze_2156: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg499_1, -1);  arg499_1 = None
        unsqueeze_2157: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2156, -1);  unsqueeze_2156 = None
        mul_809: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_808, unsqueeze_2157);  mul_808 = unsqueeze_2157 = None
        unsqueeze_2158: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg500_1, -1);  arg500_1 = None
        unsqueeze_2159: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2158, -1);  unsqueeze_2158 = None
        add_681: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_809, unsqueeze_2159);  mul_809 = unsqueeze_2159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_262: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_681);  add_681 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_261 = torch.ops.aten.split.Tensor(relu_262, 104, 1)
        getitem_1048: "f32[8, 104, 14, 14]" = split_261[0];  split_261 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_262 = torch.ops.aten.split.Tensor(relu_262, 104, 1)
        getitem_1053: "f32[8, 104, 14, 14]" = split_262[1];  split_262 = None
        split_263 = torch.ops.aten.split.Tensor(relu_262, 104, 1)
        getitem_1058: "f32[8, 104, 14, 14]" = split_263[2];  split_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_264 = torch.ops.aten.split.Tensor(relu_262, 104, 1);  relu_262 = None
        getitem_1063: "f32[8, 104, 14, 14]" = split_264[3];  split_264 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_270: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_1048, arg501_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1048 = arg501_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2160: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg502_1, -1);  arg502_1 = None
        unsqueeze_2161: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2160, -1);  unsqueeze_2160 = None
        sub_270: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_270, unsqueeze_2161);  convolution_270 = unsqueeze_2161 = None
        add_682: "f32[104]" = torch.ops.aten.add.Tensor(arg503_1, 1e-05);  arg503_1 = None
        sqrt_270: "f32[104]" = torch.ops.aten.sqrt.default(add_682);  add_682 = None
        reciprocal_270: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_270);  sqrt_270 = None
        mul_810: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_270, 1);  reciprocal_270 = None
        unsqueeze_2162: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_810, -1);  mul_810 = None
        unsqueeze_2163: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2162, -1);  unsqueeze_2162 = None
        mul_811: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_2163);  sub_270 = unsqueeze_2163 = None
        unsqueeze_2164: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg504_1, -1);  arg504_1 = None
        unsqueeze_2165: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2164, -1);  unsqueeze_2164 = None
        mul_812: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_811, unsqueeze_2165);  mul_811 = unsqueeze_2165 = None
        unsqueeze_2166: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg505_1, -1);  arg505_1 = None
        unsqueeze_2167: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2166, -1);  unsqueeze_2166 = None
        add_683: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_812, unsqueeze_2167);  mul_812 = unsqueeze_2167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_263: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_683);  add_683 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_684: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_263, getitem_1053);  getitem_1053 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_271: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_684, arg506_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_684 = arg506_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2168: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg507_1, -1);  arg507_1 = None
        unsqueeze_2169: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2168, -1);  unsqueeze_2168 = None
        sub_271: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_271, unsqueeze_2169);  convolution_271 = unsqueeze_2169 = None
        add_685: "f32[104]" = torch.ops.aten.add.Tensor(arg508_1, 1e-05);  arg508_1 = None
        sqrt_271: "f32[104]" = torch.ops.aten.sqrt.default(add_685);  add_685 = None
        reciprocal_271: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_271);  sqrt_271 = None
        mul_813: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_271, 1);  reciprocal_271 = None
        unsqueeze_2170: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_813, -1);  mul_813 = None
        unsqueeze_2171: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2170, -1);  unsqueeze_2170 = None
        mul_814: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_2171);  sub_271 = unsqueeze_2171 = None
        unsqueeze_2172: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg509_1, -1);  arg509_1 = None
        unsqueeze_2173: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2172, -1);  unsqueeze_2172 = None
        mul_815: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_814, unsqueeze_2173);  mul_814 = unsqueeze_2173 = None
        unsqueeze_2174: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg510_1, -1);  arg510_1 = None
        unsqueeze_2175: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2174, -1);  unsqueeze_2174 = None
        add_686: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_815, unsqueeze_2175);  mul_815 = unsqueeze_2175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_264: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_686);  add_686 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_687: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_264, getitem_1058);  getitem_1058 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_272: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_687, arg511_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_687 = arg511_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2176: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg512_1, -1);  arg512_1 = None
        unsqueeze_2177: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2176, -1);  unsqueeze_2176 = None
        sub_272: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_272, unsqueeze_2177);  convolution_272 = unsqueeze_2177 = None
        add_688: "f32[104]" = torch.ops.aten.add.Tensor(arg513_1, 1e-05);  arg513_1 = None
        sqrt_272: "f32[104]" = torch.ops.aten.sqrt.default(add_688);  add_688 = None
        reciprocal_272: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_272);  sqrt_272 = None
        mul_816: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_272, 1);  reciprocal_272 = None
        unsqueeze_2178: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_816, -1);  mul_816 = None
        unsqueeze_2179: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2178, -1);  unsqueeze_2178 = None
        mul_817: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_2179);  sub_272 = unsqueeze_2179 = None
        unsqueeze_2180: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg514_1, -1);  arg514_1 = None
        unsqueeze_2181: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2180, -1);  unsqueeze_2180 = None
        mul_818: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_817, unsqueeze_2181);  mul_817 = unsqueeze_2181 = None
        unsqueeze_2182: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg515_1, -1);  arg515_1 = None
        unsqueeze_2183: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2182, -1);  unsqueeze_2182 = None
        add_689: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_818, unsqueeze_2183);  mul_818 = unsqueeze_2183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_265: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_689);  add_689 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_52: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_263, relu_264, relu_265, getitem_1063], 1);  relu_263 = relu_264 = relu_265 = getitem_1063 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_273: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_52, arg516_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_52 = arg516_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2184: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg517_1, -1);  arg517_1 = None
        unsqueeze_2185: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2184, -1);  unsqueeze_2184 = None
        sub_273: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_273, unsqueeze_2185);  convolution_273 = unsqueeze_2185 = None
        add_690: "f32[1024]" = torch.ops.aten.add.Tensor(arg518_1, 1e-05);  arg518_1 = None
        sqrt_273: "f32[1024]" = torch.ops.aten.sqrt.default(add_690);  add_690 = None
        reciprocal_273: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_273);  sqrt_273 = None
        mul_819: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_273, 1);  reciprocal_273 = None
        unsqueeze_2186: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_819, -1);  mul_819 = None
        unsqueeze_2187: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2186, -1);  unsqueeze_2186 = None
        mul_820: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_2187);  sub_273 = unsqueeze_2187 = None
        unsqueeze_2188: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg519_1, -1);  arg519_1 = None
        unsqueeze_2189: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2188, -1);  unsqueeze_2188 = None
        mul_821: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_820, unsqueeze_2189);  mul_820 = unsqueeze_2189 = None
        unsqueeze_2190: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg520_1, -1);  arg520_1 = None
        unsqueeze_2191: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2190, -1);  unsqueeze_2190 = None
        add_691: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_821, unsqueeze_2191);  mul_821 = unsqueeze_2191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_692: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_691, relu_261);  add_691 = relu_261 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_266: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_692);  add_692 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_274: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_266, arg521_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg521_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2192: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg522_1, -1);  arg522_1 = None
        unsqueeze_2193: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2192, -1);  unsqueeze_2192 = None
        sub_274: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_274, unsqueeze_2193);  convolution_274 = unsqueeze_2193 = None
        add_693: "f32[416]" = torch.ops.aten.add.Tensor(arg523_1, 1e-05);  arg523_1 = None
        sqrt_274: "f32[416]" = torch.ops.aten.sqrt.default(add_693);  add_693 = None
        reciprocal_274: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_274);  sqrt_274 = None
        mul_822: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_274, 1);  reciprocal_274 = None
        unsqueeze_2194: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_822, -1);  mul_822 = None
        unsqueeze_2195: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2194, -1);  unsqueeze_2194 = None
        mul_823: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_2195);  sub_274 = unsqueeze_2195 = None
        unsqueeze_2196: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg524_1, -1);  arg524_1 = None
        unsqueeze_2197: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2196, -1);  unsqueeze_2196 = None
        mul_824: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_823, unsqueeze_2197);  mul_823 = unsqueeze_2197 = None
        unsqueeze_2198: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg525_1, -1);  arg525_1 = None
        unsqueeze_2199: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2198, -1);  unsqueeze_2198 = None
        add_694: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_824, unsqueeze_2199);  mul_824 = unsqueeze_2199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_267: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_694);  add_694 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_266 = torch.ops.aten.split.Tensor(relu_267, 104, 1)
        getitem_1068: "f32[8, 104, 14, 14]" = split_266[0];  split_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_267 = torch.ops.aten.split.Tensor(relu_267, 104, 1)
        getitem_1073: "f32[8, 104, 14, 14]" = split_267[1];  split_267 = None
        split_268 = torch.ops.aten.split.Tensor(relu_267, 104, 1)
        getitem_1078: "f32[8, 104, 14, 14]" = split_268[2];  split_268 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_269 = torch.ops.aten.split.Tensor(relu_267, 104, 1);  relu_267 = None
        getitem_1083: "f32[8, 104, 14, 14]" = split_269[3];  split_269 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_275: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_1068, arg526_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1068 = arg526_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2200: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg527_1, -1);  arg527_1 = None
        unsqueeze_2201: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2200, -1);  unsqueeze_2200 = None
        sub_275: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_275, unsqueeze_2201);  convolution_275 = unsqueeze_2201 = None
        add_695: "f32[104]" = torch.ops.aten.add.Tensor(arg528_1, 1e-05);  arg528_1 = None
        sqrt_275: "f32[104]" = torch.ops.aten.sqrt.default(add_695);  add_695 = None
        reciprocal_275: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_275);  sqrt_275 = None
        mul_825: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_275, 1);  reciprocal_275 = None
        unsqueeze_2202: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_825, -1);  mul_825 = None
        unsqueeze_2203: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2202, -1);  unsqueeze_2202 = None
        mul_826: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_2203);  sub_275 = unsqueeze_2203 = None
        unsqueeze_2204: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg529_1, -1);  arg529_1 = None
        unsqueeze_2205: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2204, -1);  unsqueeze_2204 = None
        mul_827: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_826, unsqueeze_2205);  mul_826 = unsqueeze_2205 = None
        unsqueeze_2206: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg530_1, -1);  arg530_1 = None
        unsqueeze_2207: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2206, -1);  unsqueeze_2206 = None
        add_696: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_827, unsqueeze_2207);  mul_827 = unsqueeze_2207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_268: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_696);  add_696 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_697: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_268, getitem_1073);  getitem_1073 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_276: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_697, arg531_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_697 = arg531_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2208: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg532_1, -1);  arg532_1 = None
        unsqueeze_2209: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2208, -1);  unsqueeze_2208 = None
        sub_276: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_276, unsqueeze_2209);  convolution_276 = unsqueeze_2209 = None
        add_698: "f32[104]" = torch.ops.aten.add.Tensor(arg533_1, 1e-05);  arg533_1 = None
        sqrt_276: "f32[104]" = torch.ops.aten.sqrt.default(add_698);  add_698 = None
        reciprocal_276: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_276);  sqrt_276 = None
        mul_828: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_276, 1);  reciprocal_276 = None
        unsqueeze_2210: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_828, -1);  mul_828 = None
        unsqueeze_2211: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2210, -1);  unsqueeze_2210 = None
        mul_829: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_2211);  sub_276 = unsqueeze_2211 = None
        unsqueeze_2212: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg534_1, -1);  arg534_1 = None
        unsqueeze_2213: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2212, -1);  unsqueeze_2212 = None
        mul_830: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_829, unsqueeze_2213);  mul_829 = unsqueeze_2213 = None
        unsqueeze_2214: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg535_1, -1);  arg535_1 = None
        unsqueeze_2215: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2214, -1);  unsqueeze_2214 = None
        add_699: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_830, unsqueeze_2215);  mul_830 = unsqueeze_2215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_269: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_699);  add_699 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_700: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_269, getitem_1078);  getitem_1078 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_277: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_700, arg536_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_700 = arg536_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2216: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg537_1, -1);  arg537_1 = None
        unsqueeze_2217: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2216, -1);  unsqueeze_2216 = None
        sub_277: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_277, unsqueeze_2217);  convolution_277 = unsqueeze_2217 = None
        add_701: "f32[104]" = torch.ops.aten.add.Tensor(arg538_1, 1e-05);  arg538_1 = None
        sqrt_277: "f32[104]" = torch.ops.aten.sqrt.default(add_701);  add_701 = None
        reciprocal_277: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_277);  sqrt_277 = None
        mul_831: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_277, 1);  reciprocal_277 = None
        unsqueeze_2218: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_831, -1);  mul_831 = None
        unsqueeze_2219: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2218, -1);  unsqueeze_2218 = None
        mul_832: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_2219);  sub_277 = unsqueeze_2219 = None
        unsqueeze_2220: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg539_1, -1);  arg539_1 = None
        unsqueeze_2221: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2220, -1);  unsqueeze_2220 = None
        mul_833: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_832, unsqueeze_2221);  mul_832 = unsqueeze_2221 = None
        unsqueeze_2222: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg540_1, -1);  arg540_1 = None
        unsqueeze_2223: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2222, -1);  unsqueeze_2222 = None
        add_702: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_833, unsqueeze_2223);  mul_833 = unsqueeze_2223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_270: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_702);  add_702 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_53: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_268, relu_269, relu_270, getitem_1083], 1);  relu_268 = relu_269 = relu_270 = getitem_1083 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_278: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_53, arg541_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_53 = arg541_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2224: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg542_1, -1);  arg542_1 = None
        unsqueeze_2225: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2224, -1);  unsqueeze_2224 = None
        sub_278: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_278, unsqueeze_2225);  convolution_278 = unsqueeze_2225 = None
        add_703: "f32[1024]" = torch.ops.aten.add.Tensor(arg543_1, 1e-05);  arg543_1 = None
        sqrt_278: "f32[1024]" = torch.ops.aten.sqrt.default(add_703);  add_703 = None
        reciprocal_278: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_278);  sqrt_278 = None
        mul_834: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_278, 1);  reciprocal_278 = None
        unsqueeze_2226: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_834, -1);  mul_834 = None
        unsqueeze_2227: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2226, -1);  unsqueeze_2226 = None
        mul_835: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_2227);  sub_278 = unsqueeze_2227 = None
        unsqueeze_2228: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg544_1, -1);  arg544_1 = None
        unsqueeze_2229: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2228, -1);  unsqueeze_2228 = None
        mul_836: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_835, unsqueeze_2229);  mul_835 = unsqueeze_2229 = None
        unsqueeze_2230: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg545_1, -1);  arg545_1 = None
        unsqueeze_2231: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2230, -1);  unsqueeze_2230 = None
        add_704: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_836, unsqueeze_2231);  mul_836 = unsqueeze_2231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_705: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_704, relu_266);  add_704 = relu_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_271: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_705);  add_705 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_279: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_271, arg546_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg546_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2232: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg547_1, -1);  arg547_1 = None
        unsqueeze_2233: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2232, -1);  unsqueeze_2232 = None
        sub_279: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_279, unsqueeze_2233);  convolution_279 = unsqueeze_2233 = None
        add_706: "f32[416]" = torch.ops.aten.add.Tensor(arg548_1, 1e-05);  arg548_1 = None
        sqrt_279: "f32[416]" = torch.ops.aten.sqrt.default(add_706);  add_706 = None
        reciprocal_279: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_279);  sqrt_279 = None
        mul_837: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_279, 1);  reciprocal_279 = None
        unsqueeze_2234: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_837, -1);  mul_837 = None
        unsqueeze_2235: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2234, -1);  unsqueeze_2234 = None
        mul_838: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_2235);  sub_279 = unsqueeze_2235 = None
        unsqueeze_2236: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg549_1, -1);  arg549_1 = None
        unsqueeze_2237: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2236, -1);  unsqueeze_2236 = None
        mul_839: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_838, unsqueeze_2237);  mul_838 = unsqueeze_2237 = None
        unsqueeze_2238: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg550_1, -1);  arg550_1 = None
        unsqueeze_2239: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2238, -1);  unsqueeze_2238 = None
        add_707: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_839, unsqueeze_2239);  mul_839 = unsqueeze_2239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_272: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_707);  add_707 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_271 = torch.ops.aten.split.Tensor(relu_272, 104, 1)
        getitem_1088: "f32[8, 104, 14, 14]" = split_271[0];  split_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_272 = torch.ops.aten.split.Tensor(relu_272, 104, 1)
        getitem_1093: "f32[8, 104, 14, 14]" = split_272[1];  split_272 = None
        split_273 = torch.ops.aten.split.Tensor(relu_272, 104, 1)
        getitem_1098: "f32[8, 104, 14, 14]" = split_273[2];  split_273 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_274 = torch.ops.aten.split.Tensor(relu_272, 104, 1);  relu_272 = None
        getitem_1103: "f32[8, 104, 14, 14]" = split_274[3];  split_274 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_280: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_1088, arg551_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1088 = arg551_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2240: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg552_1, -1);  arg552_1 = None
        unsqueeze_2241: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2240, -1);  unsqueeze_2240 = None
        sub_280: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_280, unsqueeze_2241);  convolution_280 = unsqueeze_2241 = None
        add_708: "f32[104]" = torch.ops.aten.add.Tensor(arg553_1, 1e-05);  arg553_1 = None
        sqrt_280: "f32[104]" = torch.ops.aten.sqrt.default(add_708);  add_708 = None
        reciprocal_280: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_280);  sqrt_280 = None
        mul_840: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_280, 1);  reciprocal_280 = None
        unsqueeze_2242: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_840, -1);  mul_840 = None
        unsqueeze_2243: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2242, -1);  unsqueeze_2242 = None
        mul_841: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_2243);  sub_280 = unsqueeze_2243 = None
        unsqueeze_2244: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg554_1, -1);  arg554_1 = None
        unsqueeze_2245: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2244, -1);  unsqueeze_2244 = None
        mul_842: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_841, unsqueeze_2245);  mul_841 = unsqueeze_2245 = None
        unsqueeze_2246: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg555_1, -1);  arg555_1 = None
        unsqueeze_2247: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2246, -1);  unsqueeze_2246 = None
        add_709: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_842, unsqueeze_2247);  mul_842 = unsqueeze_2247 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_273: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_709);  add_709 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_710: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_273, getitem_1093);  getitem_1093 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_281: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_710, arg556_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_710 = arg556_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2248: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg557_1, -1);  arg557_1 = None
        unsqueeze_2249: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2248, -1);  unsqueeze_2248 = None
        sub_281: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_281, unsqueeze_2249);  convolution_281 = unsqueeze_2249 = None
        add_711: "f32[104]" = torch.ops.aten.add.Tensor(arg558_1, 1e-05);  arg558_1 = None
        sqrt_281: "f32[104]" = torch.ops.aten.sqrt.default(add_711);  add_711 = None
        reciprocal_281: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_281);  sqrt_281 = None
        mul_843: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_281, 1);  reciprocal_281 = None
        unsqueeze_2250: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_843, -1);  mul_843 = None
        unsqueeze_2251: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2250, -1);  unsqueeze_2250 = None
        mul_844: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_2251);  sub_281 = unsqueeze_2251 = None
        unsqueeze_2252: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg559_1, -1);  arg559_1 = None
        unsqueeze_2253: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2252, -1);  unsqueeze_2252 = None
        mul_845: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_844, unsqueeze_2253);  mul_844 = unsqueeze_2253 = None
        unsqueeze_2254: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg560_1, -1);  arg560_1 = None
        unsqueeze_2255: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2254, -1);  unsqueeze_2254 = None
        add_712: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_845, unsqueeze_2255);  mul_845 = unsqueeze_2255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_274: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_712);  add_712 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_713: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_274, getitem_1098);  getitem_1098 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_282: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_713, arg561_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_713 = arg561_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2256: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg562_1, -1);  arg562_1 = None
        unsqueeze_2257: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2256, -1);  unsqueeze_2256 = None
        sub_282: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_282, unsqueeze_2257);  convolution_282 = unsqueeze_2257 = None
        add_714: "f32[104]" = torch.ops.aten.add.Tensor(arg563_1, 1e-05);  arg563_1 = None
        sqrt_282: "f32[104]" = torch.ops.aten.sqrt.default(add_714);  add_714 = None
        reciprocal_282: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_282);  sqrt_282 = None
        mul_846: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_282, 1);  reciprocal_282 = None
        unsqueeze_2258: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_846, -1);  mul_846 = None
        unsqueeze_2259: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2258, -1);  unsqueeze_2258 = None
        mul_847: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_2259);  sub_282 = unsqueeze_2259 = None
        unsqueeze_2260: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg564_1, -1);  arg564_1 = None
        unsqueeze_2261: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2260, -1);  unsqueeze_2260 = None
        mul_848: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_847, unsqueeze_2261);  mul_847 = unsqueeze_2261 = None
        unsqueeze_2262: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg565_1, -1);  arg565_1 = None
        unsqueeze_2263: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2262, -1);  unsqueeze_2262 = None
        add_715: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_848, unsqueeze_2263);  mul_848 = unsqueeze_2263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_275: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_715);  add_715 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_54: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_273, relu_274, relu_275, getitem_1103], 1);  relu_273 = relu_274 = relu_275 = getitem_1103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_283: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_54, arg566_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_54 = arg566_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2264: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg567_1, -1);  arg567_1 = None
        unsqueeze_2265: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2264, -1);  unsqueeze_2264 = None
        sub_283: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_283, unsqueeze_2265);  convolution_283 = unsqueeze_2265 = None
        add_716: "f32[1024]" = torch.ops.aten.add.Tensor(arg568_1, 1e-05);  arg568_1 = None
        sqrt_283: "f32[1024]" = torch.ops.aten.sqrt.default(add_716);  add_716 = None
        reciprocal_283: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_283);  sqrt_283 = None
        mul_849: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_283, 1);  reciprocal_283 = None
        unsqueeze_2266: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_849, -1);  mul_849 = None
        unsqueeze_2267: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2266, -1);  unsqueeze_2266 = None
        mul_850: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_2267);  sub_283 = unsqueeze_2267 = None
        unsqueeze_2268: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg569_1, -1);  arg569_1 = None
        unsqueeze_2269: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2268, -1);  unsqueeze_2268 = None
        mul_851: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_850, unsqueeze_2269);  mul_850 = unsqueeze_2269 = None
        unsqueeze_2270: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg570_1, -1);  arg570_1 = None
        unsqueeze_2271: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2270, -1);  unsqueeze_2270 = None
        add_717: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_851, unsqueeze_2271);  mul_851 = unsqueeze_2271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_718: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_717, relu_271);  add_717 = relu_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_276: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_718);  add_718 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_284: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_276, arg571_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg571_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2272: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg572_1, -1);  arg572_1 = None
        unsqueeze_2273: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2272, -1);  unsqueeze_2272 = None
        sub_284: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_284, unsqueeze_2273);  convolution_284 = unsqueeze_2273 = None
        add_719: "f32[416]" = torch.ops.aten.add.Tensor(arg573_1, 1e-05);  arg573_1 = None
        sqrt_284: "f32[416]" = torch.ops.aten.sqrt.default(add_719);  add_719 = None
        reciprocal_284: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_284);  sqrt_284 = None
        mul_852: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_284, 1);  reciprocal_284 = None
        unsqueeze_2274: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_852, -1);  mul_852 = None
        unsqueeze_2275: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2274, -1);  unsqueeze_2274 = None
        mul_853: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_2275);  sub_284 = unsqueeze_2275 = None
        unsqueeze_2276: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg574_1, -1);  arg574_1 = None
        unsqueeze_2277: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2276, -1);  unsqueeze_2276 = None
        mul_854: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_853, unsqueeze_2277);  mul_853 = unsqueeze_2277 = None
        unsqueeze_2278: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg575_1, -1);  arg575_1 = None
        unsqueeze_2279: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2278, -1);  unsqueeze_2278 = None
        add_720: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_854, unsqueeze_2279);  mul_854 = unsqueeze_2279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_277: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_720);  add_720 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_276 = torch.ops.aten.split.Tensor(relu_277, 104, 1)
        getitem_1108: "f32[8, 104, 14, 14]" = split_276[0];  split_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_277 = torch.ops.aten.split.Tensor(relu_277, 104, 1)
        getitem_1113: "f32[8, 104, 14, 14]" = split_277[1];  split_277 = None
        split_278 = torch.ops.aten.split.Tensor(relu_277, 104, 1)
        getitem_1118: "f32[8, 104, 14, 14]" = split_278[2];  split_278 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_279 = torch.ops.aten.split.Tensor(relu_277, 104, 1);  relu_277 = None
        getitem_1123: "f32[8, 104, 14, 14]" = split_279[3];  split_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_285: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_1108, arg576_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1108 = arg576_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2280: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg577_1, -1);  arg577_1 = None
        unsqueeze_2281: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2280, -1);  unsqueeze_2280 = None
        sub_285: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_285, unsqueeze_2281);  convolution_285 = unsqueeze_2281 = None
        add_721: "f32[104]" = torch.ops.aten.add.Tensor(arg578_1, 1e-05);  arg578_1 = None
        sqrt_285: "f32[104]" = torch.ops.aten.sqrt.default(add_721);  add_721 = None
        reciprocal_285: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_285);  sqrt_285 = None
        mul_855: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_285, 1);  reciprocal_285 = None
        unsqueeze_2282: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_855, -1);  mul_855 = None
        unsqueeze_2283: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2282, -1);  unsqueeze_2282 = None
        mul_856: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_2283);  sub_285 = unsqueeze_2283 = None
        unsqueeze_2284: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg579_1, -1);  arg579_1 = None
        unsqueeze_2285: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2284, -1);  unsqueeze_2284 = None
        mul_857: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_856, unsqueeze_2285);  mul_856 = unsqueeze_2285 = None
        unsqueeze_2286: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg580_1, -1);  arg580_1 = None
        unsqueeze_2287: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2286, -1);  unsqueeze_2286 = None
        add_722: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_857, unsqueeze_2287);  mul_857 = unsqueeze_2287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_278: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_722);  add_722 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_723: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_278, getitem_1113);  getitem_1113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_286: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_723, arg581_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_723 = arg581_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2288: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg582_1, -1);  arg582_1 = None
        unsqueeze_2289: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2288, -1);  unsqueeze_2288 = None
        sub_286: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_286, unsqueeze_2289);  convolution_286 = unsqueeze_2289 = None
        add_724: "f32[104]" = torch.ops.aten.add.Tensor(arg583_1, 1e-05);  arg583_1 = None
        sqrt_286: "f32[104]" = torch.ops.aten.sqrt.default(add_724);  add_724 = None
        reciprocal_286: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_286);  sqrt_286 = None
        mul_858: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_286, 1);  reciprocal_286 = None
        unsqueeze_2290: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_858, -1);  mul_858 = None
        unsqueeze_2291: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2290, -1);  unsqueeze_2290 = None
        mul_859: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_2291);  sub_286 = unsqueeze_2291 = None
        unsqueeze_2292: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg584_1, -1);  arg584_1 = None
        unsqueeze_2293: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2292, -1);  unsqueeze_2292 = None
        mul_860: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_859, unsqueeze_2293);  mul_859 = unsqueeze_2293 = None
        unsqueeze_2294: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg585_1, -1);  arg585_1 = None
        unsqueeze_2295: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2294, -1);  unsqueeze_2294 = None
        add_725: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_860, unsqueeze_2295);  mul_860 = unsqueeze_2295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_279: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_725);  add_725 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_726: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_279, getitem_1118);  getitem_1118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_287: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_726, arg586_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_726 = arg586_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2296: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg587_1, -1);  arg587_1 = None
        unsqueeze_2297: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2296, -1);  unsqueeze_2296 = None
        sub_287: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_287, unsqueeze_2297);  convolution_287 = unsqueeze_2297 = None
        add_727: "f32[104]" = torch.ops.aten.add.Tensor(arg588_1, 1e-05);  arg588_1 = None
        sqrt_287: "f32[104]" = torch.ops.aten.sqrt.default(add_727);  add_727 = None
        reciprocal_287: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_287);  sqrt_287 = None
        mul_861: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_287, 1);  reciprocal_287 = None
        unsqueeze_2298: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_861, -1);  mul_861 = None
        unsqueeze_2299: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2298, -1);  unsqueeze_2298 = None
        mul_862: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_2299);  sub_287 = unsqueeze_2299 = None
        unsqueeze_2300: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg589_1, -1);  arg589_1 = None
        unsqueeze_2301: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2300, -1);  unsqueeze_2300 = None
        mul_863: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_862, unsqueeze_2301);  mul_862 = unsqueeze_2301 = None
        unsqueeze_2302: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg590_1, -1);  arg590_1 = None
        unsqueeze_2303: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2302, -1);  unsqueeze_2302 = None
        add_728: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_863, unsqueeze_2303);  mul_863 = unsqueeze_2303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_280: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_728);  add_728 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_55: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_278, relu_279, relu_280, getitem_1123], 1);  relu_278 = relu_279 = relu_280 = getitem_1123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_288: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_55, arg591_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_55 = arg591_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2304: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg592_1, -1);  arg592_1 = None
        unsqueeze_2305: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2304, -1);  unsqueeze_2304 = None
        sub_288: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_288, unsqueeze_2305);  convolution_288 = unsqueeze_2305 = None
        add_729: "f32[1024]" = torch.ops.aten.add.Tensor(arg593_1, 1e-05);  arg593_1 = None
        sqrt_288: "f32[1024]" = torch.ops.aten.sqrt.default(add_729);  add_729 = None
        reciprocal_288: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_288);  sqrt_288 = None
        mul_864: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_288, 1);  reciprocal_288 = None
        unsqueeze_2306: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_864, -1);  mul_864 = None
        unsqueeze_2307: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2306, -1);  unsqueeze_2306 = None
        mul_865: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_2307);  sub_288 = unsqueeze_2307 = None
        unsqueeze_2308: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg594_1, -1);  arg594_1 = None
        unsqueeze_2309: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2308, -1);  unsqueeze_2308 = None
        mul_866: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_865, unsqueeze_2309);  mul_865 = unsqueeze_2309 = None
        unsqueeze_2310: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg595_1, -1);  arg595_1 = None
        unsqueeze_2311: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2310, -1);  unsqueeze_2310 = None
        add_730: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_866, unsqueeze_2311);  mul_866 = unsqueeze_2311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_731: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_730, relu_276);  add_730 = relu_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_281: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_731);  add_731 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_289: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_281, arg596_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg596_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2312: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg597_1, -1);  arg597_1 = None
        unsqueeze_2313: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2312, -1);  unsqueeze_2312 = None
        sub_289: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_289, unsqueeze_2313);  convolution_289 = unsqueeze_2313 = None
        add_732: "f32[416]" = torch.ops.aten.add.Tensor(arg598_1, 1e-05);  arg598_1 = None
        sqrt_289: "f32[416]" = torch.ops.aten.sqrt.default(add_732);  add_732 = None
        reciprocal_289: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_289);  sqrt_289 = None
        mul_867: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_289, 1);  reciprocal_289 = None
        unsqueeze_2314: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_867, -1);  mul_867 = None
        unsqueeze_2315: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2314, -1);  unsqueeze_2314 = None
        mul_868: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_2315);  sub_289 = unsqueeze_2315 = None
        unsqueeze_2316: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg599_1, -1);  arg599_1 = None
        unsqueeze_2317: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2316, -1);  unsqueeze_2316 = None
        mul_869: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_868, unsqueeze_2317);  mul_868 = unsqueeze_2317 = None
        unsqueeze_2318: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg600_1, -1);  arg600_1 = None
        unsqueeze_2319: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2318, -1);  unsqueeze_2318 = None
        add_733: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_869, unsqueeze_2319);  mul_869 = unsqueeze_2319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_282: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_733);  add_733 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_281 = torch.ops.aten.split.Tensor(relu_282, 104, 1)
        getitem_1128: "f32[8, 104, 14, 14]" = split_281[0];  split_281 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_282 = torch.ops.aten.split.Tensor(relu_282, 104, 1)
        getitem_1133: "f32[8, 104, 14, 14]" = split_282[1];  split_282 = None
        split_283 = torch.ops.aten.split.Tensor(relu_282, 104, 1)
        getitem_1138: "f32[8, 104, 14, 14]" = split_283[2];  split_283 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_284 = torch.ops.aten.split.Tensor(relu_282, 104, 1);  relu_282 = None
        getitem_1143: "f32[8, 104, 14, 14]" = split_284[3];  split_284 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_290: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_1128, arg601_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1128 = arg601_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2320: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg602_1, -1);  arg602_1 = None
        unsqueeze_2321: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2320, -1);  unsqueeze_2320 = None
        sub_290: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_290, unsqueeze_2321);  convolution_290 = unsqueeze_2321 = None
        add_734: "f32[104]" = torch.ops.aten.add.Tensor(arg603_1, 1e-05);  arg603_1 = None
        sqrt_290: "f32[104]" = torch.ops.aten.sqrt.default(add_734);  add_734 = None
        reciprocal_290: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_290);  sqrt_290 = None
        mul_870: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_290, 1);  reciprocal_290 = None
        unsqueeze_2322: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_870, -1);  mul_870 = None
        unsqueeze_2323: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2322, -1);  unsqueeze_2322 = None
        mul_871: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_2323);  sub_290 = unsqueeze_2323 = None
        unsqueeze_2324: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg604_1, -1);  arg604_1 = None
        unsqueeze_2325: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2324, -1);  unsqueeze_2324 = None
        mul_872: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_871, unsqueeze_2325);  mul_871 = unsqueeze_2325 = None
        unsqueeze_2326: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg605_1, -1);  arg605_1 = None
        unsqueeze_2327: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2326, -1);  unsqueeze_2326 = None
        add_735: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_872, unsqueeze_2327);  mul_872 = unsqueeze_2327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_283: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_735);  add_735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_736: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_283, getitem_1133);  getitem_1133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_291: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_736, arg606_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_736 = arg606_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2328: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg607_1, -1);  arg607_1 = None
        unsqueeze_2329: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2328, -1);  unsqueeze_2328 = None
        sub_291: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_291, unsqueeze_2329);  convolution_291 = unsqueeze_2329 = None
        add_737: "f32[104]" = torch.ops.aten.add.Tensor(arg608_1, 1e-05);  arg608_1 = None
        sqrt_291: "f32[104]" = torch.ops.aten.sqrt.default(add_737);  add_737 = None
        reciprocal_291: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_291);  sqrt_291 = None
        mul_873: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_291, 1);  reciprocal_291 = None
        unsqueeze_2330: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_873, -1);  mul_873 = None
        unsqueeze_2331: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2330, -1);  unsqueeze_2330 = None
        mul_874: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_2331);  sub_291 = unsqueeze_2331 = None
        unsqueeze_2332: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg609_1, -1);  arg609_1 = None
        unsqueeze_2333: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2332, -1);  unsqueeze_2332 = None
        mul_875: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_874, unsqueeze_2333);  mul_874 = unsqueeze_2333 = None
        unsqueeze_2334: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg610_1, -1);  arg610_1 = None
        unsqueeze_2335: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2334, -1);  unsqueeze_2334 = None
        add_738: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_875, unsqueeze_2335);  mul_875 = unsqueeze_2335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_284: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_738);  add_738 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_739: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_284, getitem_1138);  getitem_1138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_292: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_739, arg611_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_739 = arg611_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2336: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg612_1, -1);  arg612_1 = None
        unsqueeze_2337: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2336, -1);  unsqueeze_2336 = None
        sub_292: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_292, unsqueeze_2337);  convolution_292 = unsqueeze_2337 = None
        add_740: "f32[104]" = torch.ops.aten.add.Tensor(arg613_1, 1e-05);  arg613_1 = None
        sqrt_292: "f32[104]" = torch.ops.aten.sqrt.default(add_740);  add_740 = None
        reciprocal_292: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_292);  sqrt_292 = None
        mul_876: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_292, 1);  reciprocal_292 = None
        unsqueeze_2338: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_876, -1);  mul_876 = None
        unsqueeze_2339: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2338, -1);  unsqueeze_2338 = None
        mul_877: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_2339);  sub_292 = unsqueeze_2339 = None
        unsqueeze_2340: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg614_1, -1);  arg614_1 = None
        unsqueeze_2341: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2340, -1);  unsqueeze_2340 = None
        mul_878: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_877, unsqueeze_2341);  mul_877 = unsqueeze_2341 = None
        unsqueeze_2342: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg615_1, -1);  arg615_1 = None
        unsqueeze_2343: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2342, -1);  unsqueeze_2342 = None
        add_741: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_878, unsqueeze_2343);  mul_878 = unsqueeze_2343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_285: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_741);  add_741 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_56: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_283, relu_284, relu_285, getitem_1143], 1);  relu_283 = relu_284 = relu_285 = getitem_1143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_293: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_56, arg616_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_56 = arg616_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2344: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg617_1, -1);  arg617_1 = None
        unsqueeze_2345: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2344, -1);  unsqueeze_2344 = None
        sub_293: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_293, unsqueeze_2345);  convolution_293 = unsqueeze_2345 = None
        add_742: "f32[1024]" = torch.ops.aten.add.Tensor(arg618_1, 1e-05);  arg618_1 = None
        sqrt_293: "f32[1024]" = torch.ops.aten.sqrt.default(add_742);  add_742 = None
        reciprocal_293: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_293);  sqrt_293 = None
        mul_879: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_293, 1);  reciprocal_293 = None
        unsqueeze_2346: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_879, -1);  mul_879 = None
        unsqueeze_2347: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2346, -1);  unsqueeze_2346 = None
        mul_880: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_2347);  sub_293 = unsqueeze_2347 = None
        unsqueeze_2348: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg619_1, -1);  arg619_1 = None
        unsqueeze_2349: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2348, -1);  unsqueeze_2348 = None
        mul_881: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_880, unsqueeze_2349);  mul_880 = unsqueeze_2349 = None
        unsqueeze_2350: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg620_1, -1);  arg620_1 = None
        unsqueeze_2351: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2350, -1);  unsqueeze_2350 = None
        add_743: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_881, unsqueeze_2351);  mul_881 = unsqueeze_2351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_744: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_743, relu_281);  add_743 = relu_281 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_286: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_744);  add_744 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_294: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_286, arg621_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg621_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2352: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg622_1, -1);  arg622_1 = None
        unsqueeze_2353: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2352, -1);  unsqueeze_2352 = None
        sub_294: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_294, unsqueeze_2353);  convolution_294 = unsqueeze_2353 = None
        add_745: "f32[416]" = torch.ops.aten.add.Tensor(arg623_1, 1e-05);  arg623_1 = None
        sqrt_294: "f32[416]" = torch.ops.aten.sqrt.default(add_745);  add_745 = None
        reciprocal_294: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_294);  sqrt_294 = None
        mul_882: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_294, 1);  reciprocal_294 = None
        unsqueeze_2354: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_882, -1);  mul_882 = None
        unsqueeze_2355: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2354, -1);  unsqueeze_2354 = None
        mul_883: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_2355);  sub_294 = unsqueeze_2355 = None
        unsqueeze_2356: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg624_1, -1);  arg624_1 = None
        unsqueeze_2357: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2356, -1);  unsqueeze_2356 = None
        mul_884: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_883, unsqueeze_2357);  mul_883 = unsqueeze_2357 = None
        unsqueeze_2358: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg625_1, -1);  arg625_1 = None
        unsqueeze_2359: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2358, -1);  unsqueeze_2358 = None
        add_746: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_884, unsqueeze_2359);  mul_884 = unsqueeze_2359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_287: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_746);  add_746 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_286 = torch.ops.aten.split.Tensor(relu_287, 104, 1)
        getitem_1148: "f32[8, 104, 14, 14]" = split_286[0];  split_286 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_287 = torch.ops.aten.split.Tensor(relu_287, 104, 1)
        getitem_1153: "f32[8, 104, 14, 14]" = split_287[1];  split_287 = None
        split_288 = torch.ops.aten.split.Tensor(relu_287, 104, 1)
        getitem_1158: "f32[8, 104, 14, 14]" = split_288[2];  split_288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_289 = torch.ops.aten.split.Tensor(relu_287, 104, 1);  relu_287 = None
        getitem_1163: "f32[8, 104, 14, 14]" = split_289[3];  split_289 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_295: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_1148, arg626_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1148 = arg626_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2360: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg627_1, -1);  arg627_1 = None
        unsqueeze_2361: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2360, -1);  unsqueeze_2360 = None
        sub_295: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_295, unsqueeze_2361);  convolution_295 = unsqueeze_2361 = None
        add_747: "f32[104]" = torch.ops.aten.add.Tensor(arg628_1, 1e-05);  arg628_1 = None
        sqrt_295: "f32[104]" = torch.ops.aten.sqrt.default(add_747);  add_747 = None
        reciprocal_295: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_295);  sqrt_295 = None
        mul_885: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_295, 1);  reciprocal_295 = None
        unsqueeze_2362: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_885, -1);  mul_885 = None
        unsqueeze_2363: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2362, -1);  unsqueeze_2362 = None
        mul_886: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_2363);  sub_295 = unsqueeze_2363 = None
        unsqueeze_2364: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg629_1, -1);  arg629_1 = None
        unsqueeze_2365: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2364, -1);  unsqueeze_2364 = None
        mul_887: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_886, unsqueeze_2365);  mul_886 = unsqueeze_2365 = None
        unsqueeze_2366: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg630_1, -1);  arg630_1 = None
        unsqueeze_2367: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2366, -1);  unsqueeze_2366 = None
        add_748: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_887, unsqueeze_2367);  mul_887 = unsqueeze_2367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_288: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_748);  add_748 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_749: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_288, getitem_1153);  getitem_1153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_296: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_749, arg631_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_749 = arg631_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2368: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg632_1, -1);  arg632_1 = None
        unsqueeze_2369: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2368, -1);  unsqueeze_2368 = None
        sub_296: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_296, unsqueeze_2369);  convolution_296 = unsqueeze_2369 = None
        add_750: "f32[104]" = torch.ops.aten.add.Tensor(arg633_1, 1e-05);  arg633_1 = None
        sqrt_296: "f32[104]" = torch.ops.aten.sqrt.default(add_750);  add_750 = None
        reciprocal_296: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_296);  sqrt_296 = None
        mul_888: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_296, 1);  reciprocal_296 = None
        unsqueeze_2370: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_888, -1);  mul_888 = None
        unsqueeze_2371: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2370, -1);  unsqueeze_2370 = None
        mul_889: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_2371);  sub_296 = unsqueeze_2371 = None
        unsqueeze_2372: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg634_1, -1);  arg634_1 = None
        unsqueeze_2373: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2372, -1);  unsqueeze_2372 = None
        mul_890: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_889, unsqueeze_2373);  mul_889 = unsqueeze_2373 = None
        unsqueeze_2374: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg635_1, -1);  arg635_1 = None
        unsqueeze_2375: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2374, -1);  unsqueeze_2374 = None
        add_751: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_890, unsqueeze_2375);  mul_890 = unsqueeze_2375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_289: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_751);  add_751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_752: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_289, getitem_1158);  getitem_1158 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_297: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_752, arg636_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_752 = arg636_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2376: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg637_1, -1);  arg637_1 = None
        unsqueeze_2377: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2376, -1);  unsqueeze_2376 = None
        sub_297: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_297, unsqueeze_2377);  convolution_297 = unsqueeze_2377 = None
        add_753: "f32[104]" = torch.ops.aten.add.Tensor(arg638_1, 1e-05);  arg638_1 = None
        sqrt_297: "f32[104]" = torch.ops.aten.sqrt.default(add_753);  add_753 = None
        reciprocal_297: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_297);  sqrt_297 = None
        mul_891: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_297, 1);  reciprocal_297 = None
        unsqueeze_2378: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_891, -1);  mul_891 = None
        unsqueeze_2379: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2378, -1);  unsqueeze_2378 = None
        mul_892: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_2379);  sub_297 = unsqueeze_2379 = None
        unsqueeze_2380: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg639_1, -1);  arg639_1 = None
        unsqueeze_2381: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2380, -1);  unsqueeze_2380 = None
        mul_893: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_892, unsqueeze_2381);  mul_892 = unsqueeze_2381 = None
        unsqueeze_2382: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg640_1, -1);  arg640_1 = None
        unsqueeze_2383: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2382, -1);  unsqueeze_2382 = None
        add_754: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_893, unsqueeze_2383);  mul_893 = unsqueeze_2383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_290: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_754);  add_754 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_57: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_288, relu_289, relu_290, getitem_1163], 1);  relu_288 = relu_289 = relu_290 = getitem_1163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_298: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_57, arg641_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_57 = arg641_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2384: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg642_1, -1);  arg642_1 = None
        unsqueeze_2385: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2384, -1);  unsqueeze_2384 = None
        sub_298: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_298, unsqueeze_2385);  convolution_298 = unsqueeze_2385 = None
        add_755: "f32[1024]" = torch.ops.aten.add.Tensor(arg643_1, 1e-05);  arg643_1 = None
        sqrt_298: "f32[1024]" = torch.ops.aten.sqrt.default(add_755);  add_755 = None
        reciprocal_298: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_298);  sqrt_298 = None
        mul_894: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_298, 1);  reciprocal_298 = None
        unsqueeze_2386: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_894, -1);  mul_894 = None
        unsqueeze_2387: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2386, -1);  unsqueeze_2386 = None
        mul_895: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_2387);  sub_298 = unsqueeze_2387 = None
        unsqueeze_2388: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg644_1, -1);  arg644_1 = None
        unsqueeze_2389: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2388, -1);  unsqueeze_2388 = None
        mul_896: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_895, unsqueeze_2389);  mul_895 = unsqueeze_2389 = None
        unsqueeze_2390: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg645_1, -1);  arg645_1 = None
        unsqueeze_2391: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2390, -1);  unsqueeze_2390 = None
        add_756: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_896, unsqueeze_2391);  mul_896 = unsqueeze_2391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_757: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_756, relu_286);  add_756 = relu_286 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_291: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_757);  add_757 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_299: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_291, arg646_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg646_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2392: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg647_1, -1);  arg647_1 = None
        unsqueeze_2393: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2392, -1);  unsqueeze_2392 = None
        sub_299: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_299, unsqueeze_2393);  convolution_299 = unsqueeze_2393 = None
        add_758: "f32[416]" = torch.ops.aten.add.Tensor(arg648_1, 1e-05);  arg648_1 = None
        sqrt_299: "f32[416]" = torch.ops.aten.sqrt.default(add_758);  add_758 = None
        reciprocal_299: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_299);  sqrt_299 = None
        mul_897: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_299, 1);  reciprocal_299 = None
        unsqueeze_2394: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_897, -1);  mul_897 = None
        unsqueeze_2395: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2394, -1);  unsqueeze_2394 = None
        mul_898: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_2395);  sub_299 = unsqueeze_2395 = None
        unsqueeze_2396: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg649_1, -1);  arg649_1 = None
        unsqueeze_2397: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2396, -1);  unsqueeze_2396 = None
        mul_899: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_898, unsqueeze_2397);  mul_898 = unsqueeze_2397 = None
        unsqueeze_2398: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg650_1, -1);  arg650_1 = None
        unsqueeze_2399: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2398, -1);  unsqueeze_2398 = None
        add_759: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_899, unsqueeze_2399);  mul_899 = unsqueeze_2399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_292: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_759);  add_759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_291 = torch.ops.aten.split.Tensor(relu_292, 104, 1)
        getitem_1168: "f32[8, 104, 14, 14]" = split_291[0];  split_291 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_292 = torch.ops.aten.split.Tensor(relu_292, 104, 1)
        getitem_1173: "f32[8, 104, 14, 14]" = split_292[1];  split_292 = None
        split_293 = torch.ops.aten.split.Tensor(relu_292, 104, 1)
        getitem_1178: "f32[8, 104, 14, 14]" = split_293[2];  split_293 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_294 = torch.ops.aten.split.Tensor(relu_292, 104, 1);  relu_292 = None
        getitem_1183: "f32[8, 104, 14, 14]" = split_294[3];  split_294 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_300: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_1168, arg651_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1168 = arg651_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2400: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg652_1, -1);  arg652_1 = None
        unsqueeze_2401: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2400, -1);  unsqueeze_2400 = None
        sub_300: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_300, unsqueeze_2401);  convolution_300 = unsqueeze_2401 = None
        add_760: "f32[104]" = torch.ops.aten.add.Tensor(arg653_1, 1e-05);  arg653_1 = None
        sqrt_300: "f32[104]" = torch.ops.aten.sqrt.default(add_760);  add_760 = None
        reciprocal_300: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_300);  sqrt_300 = None
        mul_900: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_300, 1);  reciprocal_300 = None
        unsqueeze_2402: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_900, -1);  mul_900 = None
        unsqueeze_2403: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2402, -1);  unsqueeze_2402 = None
        mul_901: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_2403);  sub_300 = unsqueeze_2403 = None
        unsqueeze_2404: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg654_1, -1);  arg654_1 = None
        unsqueeze_2405: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2404, -1);  unsqueeze_2404 = None
        mul_902: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_901, unsqueeze_2405);  mul_901 = unsqueeze_2405 = None
        unsqueeze_2406: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg655_1, -1);  arg655_1 = None
        unsqueeze_2407: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2406, -1);  unsqueeze_2406 = None
        add_761: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_902, unsqueeze_2407);  mul_902 = unsqueeze_2407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_293: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_761);  add_761 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_762: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_293, getitem_1173);  getitem_1173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_301: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_762, arg656_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_762 = arg656_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2408: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg657_1, -1);  arg657_1 = None
        unsqueeze_2409: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2408, -1);  unsqueeze_2408 = None
        sub_301: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_301, unsqueeze_2409);  convolution_301 = unsqueeze_2409 = None
        add_763: "f32[104]" = torch.ops.aten.add.Tensor(arg658_1, 1e-05);  arg658_1 = None
        sqrt_301: "f32[104]" = torch.ops.aten.sqrt.default(add_763);  add_763 = None
        reciprocal_301: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_301);  sqrt_301 = None
        mul_903: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_301, 1);  reciprocal_301 = None
        unsqueeze_2410: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_903, -1);  mul_903 = None
        unsqueeze_2411: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2410, -1);  unsqueeze_2410 = None
        mul_904: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_2411);  sub_301 = unsqueeze_2411 = None
        unsqueeze_2412: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg659_1, -1);  arg659_1 = None
        unsqueeze_2413: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2412, -1);  unsqueeze_2412 = None
        mul_905: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_904, unsqueeze_2413);  mul_904 = unsqueeze_2413 = None
        unsqueeze_2414: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg660_1, -1);  arg660_1 = None
        unsqueeze_2415: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2414, -1);  unsqueeze_2414 = None
        add_764: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_905, unsqueeze_2415);  mul_905 = unsqueeze_2415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_294: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_764);  add_764 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_765: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_294, getitem_1178);  getitem_1178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_302: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_765, arg661_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_765 = arg661_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2416: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg662_1, -1);  arg662_1 = None
        unsqueeze_2417: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2416, -1);  unsqueeze_2416 = None
        sub_302: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_302, unsqueeze_2417);  convolution_302 = unsqueeze_2417 = None
        add_766: "f32[104]" = torch.ops.aten.add.Tensor(arg663_1, 1e-05);  arg663_1 = None
        sqrt_302: "f32[104]" = torch.ops.aten.sqrt.default(add_766);  add_766 = None
        reciprocal_302: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_302);  sqrt_302 = None
        mul_906: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_302, 1);  reciprocal_302 = None
        unsqueeze_2418: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_906, -1);  mul_906 = None
        unsqueeze_2419: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2418, -1);  unsqueeze_2418 = None
        mul_907: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_2419);  sub_302 = unsqueeze_2419 = None
        unsqueeze_2420: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg664_1, -1);  arg664_1 = None
        unsqueeze_2421: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2420, -1);  unsqueeze_2420 = None
        mul_908: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_907, unsqueeze_2421);  mul_907 = unsqueeze_2421 = None
        unsqueeze_2422: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg665_1, -1);  arg665_1 = None
        unsqueeze_2423: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2422, -1);  unsqueeze_2422 = None
        add_767: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_908, unsqueeze_2423);  mul_908 = unsqueeze_2423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_295: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_767);  add_767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_58: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_293, relu_294, relu_295, getitem_1183], 1);  relu_293 = relu_294 = relu_295 = getitem_1183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_303: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_58, arg666_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_58 = arg666_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2424: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg667_1, -1);  arg667_1 = None
        unsqueeze_2425: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2424, -1);  unsqueeze_2424 = None
        sub_303: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_303, unsqueeze_2425);  convolution_303 = unsqueeze_2425 = None
        add_768: "f32[1024]" = torch.ops.aten.add.Tensor(arg668_1, 1e-05);  arg668_1 = None
        sqrt_303: "f32[1024]" = torch.ops.aten.sqrt.default(add_768);  add_768 = None
        reciprocal_303: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_303);  sqrt_303 = None
        mul_909: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_303, 1);  reciprocal_303 = None
        unsqueeze_2426: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_909, -1);  mul_909 = None
        unsqueeze_2427: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2426, -1);  unsqueeze_2426 = None
        mul_910: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_2427);  sub_303 = unsqueeze_2427 = None
        unsqueeze_2428: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg669_1, -1);  arg669_1 = None
        unsqueeze_2429: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2428, -1);  unsqueeze_2428 = None
        mul_911: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_910, unsqueeze_2429);  mul_910 = unsqueeze_2429 = None
        unsqueeze_2430: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg670_1, -1);  arg670_1 = None
        unsqueeze_2431: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2430, -1);  unsqueeze_2430 = None
        add_769: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_911, unsqueeze_2431);  mul_911 = unsqueeze_2431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_770: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_769, relu_291);  add_769 = relu_291 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_296: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_770);  add_770 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_304: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_296, arg671_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg671_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2432: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg672_1, -1);  arg672_1 = None
        unsqueeze_2433: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2432, -1);  unsqueeze_2432 = None
        sub_304: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_304, unsqueeze_2433);  convolution_304 = unsqueeze_2433 = None
        add_771: "f32[416]" = torch.ops.aten.add.Tensor(arg673_1, 1e-05);  arg673_1 = None
        sqrt_304: "f32[416]" = torch.ops.aten.sqrt.default(add_771);  add_771 = None
        reciprocal_304: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_304);  sqrt_304 = None
        mul_912: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_304, 1);  reciprocal_304 = None
        unsqueeze_2434: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_912, -1);  mul_912 = None
        unsqueeze_2435: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2434, -1);  unsqueeze_2434 = None
        mul_913: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_2435);  sub_304 = unsqueeze_2435 = None
        unsqueeze_2436: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg674_1, -1);  arg674_1 = None
        unsqueeze_2437: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2436, -1);  unsqueeze_2436 = None
        mul_914: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_913, unsqueeze_2437);  mul_913 = unsqueeze_2437 = None
        unsqueeze_2438: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg675_1, -1);  arg675_1 = None
        unsqueeze_2439: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2438, -1);  unsqueeze_2438 = None
        add_772: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_914, unsqueeze_2439);  mul_914 = unsqueeze_2439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_297: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_772);  add_772 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_296 = torch.ops.aten.split.Tensor(relu_297, 104, 1)
        getitem_1188: "f32[8, 104, 14, 14]" = split_296[0];  split_296 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_297 = torch.ops.aten.split.Tensor(relu_297, 104, 1)
        getitem_1193: "f32[8, 104, 14, 14]" = split_297[1];  split_297 = None
        split_298 = torch.ops.aten.split.Tensor(relu_297, 104, 1)
        getitem_1198: "f32[8, 104, 14, 14]" = split_298[2];  split_298 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_299 = torch.ops.aten.split.Tensor(relu_297, 104, 1);  relu_297 = None
        getitem_1203: "f32[8, 104, 14, 14]" = split_299[3];  split_299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_305: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_1188, arg676_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1188 = arg676_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2440: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg677_1, -1);  arg677_1 = None
        unsqueeze_2441: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2440, -1);  unsqueeze_2440 = None
        sub_305: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_305, unsqueeze_2441);  convolution_305 = unsqueeze_2441 = None
        add_773: "f32[104]" = torch.ops.aten.add.Tensor(arg678_1, 1e-05);  arg678_1 = None
        sqrt_305: "f32[104]" = torch.ops.aten.sqrt.default(add_773);  add_773 = None
        reciprocal_305: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_305);  sqrt_305 = None
        mul_915: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_305, 1);  reciprocal_305 = None
        unsqueeze_2442: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_915, -1);  mul_915 = None
        unsqueeze_2443: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2442, -1);  unsqueeze_2442 = None
        mul_916: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_2443);  sub_305 = unsqueeze_2443 = None
        unsqueeze_2444: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg679_1, -1);  arg679_1 = None
        unsqueeze_2445: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2444, -1);  unsqueeze_2444 = None
        mul_917: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_916, unsqueeze_2445);  mul_916 = unsqueeze_2445 = None
        unsqueeze_2446: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg680_1, -1);  arg680_1 = None
        unsqueeze_2447: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2446, -1);  unsqueeze_2446 = None
        add_774: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_917, unsqueeze_2447);  mul_917 = unsqueeze_2447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_298: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_774);  add_774 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_775: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_298, getitem_1193);  getitem_1193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_306: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_775, arg681_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_775 = arg681_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2448: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg682_1, -1);  arg682_1 = None
        unsqueeze_2449: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2448, -1);  unsqueeze_2448 = None
        sub_306: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_306, unsqueeze_2449);  convolution_306 = unsqueeze_2449 = None
        add_776: "f32[104]" = torch.ops.aten.add.Tensor(arg683_1, 1e-05);  arg683_1 = None
        sqrt_306: "f32[104]" = torch.ops.aten.sqrt.default(add_776);  add_776 = None
        reciprocal_306: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_306);  sqrt_306 = None
        mul_918: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_306, 1);  reciprocal_306 = None
        unsqueeze_2450: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_918, -1);  mul_918 = None
        unsqueeze_2451: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2450, -1);  unsqueeze_2450 = None
        mul_919: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_2451);  sub_306 = unsqueeze_2451 = None
        unsqueeze_2452: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg684_1, -1);  arg684_1 = None
        unsqueeze_2453: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2452, -1);  unsqueeze_2452 = None
        mul_920: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_919, unsqueeze_2453);  mul_919 = unsqueeze_2453 = None
        unsqueeze_2454: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg685_1, -1);  arg685_1 = None
        unsqueeze_2455: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2454, -1);  unsqueeze_2454 = None
        add_777: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_920, unsqueeze_2455);  mul_920 = unsqueeze_2455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_299: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_777);  add_777 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_778: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_299, getitem_1198);  getitem_1198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_307: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_778, arg686_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_778 = arg686_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2456: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg687_1, -1);  arg687_1 = None
        unsqueeze_2457: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2456, -1);  unsqueeze_2456 = None
        sub_307: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_307, unsqueeze_2457);  convolution_307 = unsqueeze_2457 = None
        add_779: "f32[104]" = torch.ops.aten.add.Tensor(arg688_1, 1e-05);  arg688_1 = None
        sqrt_307: "f32[104]" = torch.ops.aten.sqrt.default(add_779);  add_779 = None
        reciprocal_307: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_307);  sqrt_307 = None
        mul_921: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_307, 1);  reciprocal_307 = None
        unsqueeze_2458: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_921, -1);  mul_921 = None
        unsqueeze_2459: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2458, -1);  unsqueeze_2458 = None
        mul_922: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_2459);  sub_307 = unsqueeze_2459 = None
        unsqueeze_2460: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg689_1, -1);  arg689_1 = None
        unsqueeze_2461: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2460, -1);  unsqueeze_2460 = None
        mul_923: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_922, unsqueeze_2461);  mul_922 = unsqueeze_2461 = None
        unsqueeze_2462: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg690_1, -1);  arg690_1 = None
        unsqueeze_2463: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2462, -1);  unsqueeze_2462 = None
        add_780: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_923, unsqueeze_2463);  mul_923 = unsqueeze_2463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_300: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_780);  add_780 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_59: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_298, relu_299, relu_300, getitem_1203], 1);  relu_298 = relu_299 = relu_300 = getitem_1203 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_308: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_59, arg691_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_59 = arg691_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2464: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg692_1, -1);  arg692_1 = None
        unsqueeze_2465: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2464, -1);  unsqueeze_2464 = None
        sub_308: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_308, unsqueeze_2465);  convolution_308 = unsqueeze_2465 = None
        add_781: "f32[1024]" = torch.ops.aten.add.Tensor(arg693_1, 1e-05);  arg693_1 = None
        sqrt_308: "f32[1024]" = torch.ops.aten.sqrt.default(add_781);  add_781 = None
        reciprocal_308: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_308);  sqrt_308 = None
        mul_924: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_308, 1);  reciprocal_308 = None
        unsqueeze_2466: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_924, -1);  mul_924 = None
        unsqueeze_2467: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2466, -1);  unsqueeze_2466 = None
        mul_925: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_2467);  sub_308 = unsqueeze_2467 = None
        unsqueeze_2468: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg694_1, -1);  arg694_1 = None
        unsqueeze_2469: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2468, -1);  unsqueeze_2468 = None
        mul_926: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_925, unsqueeze_2469);  mul_925 = unsqueeze_2469 = None
        unsqueeze_2470: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg695_1, -1);  arg695_1 = None
        unsqueeze_2471: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2470, -1);  unsqueeze_2470 = None
        add_782: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_926, unsqueeze_2471);  mul_926 = unsqueeze_2471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_783: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_782, relu_296);  add_782 = relu_296 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_301: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_783);  add_783 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_309: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_301, arg696_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg696_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2472: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg697_1, -1);  arg697_1 = None
        unsqueeze_2473: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2472, -1);  unsqueeze_2472 = None
        sub_309: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_309, unsqueeze_2473);  convolution_309 = unsqueeze_2473 = None
        add_784: "f32[416]" = torch.ops.aten.add.Tensor(arg698_1, 1e-05);  arg698_1 = None
        sqrt_309: "f32[416]" = torch.ops.aten.sqrt.default(add_784);  add_784 = None
        reciprocal_309: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_309);  sqrt_309 = None
        mul_927: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_309, 1);  reciprocal_309 = None
        unsqueeze_2474: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_927, -1);  mul_927 = None
        unsqueeze_2475: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2474, -1);  unsqueeze_2474 = None
        mul_928: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_2475);  sub_309 = unsqueeze_2475 = None
        unsqueeze_2476: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg699_1, -1);  arg699_1 = None
        unsqueeze_2477: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2476, -1);  unsqueeze_2476 = None
        mul_929: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_928, unsqueeze_2477);  mul_928 = unsqueeze_2477 = None
        unsqueeze_2478: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg700_1, -1);  arg700_1 = None
        unsqueeze_2479: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2478, -1);  unsqueeze_2478 = None
        add_785: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_929, unsqueeze_2479);  mul_929 = unsqueeze_2479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_302: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_785);  add_785 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_301 = torch.ops.aten.split.Tensor(relu_302, 104, 1)
        getitem_1208: "f32[8, 104, 14, 14]" = split_301[0];  split_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_302 = torch.ops.aten.split.Tensor(relu_302, 104, 1)
        getitem_1213: "f32[8, 104, 14, 14]" = split_302[1];  split_302 = None
        split_303 = torch.ops.aten.split.Tensor(relu_302, 104, 1)
        getitem_1218: "f32[8, 104, 14, 14]" = split_303[2];  split_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_304 = torch.ops.aten.split.Tensor(relu_302, 104, 1);  relu_302 = None
        getitem_1223: "f32[8, 104, 14, 14]" = split_304[3];  split_304 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_310: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_1208, arg701_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1208 = arg701_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2480: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg702_1, -1);  arg702_1 = None
        unsqueeze_2481: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2480, -1);  unsqueeze_2480 = None
        sub_310: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_310, unsqueeze_2481);  convolution_310 = unsqueeze_2481 = None
        add_786: "f32[104]" = torch.ops.aten.add.Tensor(arg703_1, 1e-05);  arg703_1 = None
        sqrt_310: "f32[104]" = torch.ops.aten.sqrt.default(add_786);  add_786 = None
        reciprocal_310: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_310);  sqrt_310 = None
        mul_930: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_310, 1);  reciprocal_310 = None
        unsqueeze_2482: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_930, -1);  mul_930 = None
        unsqueeze_2483: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2482, -1);  unsqueeze_2482 = None
        mul_931: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_2483);  sub_310 = unsqueeze_2483 = None
        unsqueeze_2484: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg704_1, -1);  arg704_1 = None
        unsqueeze_2485: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2484, -1);  unsqueeze_2484 = None
        mul_932: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_931, unsqueeze_2485);  mul_931 = unsqueeze_2485 = None
        unsqueeze_2486: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg705_1, -1);  arg705_1 = None
        unsqueeze_2487: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2486, -1);  unsqueeze_2486 = None
        add_787: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_932, unsqueeze_2487);  mul_932 = unsqueeze_2487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_303: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_787);  add_787 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_788: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_303, getitem_1213);  getitem_1213 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_311: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_788, arg706_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_788 = arg706_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2488: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg707_1, -1);  arg707_1 = None
        unsqueeze_2489: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2488, -1);  unsqueeze_2488 = None
        sub_311: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_311, unsqueeze_2489);  convolution_311 = unsqueeze_2489 = None
        add_789: "f32[104]" = torch.ops.aten.add.Tensor(arg708_1, 1e-05);  arg708_1 = None
        sqrt_311: "f32[104]" = torch.ops.aten.sqrt.default(add_789);  add_789 = None
        reciprocal_311: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_311);  sqrt_311 = None
        mul_933: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_311, 1);  reciprocal_311 = None
        unsqueeze_2490: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_933, -1);  mul_933 = None
        unsqueeze_2491: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2490, -1);  unsqueeze_2490 = None
        mul_934: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_2491);  sub_311 = unsqueeze_2491 = None
        unsqueeze_2492: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg709_1, -1);  arg709_1 = None
        unsqueeze_2493: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2492, -1);  unsqueeze_2492 = None
        mul_935: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_934, unsqueeze_2493);  mul_934 = unsqueeze_2493 = None
        unsqueeze_2494: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg710_1, -1);  arg710_1 = None
        unsqueeze_2495: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2494, -1);  unsqueeze_2494 = None
        add_790: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_935, unsqueeze_2495);  mul_935 = unsqueeze_2495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_304: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_790);  add_790 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_791: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_304, getitem_1218);  getitem_1218 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_312: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_791, arg711_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_791 = arg711_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2496: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg712_1, -1);  arg712_1 = None
        unsqueeze_2497: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2496, -1);  unsqueeze_2496 = None
        sub_312: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_312, unsqueeze_2497);  convolution_312 = unsqueeze_2497 = None
        add_792: "f32[104]" = torch.ops.aten.add.Tensor(arg713_1, 1e-05);  arg713_1 = None
        sqrt_312: "f32[104]" = torch.ops.aten.sqrt.default(add_792);  add_792 = None
        reciprocal_312: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_312);  sqrt_312 = None
        mul_936: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_312, 1);  reciprocal_312 = None
        unsqueeze_2498: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_936, -1);  mul_936 = None
        unsqueeze_2499: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2498, -1);  unsqueeze_2498 = None
        mul_937: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_2499);  sub_312 = unsqueeze_2499 = None
        unsqueeze_2500: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg714_1, -1);  arg714_1 = None
        unsqueeze_2501: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2500, -1);  unsqueeze_2500 = None
        mul_938: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_937, unsqueeze_2501);  mul_937 = unsqueeze_2501 = None
        unsqueeze_2502: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg715_1, -1);  arg715_1 = None
        unsqueeze_2503: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2502, -1);  unsqueeze_2502 = None
        add_793: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_938, unsqueeze_2503);  mul_938 = unsqueeze_2503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_305: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_793);  add_793 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_60: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_303, relu_304, relu_305, getitem_1223], 1);  relu_303 = relu_304 = relu_305 = getitem_1223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_313: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_60, arg716_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_60 = arg716_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2504: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg717_1, -1);  arg717_1 = None
        unsqueeze_2505: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2504, -1);  unsqueeze_2504 = None
        sub_313: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_313, unsqueeze_2505);  convolution_313 = unsqueeze_2505 = None
        add_794: "f32[1024]" = torch.ops.aten.add.Tensor(arg718_1, 1e-05);  arg718_1 = None
        sqrt_313: "f32[1024]" = torch.ops.aten.sqrt.default(add_794);  add_794 = None
        reciprocal_313: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_313);  sqrt_313 = None
        mul_939: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_313, 1);  reciprocal_313 = None
        unsqueeze_2506: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_939, -1);  mul_939 = None
        unsqueeze_2507: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2506, -1);  unsqueeze_2506 = None
        mul_940: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_2507);  sub_313 = unsqueeze_2507 = None
        unsqueeze_2508: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg719_1, -1);  arg719_1 = None
        unsqueeze_2509: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2508, -1);  unsqueeze_2508 = None
        mul_941: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_940, unsqueeze_2509);  mul_940 = unsqueeze_2509 = None
        unsqueeze_2510: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg720_1, -1);  arg720_1 = None
        unsqueeze_2511: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2510, -1);  unsqueeze_2510 = None
        add_795: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_941, unsqueeze_2511);  mul_941 = unsqueeze_2511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_796: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_795, relu_301);  add_795 = relu_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_306: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_796);  add_796 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_314: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_306, arg721_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg721_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2512: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg722_1, -1);  arg722_1 = None
        unsqueeze_2513: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2512, -1);  unsqueeze_2512 = None
        sub_314: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_314, unsqueeze_2513);  convolution_314 = unsqueeze_2513 = None
        add_797: "f32[416]" = torch.ops.aten.add.Tensor(arg723_1, 1e-05);  arg723_1 = None
        sqrt_314: "f32[416]" = torch.ops.aten.sqrt.default(add_797);  add_797 = None
        reciprocal_314: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_314);  sqrt_314 = None
        mul_942: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_314, 1);  reciprocal_314 = None
        unsqueeze_2514: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_942, -1);  mul_942 = None
        unsqueeze_2515: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2514, -1);  unsqueeze_2514 = None
        mul_943: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_2515);  sub_314 = unsqueeze_2515 = None
        unsqueeze_2516: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg724_1, -1);  arg724_1 = None
        unsqueeze_2517: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2516, -1);  unsqueeze_2516 = None
        mul_944: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_943, unsqueeze_2517);  mul_943 = unsqueeze_2517 = None
        unsqueeze_2518: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg725_1, -1);  arg725_1 = None
        unsqueeze_2519: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2518, -1);  unsqueeze_2518 = None
        add_798: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_944, unsqueeze_2519);  mul_944 = unsqueeze_2519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_307: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_798);  add_798 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_306 = torch.ops.aten.split.Tensor(relu_307, 104, 1)
        getitem_1228: "f32[8, 104, 14, 14]" = split_306[0];  split_306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_307 = torch.ops.aten.split.Tensor(relu_307, 104, 1)
        getitem_1233: "f32[8, 104, 14, 14]" = split_307[1];  split_307 = None
        split_308 = torch.ops.aten.split.Tensor(relu_307, 104, 1)
        getitem_1238: "f32[8, 104, 14, 14]" = split_308[2];  split_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_309 = torch.ops.aten.split.Tensor(relu_307, 104, 1);  relu_307 = None
        getitem_1243: "f32[8, 104, 14, 14]" = split_309[3];  split_309 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_315: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_1228, arg726_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1228 = arg726_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2520: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg727_1, -1);  arg727_1 = None
        unsqueeze_2521: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2520, -1);  unsqueeze_2520 = None
        sub_315: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_315, unsqueeze_2521);  convolution_315 = unsqueeze_2521 = None
        add_799: "f32[104]" = torch.ops.aten.add.Tensor(arg728_1, 1e-05);  arg728_1 = None
        sqrt_315: "f32[104]" = torch.ops.aten.sqrt.default(add_799);  add_799 = None
        reciprocal_315: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_315);  sqrt_315 = None
        mul_945: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_315, 1);  reciprocal_315 = None
        unsqueeze_2522: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_945, -1);  mul_945 = None
        unsqueeze_2523: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2522, -1);  unsqueeze_2522 = None
        mul_946: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_2523);  sub_315 = unsqueeze_2523 = None
        unsqueeze_2524: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg729_1, -1);  arg729_1 = None
        unsqueeze_2525: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2524, -1);  unsqueeze_2524 = None
        mul_947: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_946, unsqueeze_2525);  mul_946 = unsqueeze_2525 = None
        unsqueeze_2526: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg730_1, -1);  arg730_1 = None
        unsqueeze_2527: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2526, -1);  unsqueeze_2526 = None
        add_800: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_947, unsqueeze_2527);  mul_947 = unsqueeze_2527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_308: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_800);  add_800 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_801: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_308, getitem_1233);  getitem_1233 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_316: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_801, arg731_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_801 = arg731_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2528: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg732_1, -1);  arg732_1 = None
        unsqueeze_2529: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2528, -1);  unsqueeze_2528 = None
        sub_316: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_316, unsqueeze_2529);  convolution_316 = unsqueeze_2529 = None
        add_802: "f32[104]" = torch.ops.aten.add.Tensor(arg733_1, 1e-05);  arg733_1 = None
        sqrt_316: "f32[104]" = torch.ops.aten.sqrt.default(add_802);  add_802 = None
        reciprocal_316: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_316);  sqrt_316 = None
        mul_948: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_316, 1);  reciprocal_316 = None
        unsqueeze_2530: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_948, -1);  mul_948 = None
        unsqueeze_2531: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2530, -1);  unsqueeze_2530 = None
        mul_949: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_2531);  sub_316 = unsqueeze_2531 = None
        unsqueeze_2532: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg734_1, -1);  arg734_1 = None
        unsqueeze_2533: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2532, -1);  unsqueeze_2532 = None
        mul_950: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_949, unsqueeze_2533);  mul_949 = unsqueeze_2533 = None
        unsqueeze_2534: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg735_1, -1);  arg735_1 = None
        unsqueeze_2535: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2534, -1);  unsqueeze_2534 = None
        add_803: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_950, unsqueeze_2535);  mul_950 = unsqueeze_2535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_309: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_803);  add_803 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_804: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_309, getitem_1238);  getitem_1238 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_317: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_804, arg736_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_804 = arg736_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2536: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg737_1, -1);  arg737_1 = None
        unsqueeze_2537: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2536, -1);  unsqueeze_2536 = None
        sub_317: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_317, unsqueeze_2537);  convolution_317 = unsqueeze_2537 = None
        add_805: "f32[104]" = torch.ops.aten.add.Tensor(arg738_1, 1e-05);  arg738_1 = None
        sqrt_317: "f32[104]" = torch.ops.aten.sqrt.default(add_805);  add_805 = None
        reciprocal_317: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_317);  sqrt_317 = None
        mul_951: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_317, 1);  reciprocal_317 = None
        unsqueeze_2538: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_951, -1);  mul_951 = None
        unsqueeze_2539: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2538, -1);  unsqueeze_2538 = None
        mul_952: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_2539);  sub_317 = unsqueeze_2539 = None
        unsqueeze_2540: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg739_1, -1);  arg739_1 = None
        unsqueeze_2541: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2540, -1);  unsqueeze_2540 = None
        mul_953: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_952, unsqueeze_2541);  mul_952 = unsqueeze_2541 = None
        unsqueeze_2542: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg740_1, -1);  arg740_1 = None
        unsqueeze_2543: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2542, -1);  unsqueeze_2542 = None
        add_806: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_953, unsqueeze_2543);  mul_953 = unsqueeze_2543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_310: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_806);  add_806 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_61: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_308, relu_309, relu_310, getitem_1243], 1);  relu_308 = relu_309 = relu_310 = getitem_1243 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_318: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_61, arg741_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_61 = arg741_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2544: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg742_1, -1);  arg742_1 = None
        unsqueeze_2545: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2544, -1);  unsqueeze_2544 = None
        sub_318: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_318, unsqueeze_2545);  convolution_318 = unsqueeze_2545 = None
        add_807: "f32[1024]" = torch.ops.aten.add.Tensor(arg743_1, 1e-05);  arg743_1 = None
        sqrt_318: "f32[1024]" = torch.ops.aten.sqrt.default(add_807);  add_807 = None
        reciprocal_318: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_318);  sqrt_318 = None
        mul_954: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_318, 1);  reciprocal_318 = None
        unsqueeze_2546: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_954, -1);  mul_954 = None
        unsqueeze_2547: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2546, -1);  unsqueeze_2546 = None
        mul_955: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_2547);  sub_318 = unsqueeze_2547 = None
        unsqueeze_2548: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg744_1, -1);  arg744_1 = None
        unsqueeze_2549: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2548, -1);  unsqueeze_2548 = None
        mul_956: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_955, unsqueeze_2549);  mul_955 = unsqueeze_2549 = None
        unsqueeze_2550: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg745_1, -1);  arg745_1 = None
        unsqueeze_2551: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2550, -1);  unsqueeze_2550 = None
        add_808: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_956, unsqueeze_2551);  mul_956 = unsqueeze_2551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_809: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_808, relu_306);  add_808 = relu_306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_311: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_809);  add_809 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_319: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_311, arg746_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg746_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2552: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg747_1, -1);  arg747_1 = None
        unsqueeze_2553: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2552, -1);  unsqueeze_2552 = None
        sub_319: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_319, unsqueeze_2553);  convolution_319 = unsqueeze_2553 = None
        add_810: "f32[416]" = torch.ops.aten.add.Tensor(arg748_1, 1e-05);  arg748_1 = None
        sqrt_319: "f32[416]" = torch.ops.aten.sqrt.default(add_810);  add_810 = None
        reciprocal_319: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_319);  sqrt_319 = None
        mul_957: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_319, 1);  reciprocal_319 = None
        unsqueeze_2554: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_957, -1);  mul_957 = None
        unsqueeze_2555: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2554, -1);  unsqueeze_2554 = None
        mul_958: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_2555);  sub_319 = unsqueeze_2555 = None
        unsqueeze_2556: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg749_1, -1);  arg749_1 = None
        unsqueeze_2557: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2556, -1);  unsqueeze_2556 = None
        mul_959: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_958, unsqueeze_2557);  mul_958 = unsqueeze_2557 = None
        unsqueeze_2558: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(arg750_1, -1);  arg750_1 = None
        unsqueeze_2559: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2558, -1);  unsqueeze_2558 = None
        add_811: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_959, unsqueeze_2559);  mul_959 = unsqueeze_2559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_312: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_811);  add_811 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_311 = torch.ops.aten.split.Tensor(relu_312, 104, 1)
        getitem_1248: "f32[8, 104, 14, 14]" = split_311[0];  split_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_312 = torch.ops.aten.split.Tensor(relu_312, 104, 1)
        getitem_1253: "f32[8, 104, 14, 14]" = split_312[1];  split_312 = None
        split_313 = torch.ops.aten.split.Tensor(relu_312, 104, 1)
        getitem_1258: "f32[8, 104, 14, 14]" = split_313[2];  split_313 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_314 = torch.ops.aten.split.Tensor(relu_312, 104, 1);  relu_312 = None
        getitem_1263: "f32[8, 104, 14, 14]" = split_314[3];  split_314 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_320: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_1248, arg751_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1248 = arg751_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2560: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg752_1, -1);  arg752_1 = None
        unsqueeze_2561: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2560, -1);  unsqueeze_2560 = None
        sub_320: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_320, unsqueeze_2561);  convolution_320 = unsqueeze_2561 = None
        add_812: "f32[104]" = torch.ops.aten.add.Tensor(arg753_1, 1e-05);  arg753_1 = None
        sqrt_320: "f32[104]" = torch.ops.aten.sqrt.default(add_812);  add_812 = None
        reciprocal_320: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_320);  sqrt_320 = None
        mul_960: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_320, 1);  reciprocal_320 = None
        unsqueeze_2562: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_960, -1);  mul_960 = None
        unsqueeze_2563: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2562, -1);  unsqueeze_2562 = None
        mul_961: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_2563);  sub_320 = unsqueeze_2563 = None
        unsqueeze_2564: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg754_1, -1);  arg754_1 = None
        unsqueeze_2565: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2564, -1);  unsqueeze_2564 = None
        mul_962: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_961, unsqueeze_2565);  mul_961 = unsqueeze_2565 = None
        unsqueeze_2566: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg755_1, -1);  arg755_1 = None
        unsqueeze_2567: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2566, -1);  unsqueeze_2566 = None
        add_813: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_962, unsqueeze_2567);  mul_962 = unsqueeze_2567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_313: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_813);  add_813 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_814: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_313, getitem_1253);  getitem_1253 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_321: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_814, arg756_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_814 = arg756_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2568: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg757_1, -1);  arg757_1 = None
        unsqueeze_2569: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2568, -1);  unsqueeze_2568 = None
        sub_321: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_321, unsqueeze_2569);  convolution_321 = unsqueeze_2569 = None
        add_815: "f32[104]" = torch.ops.aten.add.Tensor(arg758_1, 1e-05);  arg758_1 = None
        sqrt_321: "f32[104]" = torch.ops.aten.sqrt.default(add_815);  add_815 = None
        reciprocal_321: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_321);  sqrt_321 = None
        mul_963: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_321, 1);  reciprocal_321 = None
        unsqueeze_2570: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_963, -1);  mul_963 = None
        unsqueeze_2571: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2570, -1);  unsqueeze_2570 = None
        mul_964: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_2571);  sub_321 = unsqueeze_2571 = None
        unsqueeze_2572: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg759_1, -1);  arg759_1 = None
        unsqueeze_2573: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2572, -1);  unsqueeze_2572 = None
        mul_965: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_964, unsqueeze_2573);  mul_964 = unsqueeze_2573 = None
        unsqueeze_2574: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg760_1, -1);  arg760_1 = None
        unsqueeze_2575: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2574, -1);  unsqueeze_2574 = None
        add_816: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_965, unsqueeze_2575);  mul_965 = unsqueeze_2575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_314: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_816);  add_816 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_817: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_314, getitem_1258);  getitem_1258 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_322: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_817, arg761_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_817 = arg761_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2576: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg762_1, -1);  arg762_1 = None
        unsqueeze_2577: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2576, -1);  unsqueeze_2576 = None
        sub_322: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_322, unsqueeze_2577);  convolution_322 = unsqueeze_2577 = None
        add_818: "f32[104]" = torch.ops.aten.add.Tensor(arg763_1, 1e-05);  arg763_1 = None
        sqrt_322: "f32[104]" = torch.ops.aten.sqrt.default(add_818);  add_818 = None
        reciprocal_322: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_322);  sqrt_322 = None
        mul_966: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_322, 1);  reciprocal_322 = None
        unsqueeze_2578: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_966, -1);  mul_966 = None
        unsqueeze_2579: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2578, -1);  unsqueeze_2578 = None
        mul_967: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_2579);  sub_322 = unsqueeze_2579 = None
        unsqueeze_2580: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg764_1, -1);  arg764_1 = None
        unsqueeze_2581: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2580, -1);  unsqueeze_2580 = None
        mul_968: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_967, unsqueeze_2581);  mul_967 = unsqueeze_2581 = None
        unsqueeze_2582: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg765_1, -1);  arg765_1 = None
        unsqueeze_2583: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2582, -1);  unsqueeze_2582 = None
        add_819: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_968, unsqueeze_2583);  mul_968 = unsqueeze_2583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_315: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_819);  add_819 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_62: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_313, relu_314, relu_315, getitem_1263], 1);  relu_313 = relu_314 = relu_315 = getitem_1263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_323: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_62, arg766_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_62 = arg766_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2584: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg767_1, -1);  arg767_1 = None
        unsqueeze_2585: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2584, -1);  unsqueeze_2584 = None
        sub_323: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_323, unsqueeze_2585);  convolution_323 = unsqueeze_2585 = None
        add_820: "f32[1024]" = torch.ops.aten.add.Tensor(arg768_1, 1e-05);  arg768_1 = None
        sqrt_323: "f32[1024]" = torch.ops.aten.sqrt.default(add_820);  add_820 = None
        reciprocal_323: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_323);  sqrt_323 = None
        mul_969: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_323, 1);  reciprocal_323 = None
        unsqueeze_2586: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_969, -1);  mul_969 = None
        unsqueeze_2587: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2586, -1);  unsqueeze_2586 = None
        mul_970: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_2587);  sub_323 = unsqueeze_2587 = None
        unsqueeze_2588: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg769_1, -1);  arg769_1 = None
        unsqueeze_2589: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2588, -1);  unsqueeze_2588 = None
        mul_971: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_970, unsqueeze_2589);  mul_970 = unsqueeze_2589 = None
        unsqueeze_2590: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg770_1, -1);  arg770_1 = None
        unsqueeze_2591: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2590, -1);  unsqueeze_2590 = None
        add_821: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_971, unsqueeze_2591);  mul_971 = unsqueeze_2591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_822: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_821, relu_311);  add_821 = relu_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_316: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_822);  add_822 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_324: "f32[8, 832, 14, 14]" = torch.ops.aten.convolution.default(relu_316, arg771_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg771_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2592: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg772_1, -1);  arg772_1 = None
        unsqueeze_2593: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2592, -1);  unsqueeze_2592 = None
        sub_324: "f32[8, 832, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_324, unsqueeze_2593);  convolution_324 = unsqueeze_2593 = None
        add_823: "f32[832]" = torch.ops.aten.add.Tensor(arg773_1, 1e-05);  arg773_1 = None
        sqrt_324: "f32[832]" = torch.ops.aten.sqrt.default(add_823);  add_823 = None
        reciprocal_324: "f32[832]" = torch.ops.aten.reciprocal.default(sqrt_324);  sqrt_324 = None
        mul_972: "f32[832]" = torch.ops.aten.mul.Tensor(reciprocal_324, 1);  reciprocal_324 = None
        unsqueeze_2594: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(mul_972, -1);  mul_972 = None
        unsqueeze_2595: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2594, -1);  unsqueeze_2594 = None
        mul_973: "f32[8, 832, 14, 14]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_2595);  sub_324 = unsqueeze_2595 = None
        unsqueeze_2596: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg774_1, -1);  arg774_1 = None
        unsqueeze_2597: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2596, -1);  unsqueeze_2596 = None
        mul_974: "f32[8, 832, 14, 14]" = torch.ops.aten.mul.Tensor(mul_973, unsqueeze_2597);  mul_973 = unsqueeze_2597 = None
        unsqueeze_2598: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg775_1, -1);  arg775_1 = None
        unsqueeze_2599: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2598, -1);  unsqueeze_2598 = None
        add_824: "f32[8, 832, 14, 14]" = torch.ops.aten.add.Tensor(mul_974, unsqueeze_2599);  mul_974 = unsqueeze_2599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_317: "f32[8, 832, 14, 14]" = torch.ops.aten.relu.default(add_824);  add_824 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_316 = torch.ops.aten.split.Tensor(relu_317, 208, 1)
        getitem_1268: "f32[8, 208, 14, 14]" = split_316[0];  split_316 = None
        split_317 = torch.ops.aten.split.Tensor(relu_317, 208, 1)
        getitem_1273: "f32[8, 208, 14, 14]" = split_317[1];  split_317 = None
        split_318 = torch.ops.aten.split.Tensor(relu_317, 208, 1)
        getitem_1278: "f32[8, 208, 14, 14]" = split_318[2];  split_318 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:99 in forward, code: spo.append(self.pool(spx[-1]))
        split_319 = torch.ops.aten.split.Tensor(relu_317, 208, 1);  relu_317 = None
        getitem_1283: "f32[8, 208, 14, 14]" = split_319[3];  split_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_325: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_1268, arg776_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1268 = arg776_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2600: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg777_1, -1);  arg777_1 = None
        unsqueeze_2601: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2600, -1);  unsqueeze_2600 = None
        sub_325: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_325, unsqueeze_2601);  convolution_325 = unsqueeze_2601 = None
        add_825: "f32[208]" = torch.ops.aten.add.Tensor(arg778_1, 1e-05);  arg778_1 = None
        sqrt_325: "f32[208]" = torch.ops.aten.sqrt.default(add_825);  add_825 = None
        reciprocal_325: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_325);  sqrt_325 = None
        mul_975: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_325, 1);  reciprocal_325 = None
        unsqueeze_2602: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_975, -1);  mul_975 = None
        unsqueeze_2603: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2602, -1);  unsqueeze_2602 = None
        mul_976: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_2603);  sub_325 = unsqueeze_2603 = None
        unsqueeze_2604: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg779_1, -1);  arg779_1 = None
        unsqueeze_2605: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2604, -1);  unsqueeze_2604 = None
        mul_977: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_976, unsqueeze_2605);  mul_976 = unsqueeze_2605 = None
        unsqueeze_2606: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg780_1, -1);  arg780_1 = None
        unsqueeze_2607: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2606, -1);  unsqueeze_2606 = None
        add_826: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_977, unsqueeze_2607);  mul_977 = unsqueeze_2607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_318: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_826);  add_826 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_326: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_1273, arg781_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1273 = arg781_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2608: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg782_1, -1);  arg782_1 = None
        unsqueeze_2609: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2608, -1);  unsqueeze_2608 = None
        sub_326: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_326, unsqueeze_2609);  convolution_326 = unsqueeze_2609 = None
        add_827: "f32[208]" = torch.ops.aten.add.Tensor(arg783_1, 1e-05);  arg783_1 = None
        sqrt_326: "f32[208]" = torch.ops.aten.sqrt.default(add_827);  add_827 = None
        reciprocal_326: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_326);  sqrt_326 = None
        mul_978: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_326, 1);  reciprocal_326 = None
        unsqueeze_2610: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_978, -1);  mul_978 = None
        unsqueeze_2611: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2610, -1);  unsqueeze_2610 = None
        mul_979: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_2611);  sub_326 = unsqueeze_2611 = None
        unsqueeze_2612: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg784_1, -1);  arg784_1 = None
        unsqueeze_2613: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2612, -1);  unsqueeze_2612 = None
        mul_980: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_979, unsqueeze_2613);  mul_979 = unsqueeze_2613 = None
        unsqueeze_2614: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg785_1, -1);  arg785_1 = None
        unsqueeze_2615: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2614, -1);  unsqueeze_2614 = None
        add_828: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_980, unsqueeze_2615);  mul_980 = unsqueeze_2615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_319: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_828);  add_828 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_327: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_1278, arg786_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1278 = arg786_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2616: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg787_1, -1);  arg787_1 = None
        unsqueeze_2617: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2616, -1);  unsqueeze_2616 = None
        sub_327: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_327, unsqueeze_2617);  convolution_327 = unsqueeze_2617 = None
        add_829: "f32[208]" = torch.ops.aten.add.Tensor(arg788_1, 1e-05);  arg788_1 = None
        sqrt_327: "f32[208]" = torch.ops.aten.sqrt.default(add_829);  add_829 = None
        reciprocal_327: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_327);  sqrt_327 = None
        mul_981: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_327, 1);  reciprocal_327 = None
        unsqueeze_2618: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_981, -1);  mul_981 = None
        unsqueeze_2619: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2618, -1);  unsqueeze_2618 = None
        mul_982: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_2619);  sub_327 = unsqueeze_2619 = None
        unsqueeze_2620: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg789_1, -1);  arg789_1 = None
        unsqueeze_2621: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2620, -1);  unsqueeze_2620 = None
        mul_983: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_982, unsqueeze_2621);  mul_982 = unsqueeze_2621 = None
        unsqueeze_2622: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg790_1, -1);  arg790_1 = None
        unsqueeze_2623: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2622, -1);  unsqueeze_2622 = None
        add_830: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_983, unsqueeze_2623);  mul_983 = unsqueeze_2623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_320: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_830);  add_830 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:99 in forward, code: spo.append(self.pool(spx[-1]))
        avg_pool2d_7: "f32[8, 208, 7, 7]" = torch.ops.aten.avg_pool2d.default(getitem_1283, [3, 3], [2, 2], [1, 1]);  getitem_1283 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_63: "f32[8, 832, 7, 7]" = torch.ops.aten.cat.default([relu_318, relu_319, relu_320, avg_pool2d_7], 1);  relu_318 = relu_319 = relu_320 = avg_pool2d_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_328: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_63, arg791_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_63 = arg791_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2624: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg792_1, -1);  arg792_1 = None
        unsqueeze_2625: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2624, -1);  unsqueeze_2624 = None
        sub_328: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_328, unsqueeze_2625);  convolution_328 = unsqueeze_2625 = None
        add_831: "f32[2048]" = torch.ops.aten.add.Tensor(arg793_1, 1e-05);  arg793_1 = None
        sqrt_328: "f32[2048]" = torch.ops.aten.sqrt.default(add_831);  add_831 = None
        reciprocal_328: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_328);  sqrt_328 = None
        mul_984: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_328, 1);  reciprocal_328 = None
        unsqueeze_2626: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_984, -1);  mul_984 = None
        unsqueeze_2627: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2626, -1);  unsqueeze_2626 = None
        mul_985: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_2627);  sub_328 = unsqueeze_2627 = None
        unsqueeze_2628: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg794_1, -1);  arg794_1 = None
        unsqueeze_2629: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2628, -1);  unsqueeze_2628 = None
        mul_986: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_985, unsqueeze_2629);  mul_985 = unsqueeze_2629 = None
        unsqueeze_2630: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg795_1, -1);  arg795_1 = None
        unsqueeze_2631: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2630, -1);  unsqueeze_2630 = None
        add_832: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_986, unsqueeze_2631);  mul_986 = unsqueeze_2631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:111 in forward, code: shortcut = self.downsample(x)
        convolution_329: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_316, arg796_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_316 = arg796_1 = None
        unsqueeze_2632: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg797_1, -1);  arg797_1 = None
        unsqueeze_2633: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2632, -1);  unsqueeze_2632 = None
        sub_329: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_329, unsqueeze_2633);  convolution_329 = unsqueeze_2633 = None
        add_833: "f32[2048]" = torch.ops.aten.add.Tensor(arg798_1, 1e-05);  arg798_1 = None
        sqrt_329: "f32[2048]" = torch.ops.aten.sqrt.default(add_833);  add_833 = None
        reciprocal_329: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_329);  sqrt_329 = None
        mul_987: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_329, 1);  reciprocal_329 = None
        unsqueeze_2634: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_987, -1);  mul_987 = None
        unsqueeze_2635: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2634, -1);  unsqueeze_2634 = None
        mul_988: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_2635);  sub_329 = unsqueeze_2635 = None
        unsqueeze_2636: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg799_1, -1);  arg799_1 = None
        unsqueeze_2637: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2636, -1);  unsqueeze_2636 = None
        mul_989: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_988, unsqueeze_2637);  mul_988 = unsqueeze_2637 = None
        unsqueeze_2638: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg800_1, -1);  arg800_1 = None
        unsqueeze_2639: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2638, -1);  unsqueeze_2638 = None
        add_834: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_989, unsqueeze_2639);  mul_989 = unsqueeze_2639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_835: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_832, add_834);  add_832 = add_834 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_321: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_835);  add_835 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_330: "f32[8, 832, 7, 7]" = torch.ops.aten.convolution.default(relu_321, arg801_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg801_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2640: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg802_1, -1);  arg802_1 = None
        unsqueeze_2641: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2640, -1);  unsqueeze_2640 = None
        sub_330: "f32[8, 832, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_330, unsqueeze_2641);  convolution_330 = unsqueeze_2641 = None
        add_836: "f32[832]" = torch.ops.aten.add.Tensor(arg803_1, 1e-05);  arg803_1 = None
        sqrt_330: "f32[832]" = torch.ops.aten.sqrt.default(add_836);  add_836 = None
        reciprocal_330: "f32[832]" = torch.ops.aten.reciprocal.default(sqrt_330);  sqrt_330 = None
        mul_990: "f32[832]" = torch.ops.aten.mul.Tensor(reciprocal_330, 1);  reciprocal_330 = None
        unsqueeze_2642: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(mul_990, -1);  mul_990 = None
        unsqueeze_2643: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2642, -1);  unsqueeze_2642 = None
        mul_991: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_2643);  sub_330 = unsqueeze_2643 = None
        unsqueeze_2644: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg804_1, -1);  arg804_1 = None
        unsqueeze_2645: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2644, -1);  unsqueeze_2644 = None
        mul_992: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(mul_991, unsqueeze_2645);  mul_991 = unsqueeze_2645 = None
        unsqueeze_2646: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg805_1, -1);  arg805_1 = None
        unsqueeze_2647: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2646, -1);  unsqueeze_2646 = None
        add_837: "f32[8, 832, 7, 7]" = torch.ops.aten.add.Tensor(mul_992, unsqueeze_2647);  mul_992 = unsqueeze_2647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_322: "f32[8, 832, 7, 7]" = torch.ops.aten.relu.default(add_837);  add_837 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_321 = torch.ops.aten.split.Tensor(relu_322, 208, 1)
        getitem_1288: "f32[8, 208, 7, 7]" = split_321[0];  split_321 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_322 = torch.ops.aten.split.Tensor(relu_322, 208, 1)
        getitem_1293: "f32[8, 208, 7, 7]" = split_322[1];  split_322 = None
        split_323 = torch.ops.aten.split.Tensor(relu_322, 208, 1)
        getitem_1298: "f32[8, 208, 7, 7]" = split_323[2];  split_323 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_324 = torch.ops.aten.split.Tensor(relu_322, 208, 1);  relu_322 = None
        getitem_1303: "f32[8, 208, 7, 7]" = split_324[3];  split_324 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_331: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_1288, arg806_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1288 = arg806_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2648: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg807_1, -1);  arg807_1 = None
        unsqueeze_2649: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2648, -1);  unsqueeze_2648 = None
        sub_331: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_331, unsqueeze_2649);  convolution_331 = unsqueeze_2649 = None
        add_838: "f32[208]" = torch.ops.aten.add.Tensor(arg808_1, 1e-05);  arg808_1 = None
        sqrt_331: "f32[208]" = torch.ops.aten.sqrt.default(add_838);  add_838 = None
        reciprocal_331: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_331);  sqrt_331 = None
        mul_993: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_331, 1);  reciprocal_331 = None
        unsqueeze_2650: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_993, -1);  mul_993 = None
        unsqueeze_2651: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2650, -1);  unsqueeze_2650 = None
        mul_994: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_2651);  sub_331 = unsqueeze_2651 = None
        unsqueeze_2652: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg809_1, -1);  arg809_1 = None
        unsqueeze_2653: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2652, -1);  unsqueeze_2652 = None
        mul_995: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_994, unsqueeze_2653);  mul_994 = unsqueeze_2653 = None
        unsqueeze_2654: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg810_1, -1);  arg810_1 = None
        unsqueeze_2655: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2654, -1);  unsqueeze_2654 = None
        add_839: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_995, unsqueeze_2655);  mul_995 = unsqueeze_2655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_323: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_839);  add_839 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_840: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(relu_323, getitem_1293);  getitem_1293 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_332: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(add_840, arg811_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_840 = arg811_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2656: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg812_1, -1);  arg812_1 = None
        unsqueeze_2657: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2656, -1);  unsqueeze_2656 = None
        sub_332: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_332, unsqueeze_2657);  convolution_332 = unsqueeze_2657 = None
        add_841: "f32[208]" = torch.ops.aten.add.Tensor(arg813_1, 1e-05);  arg813_1 = None
        sqrt_332: "f32[208]" = torch.ops.aten.sqrt.default(add_841);  add_841 = None
        reciprocal_332: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_332);  sqrt_332 = None
        mul_996: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_332, 1);  reciprocal_332 = None
        unsqueeze_2658: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_996, -1);  mul_996 = None
        unsqueeze_2659: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2658, -1);  unsqueeze_2658 = None
        mul_997: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_2659);  sub_332 = unsqueeze_2659 = None
        unsqueeze_2660: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg814_1, -1);  arg814_1 = None
        unsqueeze_2661: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2660, -1);  unsqueeze_2660 = None
        mul_998: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_997, unsqueeze_2661);  mul_997 = unsqueeze_2661 = None
        unsqueeze_2662: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg815_1, -1);  arg815_1 = None
        unsqueeze_2663: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2662, -1);  unsqueeze_2662 = None
        add_842: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_998, unsqueeze_2663);  mul_998 = unsqueeze_2663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_324: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_842);  add_842 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_843: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(relu_324, getitem_1298);  getitem_1298 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_333: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(add_843, arg816_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_843 = arg816_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2664: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg817_1, -1);  arg817_1 = None
        unsqueeze_2665: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2664, -1);  unsqueeze_2664 = None
        sub_333: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_333, unsqueeze_2665);  convolution_333 = unsqueeze_2665 = None
        add_844: "f32[208]" = torch.ops.aten.add.Tensor(arg818_1, 1e-05);  arg818_1 = None
        sqrt_333: "f32[208]" = torch.ops.aten.sqrt.default(add_844);  add_844 = None
        reciprocal_333: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_333);  sqrt_333 = None
        mul_999: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_333, 1);  reciprocal_333 = None
        unsqueeze_2666: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_999, -1);  mul_999 = None
        unsqueeze_2667: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2666, -1);  unsqueeze_2666 = None
        mul_1000: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_2667);  sub_333 = unsqueeze_2667 = None
        unsqueeze_2668: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg819_1, -1);  arg819_1 = None
        unsqueeze_2669: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2668, -1);  unsqueeze_2668 = None
        mul_1001: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1000, unsqueeze_2669);  mul_1000 = unsqueeze_2669 = None
        unsqueeze_2670: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg820_1, -1);  arg820_1 = None
        unsqueeze_2671: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2670, -1);  unsqueeze_2670 = None
        add_845: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_1001, unsqueeze_2671);  mul_1001 = unsqueeze_2671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_325: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_845);  add_845 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_64: "f32[8, 832, 7, 7]" = torch.ops.aten.cat.default([relu_323, relu_324, relu_325, getitem_1303], 1);  relu_323 = relu_324 = relu_325 = getitem_1303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_334: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_64, arg821_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_64 = arg821_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2672: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg822_1, -1);  arg822_1 = None
        unsqueeze_2673: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2672, -1);  unsqueeze_2672 = None
        sub_334: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_334, unsqueeze_2673);  convolution_334 = unsqueeze_2673 = None
        add_846: "f32[2048]" = torch.ops.aten.add.Tensor(arg823_1, 1e-05);  arg823_1 = None
        sqrt_334: "f32[2048]" = torch.ops.aten.sqrt.default(add_846);  add_846 = None
        reciprocal_334: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_334);  sqrt_334 = None
        mul_1002: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_334, 1);  reciprocal_334 = None
        unsqueeze_2674: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_1002, -1);  mul_1002 = None
        unsqueeze_2675: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2674, -1);  unsqueeze_2674 = None
        mul_1003: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_2675);  sub_334 = unsqueeze_2675 = None
        unsqueeze_2676: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg824_1, -1);  arg824_1 = None
        unsqueeze_2677: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2676, -1);  unsqueeze_2676 = None
        mul_1004: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1003, unsqueeze_2677);  mul_1003 = unsqueeze_2677 = None
        unsqueeze_2678: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg825_1, -1);  arg825_1 = None
        unsqueeze_2679: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2678, -1);  unsqueeze_2678 = None
        add_847: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_1004, unsqueeze_2679);  mul_1004 = unsqueeze_2679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_848: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_847, relu_321);  add_847 = relu_321 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_326: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_848);  add_848 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_335: "f32[8, 832, 7, 7]" = torch.ops.aten.convolution.default(relu_326, arg826_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg826_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        unsqueeze_2680: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg827_1, -1);  arg827_1 = None
        unsqueeze_2681: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2680, -1);  unsqueeze_2680 = None
        sub_335: "f32[8, 832, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_335, unsqueeze_2681);  convolution_335 = unsqueeze_2681 = None
        add_849: "f32[832]" = torch.ops.aten.add.Tensor(arg828_1, 1e-05);  arg828_1 = None
        sqrt_335: "f32[832]" = torch.ops.aten.sqrt.default(add_849);  add_849 = None
        reciprocal_335: "f32[832]" = torch.ops.aten.reciprocal.default(sqrt_335);  sqrt_335 = None
        mul_1005: "f32[832]" = torch.ops.aten.mul.Tensor(reciprocal_335, 1);  reciprocal_335 = None
        unsqueeze_2682: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(mul_1005, -1);  mul_1005 = None
        unsqueeze_2683: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2682, -1);  unsqueeze_2682 = None
        mul_1006: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_2683);  sub_335 = unsqueeze_2683 = None
        unsqueeze_2684: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg829_1, -1);  arg829_1 = None
        unsqueeze_2685: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2684, -1);  unsqueeze_2684 = None
        mul_1007: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1006, unsqueeze_2685);  mul_1006 = unsqueeze_2685 = None
        unsqueeze_2686: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg830_1, -1);  arg830_1 = None
        unsqueeze_2687: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2686, -1);  unsqueeze_2686 = None
        add_850: "f32[8, 832, 7, 7]" = torch.ops.aten.add.Tensor(mul_1007, unsqueeze_2687);  mul_1007 = unsqueeze_2687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_327: "f32[8, 832, 7, 7]" = torch.ops.aten.relu.default(add_850);  add_850 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_326 = torch.ops.aten.split.Tensor(relu_327, 208, 1)
        getitem_1308: "f32[8, 208, 7, 7]" = split_326[0];  split_326 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_327 = torch.ops.aten.split.Tensor(relu_327, 208, 1)
        getitem_1313: "f32[8, 208, 7, 7]" = split_327[1];  split_327 = None
        split_328 = torch.ops.aten.split.Tensor(relu_327, 208, 1)
        getitem_1318: "f32[8, 208, 7, 7]" = split_328[2];  split_328 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_329 = torch.ops.aten.split.Tensor(relu_327, 208, 1);  relu_327 = None
        getitem_1323: "f32[8, 208, 7, 7]" = split_329[3];  split_329 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_336: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_1308, arg831_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1308 = arg831_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2688: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg832_1, -1);  arg832_1 = None
        unsqueeze_2689: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2688, -1);  unsqueeze_2688 = None
        sub_336: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_336, unsqueeze_2689);  convolution_336 = unsqueeze_2689 = None
        add_851: "f32[208]" = torch.ops.aten.add.Tensor(arg833_1, 1e-05);  arg833_1 = None
        sqrt_336: "f32[208]" = torch.ops.aten.sqrt.default(add_851);  add_851 = None
        reciprocal_336: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_336);  sqrt_336 = None
        mul_1008: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_336, 1);  reciprocal_336 = None
        unsqueeze_2690: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_1008, -1);  mul_1008 = None
        unsqueeze_2691: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2690, -1);  unsqueeze_2690 = None
        mul_1009: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_2691);  sub_336 = unsqueeze_2691 = None
        unsqueeze_2692: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg834_1, -1);  arg834_1 = None
        unsqueeze_2693: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2692, -1);  unsqueeze_2692 = None
        mul_1010: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1009, unsqueeze_2693);  mul_1009 = unsqueeze_2693 = None
        unsqueeze_2694: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg835_1, -1);  arg835_1 = None
        unsqueeze_2695: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2694, -1);  unsqueeze_2694 = None
        add_852: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_1010, unsqueeze_2695);  mul_1010 = unsqueeze_2695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_328: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_852);  add_852 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_853: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(relu_328, getitem_1313);  getitem_1313 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_337: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(add_853, arg836_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_853 = arg836_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2696: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg837_1, -1);  arg837_1 = None
        unsqueeze_2697: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2696, -1);  unsqueeze_2696 = None
        sub_337: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_337, unsqueeze_2697);  convolution_337 = unsqueeze_2697 = None
        add_854: "f32[208]" = torch.ops.aten.add.Tensor(arg838_1, 1e-05);  arg838_1 = None
        sqrt_337: "f32[208]" = torch.ops.aten.sqrt.default(add_854);  add_854 = None
        reciprocal_337: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_337);  sqrt_337 = None
        mul_1011: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_337, 1);  reciprocal_337 = None
        unsqueeze_2698: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_1011, -1);  mul_1011 = None
        unsqueeze_2699: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2698, -1);  unsqueeze_2698 = None
        mul_1012: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_2699);  sub_337 = unsqueeze_2699 = None
        unsqueeze_2700: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg839_1, -1);  arg839_1 = None
        unsqueeze_2701: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2700, -1);  unsqueeze_2700 = None
        mul_1013: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1012, unsqueeze_2701);  mul_1012 = unsqueeze_2701 = None
        unsqueeze_2702: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg840_1, -1);  arg840_1 = None
        unsqueeze_2703: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2702, -1);  unsqueeze_2702 = None
        add_855: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_1013, unsqueeze_2703);  mul_1013 = unsqueeze_2703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_329: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_855);  add_855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        add_856: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(relu_329, getitem_1318);  getitem_1318 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_338: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(add_856, arg841_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_856 = arg841_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        unsqueeze_2704: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg842_1, -1);  arg842_1 = None
        unsqueeze_2705: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2704, -1);  unsqueeze_2704 = None
        sub_338: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_338, unsqueeze_2705);  convolution_338 = unsqueeze_2705 = None
        add_857: "f32[208]" = torch.ops.aten.add.Tensor(arg843_1, 1e-05);  arg843_1 = None
        sqrt_338: "f32[208]" = torch.ops.aten.sqrt.default(add_857);  add_857 = None
        reciprocal_338: "f32[208]" = torch.ops.aten.reciprocal.default(sqrt_338);  sqrt_338 = None
        mul_1014: "f32[208]" = torch.ops.aten.mul.Tensor(reciprocal_338, 1);  reciprocal_338 = None
        unsqueeze_2706: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(mul_1014, -1);  mul_1014 = None
        unsqueeze_2707: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2706, -1);  unsqueeze_2706 = None
        mul_1015: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_2707);  sub_338 = unsqueeze_2707 = None
        unsqueeze_2708: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg844_1, -1);  arg844_1 = None
        unsqueeze_2709: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2708, -1);  unsqueeze_2708 = None
        mul_1016: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1015, unsqueeze_2709);  mul_1015 = unsqueeze_2709 = None
        unsqueeze_2710: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(arg845_1, -1);  arg845_1 = None
        unsqueeze_2711: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2710, -1);  unsqueeze_2710 = None
        add_858: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_1016, unsqueeze_2711);  mul_1016 = unsqueeze_2711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_330: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_858);  add_858 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_65: "f32[8, 832, 7, 7]" = torch.ops.aten.cat.default([relu_328, relu_329, relu_330, getitem_1323], 1);  relu_328 = relu_329 = relu_330 = getitem_1323 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_339: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_65, arg846_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_65 = arg846_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        unsqueeze_2712: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg847_1, -1);  arg847_1 = None
        unsqueeze_2713: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2712, -1);  unsqueeze_2712 = None
        sub_339: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_339, unsqueeze_2713);  convolution_339 = unsqueeze_2713 = None
        add_859: "f32[2048]" = torch.ops.aten.add.Tensor(arg848_1, 1e-05);  arg848_1 = None
        sqrt_339: "f32[2048]" = torch.ops.aten.sqrt.default(add_859);  add_859 = None
        reciprocal_339: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_339);  sqrt_339 = None
        mul_1017: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_339, 1);  reciprocal_339 = None
        unsqueeze_2714: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_1017, -1);  mul_1017 = None
        unsqueeze_2715: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2714, -1);  unsqueeze_2714 = None
        mul_1018: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_2715);  sub_339 = unsqueeze_2715 = None
        unsqueeze_2716: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg849_1, -1);  arg849_1 = None
        unsqueeze_2717: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2716, -1);  unsqueeze_2716 = None
        mul_1019: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1018, unsqueeze_2717);  mul_1018 = unsqueeze_2717 = None
        unsqueeze_2718: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg850_1, -1);  arg850_1 = None
        unsqueeze_2719: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2718, -1);  unsqueeze_2718 = None
        add_860: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_1019, unsqueeze_2719);  mul_1019 = unsqueeze_2719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_861: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_860, relu_326);  add_860 = relu_326 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_331: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_861);  add_861 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_1: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_331, [-1, -2], True);  relu_331 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_1: "f32[8, 2048]" = torch.ops.aten.reshape.default(mean_1, [8, 2048]);  mean_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:633 in forward_head, code: return x if pre_logits else self.fc(x)
        permute_1: "f32[2048, 1000]" = torch.ops.aten.permute.default(arg851_1, [1, 0]);  arg851_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg852_1, view_1, permute_1);  arg852_1 = view_1 = permute_1 = None
        return (addmm_1,)
        