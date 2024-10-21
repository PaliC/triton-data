class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 3, 7, 7]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[112, 64, 1, 1]", arg7_1: "f32[112]", arg8_1: "f32[112]", arg9_1: "f32[112]", arg10_1: "f32[112]", arg11_1: "f32[14, 14, 3, 3]", arg12_1: "f32[14]", arg13_1: "f32[14]", arg14_1: "f32[14]", arg15_1: "f32[14]", arg16_1: "f32[14, 14, 3, 3]", arg17_1: "f32[14]", arg18_1: "f32[14]", arg19_1: "f32[14]", arg20_1: "f32[14]", arg21_1: "f32[14, 14, 3, 3]", arg22_1: "f32[14]", arg23_1: "f32[14]", arg24_1: "f32[14]", arg25_1: "f32[14]", arg26_1: "f32[14, 14, 3, 3]", arg27_1: "f32[14]", arg28_1: "f32[14]", arg29_1: "f32[14]", arg30_1: "f32[14]", arg31_1: "f32[14, 14, 3, 3]", arg32_1: "f32[14]", arg33_1: "f32[14]", arg34_1: "f32[14]", arg35_1: "f32[14]", arg36_1: "f32[14, 14, 3, 3]", arg37_1: "f32[14]", arg38_1: "f32[14]", arg39_1: "f32[14]", arg40_1: "f32[14]", arg41_1: "f32[14, 14, 3, 3]", arg42_1: "f32[14]", arg43_1: "f32[14]", arg44_1: "f32[14]", arg45_1: "f32[14]", arg46_1: "f32[256, 112, 1, 1]", arg47_1: "f32[256]", arg48_1: "f32[256]", arg49_1: "f32[256]", arg50_1: "f32[256]", arg51_1: "f32[256, 64, 1, 1]", arg52_1: "f32[256]", arg53_1: "f32[256]", arg54_1: "f32[256]", arg55_1: "f32[256]", arg56_1: "f32[112, 256, 1, 1]", arg57_1: "f32[112]", arg58_1: "f32[112]", arg59_1: "f32[112]", arg60_1: "f32[112]", arg61_1: "f32[14, 14, 3, 3]", arg62_1: "f32[14]", arg63_1: "f32[14]", arg64_1: "f32[14]", arg65_1: "f32[14]", arg66_1: "f32[14, 14, 3, 3]", arg67_1: "f32[14]", arg68_1: "f32[14]", arg69_1: "f32[14]", arg70_1: "f32[14]", arg71_1: "f32[14, 14, 3, 3]", arg72_1: "f32[14]", arg73_1: "f32[14]", arg74_1: "f32[14]", arg75_1: "f32[14]", arg76_1: "f32[14, 14, 3, 3]", arg77_1: "f32[14]", arg78_1: "f32[14]", arg79_1: "f32[14]", arg80_1: "f32[14]", arg81_1: "f32[14, 14, 3, 3]", arg82_1: "f32[14]", arg83_1: "f32[14]", arg84_1: "f32[14]", arg85_1: "f32[14]", arg86_1: "f32[14, 14, 3, 3]", arg87_1: "f32[14]", arg88_1: "f32[14]", arg89_1: "f32[14]", arg90_1: "f32[14]", arg91_1: "f32[14, 14, 3, 3]", arg92_1: "f32[14]", arg93_1: "f32[14]", arg94_1: "f32[14]", arg95_1: "f32[14]", arg96_1: "f32[256, 112, 1, 1]", arg97_1: "f32[256]", arg98_1: "f32[256]", arg99_1: "f32[256]", arg100_1: "f32[256]", arg101_1: "f32[112, 256, 1, 1]", arg102_1: "f32[112]", arg103_1: "f32[112]", arg104_1: "f32[112]", arg105_1: "f32[112]", arg106_1: "f32[14, 14, 3, 3]", arg107_1: "f32[14]", arg108_1: "f32[14]", arg109_1: "f32[14]", arg110_1: "f32[14]", arg111_1: "f32[14, 14, 3, 3]", arg112_1: "f32[14]", arg113_1: "f32[14]", arg114_1: "f32[14]", arg115_1: "f32[14]", arg116_1: "f32[14, 14, 3, 3]", arg117_1: "f32[14]", arg118_1: "f32[14]", arg119_1: "f32[14]", arg120_1: "f32[14]", arg121_1: "f32[14, 14, 3, 3]", arg122_1: "f32[14]", arg123_1: "f32[14]", arg124_1: "f32[14]", arg125_1: "f32[14]", arg126_1: "f32[14, 14, 3, 3]", arg127_1: "f32[14]", arg128_1: "f32[14]", arg129_1: "f32[14]", arg130_1: "f32[14]", arg131_1: "f32[14, 14, 3, 3]", arg132_1: "f32[14]", arg133_1: "f32[14]", arg134_1: "f32[14]", arg135_1: "f32[14]", arg136_1: "f32[14, 14, 3, 3]", arg137_1: "f32[14]", arg138_1: "f32[14]", arg139_1: "f32[14]", arg140_1: "f32[14]", arg141_1: "f32[256, 112, 1, 1]", arg142_1: "f32[256]", arg143_1: "f32[256]", arg144_1: "f32[256]", arg145_1: "f32[256]", arg146_1: "f32[224, 256, 1, 1]", arg147_1: "f32[224]", arg148_1: "f32[224]", arg149_1: "f32[224]", arg150_1: "f32[224]", arg151_1: "f32[28, 28, 3, 3]", arg152_1: "f32[28]", arg153_1: "f32[28]", arg154_1: "f32[28]", arg155_1: "f32[28]", arg156_1: "f32[28, 28, 3, 3]", arg157_1: "f32[28]", arg158_1: "f32[28]", arg159_1: "f32[28]", arg160_1: "f32[28]", arg161_1: "f32[28, 28, 3, 3]", arg162_1: "f32[28]", arg163_1: "f32[28]", arg164_1: "f32[28]", arg165_1: "f32[28]", arg166_1: "f32[28, 28, 3, 3]", arg167_1: "f32[28]", arg168_1: "f32[28]", arg169_1: "f32[28]", arg170_1: "f32[28]", arg171_1: "f32[28, 28, 3, 3]", arg172_1: "f32[28]", arg173_1: "f32[28]", arg174_1: "f32[28]", arg175_1: "f32[28]", arg176_1: "f32[28, 28, 3, 3]", arg177_1: "f32[28]", arg178_1: "f32[28]", arg179_1: "f32[28]", arg180_1: "f32[28]", arg181_1: "f32[28, 28, 3, 3]", arg182_1: "f32[28]", arg183_1: "f32[28]", arg184_1: "f32[28]", arg185_1: "f32[28]", arg186_1: "f32[512, 224, 1, 1]", arg187_1: "f32[512]", arg188_1: "f32[512]", arg189_1: "f32[512]", arg190_1: "f32[512]", arg191_1: "f32[512, 256, 1, 1]", arg192_1: "f32[512]", arg193_1: "f32[512]", arg194_1: "f32[512]", arg195_1: "f32[512]", arg196_1: "f32[224, 512, 1, 1]", arg197_1: "f32[224]", arg198_1: "f32[224]", arg199_1: "f32[224]", arg200_1: "f32[224]", arg201_1: "f32[28, 28, 3, 3]", arg202_1: "f32[28]", arg203_1: "f32[28]", arg204_1: "f32[28]", arg205_1: "f32[28]", arg206_1: "f32[28, 28, 3, 3]", arg207_1: "f32[28]", arg208_1: "f32[28]", arg209_1: "f32[28]", arg210_1: "f32[28]", arg211_1: "f32[28, 28, 3, 3]", arg212_1: "f32[28]", arg213_1: "f32[28]", arg214_1: "f32[28]", arg215_1: "f32[28]", arg216_1: "f32[28, 28, 3, 3]", arg217_1: "f32[28]", arg218_1: "f32[28]", arg219_1: "f32[28]", arg220_1: "f32[28]", arg221_1: "f32[28, 28, 3, 3]", arg222_1: "f32[28]", arg223_1: "f32[28]", arg224_1: "f32[28]", arg225_1: "f32[28]", arg226_1: "f32[28, 28, 3, 3]", arg227_1: "f32[28]", arg228_1: "f32[28]", arg229_1: "f32[28]", arg230_1: "f32[28]", arg231_1: "f32[28, 28, 3, 3]", arg232_1: "f32[28]", arg233_1: "f32[28]", arg234_1: "f32[28]", arg235_1: "f32[28]", arg236_1: "f32[512, 224, 1, 1]", arg237_1: "f32[512]", arg238_1: "f32[512]", arg239_1: "f32[512]", arg240_1: "f32[512]", arg241_1: "f32[224, 512, 1, 1]", arg242_1: "f32[224]", arg243_1: "f32[224]", arg244_1: "f32[224]", arg245_1: "f32[224]", arg246_1: "f32[28, 28, 3, 3]", arg247_1: "f32[28]", arg248_1: "f32[28]", arg249_1: "f32[28]", arg250_1: "f32[28]", arg251_1: "f32[28, 28, 3, 3]", arg252_1: "f32[28]", arg253_1: "f32[28]", arg254_1: "f32[28]", arg255_1: "f32[28]", arg256_1: "f32[28, 28, 3, 3]", arg257_1: "f32[28]", arg258_1: "f32[28]", arg259_1: "f32[28]", arg260_1: "f32[28]", arg261_1: "f32[28, 28, 3, 3]", arg262_1: "f32[28]", arg263_1: "f32[28]", arg264_1: "f32[28]", arg265_1: "f32[28]", arg266_1: "f32[28, 28, 3, 3]", arg267_1: "f32[28]", arg268_1: "f32[28]", arg269_1: "f32[28]", arg270_1: "f32[28]", arg271_1: "f32[28, 28, 3, 3]", arg272_1: "f32[28]", arg273_1: "f32[28]", arg274_1: "f32[28]", arg275_1: "f32[28]", arg276_1: "f32[28, 28, 3, 3]", arg277_1: "f32[28]", arg278_1: "f32[28]", arg279_1: "f32[28]", arg280_1: "f32[28]", arg281_1: "f32[512, 224, 1, 1]", arg282_1: "f32[512]", arg283_1: "f32[512]", arg284_1: "f32[512]", arg285_1: "f32[512]", arg286_1: "f32[224, 512, 1, 1]", arg287_1: "f32[224]", arg288_1: "f32[224]", arg289_1: "f32[224]", arg290_1: "f32[224]", arg291_1: "f32[28, 28, 3, 3]", arg292_1: "f32[28]", arg293_1: "f32[28]", arg294_1: "f32[28]", arg295_1: "f32[28]", arg296_1: "f32[28, 28, 3, 3]", arg297_1: "f32[28]", arg298_1: "f32[28]", arg299_1: "f32[28]", arg300_1: "f32[28]", arg301_1: "f32[28, 28, 3, 3]", arg302_1: "f32[28]", arg303_1: "f32[28]", arg304_1: "f32[28]", arg305_1: "f32[28]", arg306_1: "f32[28, 28, 3, 3]", arg307_1: "f32[28]", arg308_1: "f32[28]", arg309_1: "f32[28]", arg310_1: "f32[28]", arg311_1: "f32[28, 28, 3, 3]", arg312_1: "f32[28]", arg313_1: "f32[28]", arg314_1: "f32[28]", arg315_1: "f32[28]", arg316_1: "f32[28, 28, 3, 3]", arg317_1: "f32[28]", arg318_1: "f32[28]", arg319_1: "f32[28]", arg320_1: "f32[28]", arg321_1: "f32[28, 28, 3, 3]", arg322_1: "f32[28]", arg323_1: "f32[28]", arg324_1: "f32[28]", arg325_1: "f32[28]", arg326_1: "f32[512, 224, 1, 1]", arg327_1: "f32[512]", arg328_1: "f32[512]", arg329_1: "f32[512]", arg330_1: "f32[512]", arg331_1: "f32[448, 512, 1, 1]", arg332_1: "f32[448]", arg333_1: "f32[448]", arg334_1: "f32[448]", arg335_1: "f32[448]", arg336_1: "f32[56, 56, 3, 3]", arg337_1: "f32[56]", arg338_1: "f32[56]", arg339_1: "f32[56]", arg340_1: "f32[56]", arg341_1: "f32[56, 56, 3, 3]", arg342_1: "f32[56]", arg343_1: "f32[56]", arg344_1: "f32[56]", arg345_1: "f32[56]", arg346_1: "f32[56, 56, 3, 3]", arg347_1: "f32[56]", arg348_1: "f32[56]", arg349_1: "f32[56]", arg350_1: "f32[56]", arg351_1: "f32[56, 56, 3, 3]", arg352_1: "f32[56]", arg353_1: "f32[56]", arg354_1: "f32[56]", arg355_1: "f32[56]", arg356_1: "f32[56, 56, 3, 3]", arg357_1: "f32[56]", arg358_1: "f32[56]", arg359_1: "f32[56]", arg360_1: "f32[56]", arg361_1: "f32[56, 56, 3, 3]", arg362_1: "f32[56]", arg363_1: "f32[56]", arg364_1: "f32[56]", arg365_1: "f32[56]", arg366_1: "f32[56, 56, 3, 3]", arg367_1: "f32[56]", arg368_1: "f32[56]", arg369_1: "f32[56]", arg370_1: "f32[56]", arg371_1: "f32[1024, 448, 1, 1]", arg372_1: "f32[1024]", arg373_1: "f32[1024]", arg374_1: "f32[1024]", arg375_1: "f32[1024]", arg376_1: "f32[1024, 512, 1, 1]", arg377_1: "f32[1024]", arg378_1: "f32[1024]", arg379_1: "f32[1024]", arg380_1: "f32[1024]", arg381_1: "f32[448, 1024, 1, 1]", arg382_1: "f32[448]", arg383_1: "f32[448]", arg384_1: "f32[448]", arg385_1: "f32[448]", arg386_1: "f32[56, 56, 3, 3]", arg387_1: "f32[56]", arg388_1: "f32[56]", arg389_1: "f32[56]", arg390_1: "f32[56]", arg391_1: "f32[56, 56, 3, 3]", arg392_1: "f32[56]", arg393_1: "f32[56]", arg394_1: "f32[56]", arg395_1: "f32[56]", arg396_1: "f32[56, 56, 3, 3]", arg397_1: "f32[56]", arg398_1: "f32[56]", arg399_1: "f32[56]", arg400_1: "f32[56]", arg401_1: "f32[56, 56, 3, 3]", arg402_1: "f32[56]", arg403_1: "f32[56]", arg404_1: "f32[56]", arg405_1: "f32[56]", arg406_1: "f32[56, 56, 3, 3]", arg407_1: "f32[56]", arg408_1: "f32[56]", arg409_1: "f32[56]", arg410_1: "f32[56]", arg411_1: "f32[56, 56, 3, 3]", arg412_1: "f32[56]", arg413_1: "f32[56]", arg414_1: "f32[56]", arg415_1: "f32[56]", arg416_1: "f32[56, 56, 3, 3]", arg417_1: "f32[56]", arg418_1: "f32[56]", arg419_1: "f32[56]", arg420_1: "f32[56]", arg421_1: "f32[1024, 448, 1, 1]", arg422_1: "f32[1024]", arg423_1: "f32[1024]", arg424_1: "f32[1024]", arg425_1: "f32[1024]", arg426_1: "f32[448, 1024, 1, 1]", arg427_1: "f32[448]", arg428_1: "f32[448]", arg429_1: "f32[448]", arg430_1: "f32[448]", arg431_1: "f32[56, 56, 3, 3]", arg432_1: "f32[56]", arg433_1: "f32[56]", arg434_1: "f32[56]", arg435_1: "f32[56]", arg436_1: "f32[56, 56, 3, 3]", arg437_1: "f32[56]", arg438_1: "f32[56]", arg439_1: "f32[56]", arg440_1: "f32[56]", arg441_1: "f32[56, 56, 3, 3]", arg442_1: "f32[56]", arg443_1: "f32[56]", arg444_1: "f32[56]", arg445_1: "f32[56]", arg446_1: "f32[56, 56, 3, 3]", arg447_1: "f32[56]", arg448_1: "f32[56]", arg449_1: "f32[56]", arg450_1: "f32[56]", arg451_1: "f32[56, 56, 3, 3]", arg452_1: "f32[56]", arg453_1: "f32[56]", arg454_1: "f32[56]", arg455_1: "f32[56]", arg456_1: "f32[56, 56, 3, 3]", arg457_1: "f32[56]", arg458_1: "f32[56]", arg459_1: "f32[56]", arg460_1: "f32[56]", arg461_1: "f32[56, 56, 3, 3]", arg462_1: "f32[56]", arg463_1: "f32[56]", arg464_1: "f32[56]", arg465_1: "f32[56]", arg466_1: "f32[1024, 448, 1, 1]", arg467_1: "f32[1024]", arg468_1: "f32[1024]", arg469_1: "f32[1024]", arg470_1: "f32[1024]", arg471_1: "f32[448, 1024, 1, 1]", arg472_1: "f32[448]", arg473_1: "f32[448]", arg474_1: "f32[448]", arg475_1: "f32[448]", arg476_1: "f32[56, 56, 3, 3]", arg477_1: "f32[56]", arg478_1: "f32[56]", arg479_1: "f32[56]", arg480_1: "f32[56]", arg481_1: "f32[56, 56, 3, 3]", arg482_1: "f32[56]", arg483_1: "f32[56]", arg484_1: "f32[56]", arg485_1: "f32[56]", arg486_1: "f32[56, 56, 3, 3]", arg487_1: "f32[56]", arg488_1: "f32[56]", arg489_1: "f32[56]", arg490_1: "f32[56]", arg491_1: "f32[56, 56, 3, 3]", arg492_1: "f32[56]", arg493_1: "f32[56]", arg494_1: "f32[56]", arg495_1: "f32[56]", arg496_1: "f32[56, 56, 3, 3]", arg497_1: "f32[56]", arg498_1: "f32[56]", arg499_1: "f32[56]", arg500_1: "f32[56]", arg501_1: "f32[56, 56, 3, 3]", arg502_1: "f32[56]", arg503_1: "f32[56]", arg504_1: "f32[56]", arg505_1: "f32[56]", arg506_1: "f32[56, 56, 3, 3]", arg507_1: "f32[56]", arg508_1: "f32[56]", arg509_1: "f32[56]", arg510_1: "f32[56]", arg511_1: "f32[1024, 448, 1, 1]", arg512_1: "f32[1024]", arg513_1: "f32[1024]", arg514_1: "f32[1024]", arg515_1: "f32[1024]", arg516_1: "f32[448, 1024, 1, 1]", arg517_1: "f32[448]", arg518_1: "f32[448]", arg519_1: "f32[448]", arg520_1: "f32[448]", arg521_1: "f32[56, 56, 3, 3]", arg522_1: "f32[56]", arg523_1: "f32[56]", arg524_1: "f32[56]", arg525_1: "f32[56]", arg526_1: "f32[56, 56, 3, 3]", arg527_1: "f32[56]", arg528_1: "f32[56]", arg529_1: "f32[56]", arg530_1: "f32[56]", arg531_1: "f32[56, 56, 3, 3]", arg532_1: "f32[56]", arg533_1: "f32[56]", arg534_1: "f32[56]", arg535_1: "f32[56]", arg536_1: "f32[56, 56, 3, 3]", arg537_1: "f32[56]", arg538_1: "f32[56]", arg539_1: "f32[56]", arg540_1: "f32[56]", arg541_1: "f32[56, 56, 3, 3]", arg542_1: "f32[56]", arg543_1: "f32[56]", arg544_1: "f32[56]", arg545_1: "f32[56]", arg546_1: "f32[56, 56, 3, 3]", arg547_1: "f32[56]", arg548_1: "f32[56]", arg549_1: "f32[56]", arg550_1: "f32[56]", arg551_1: "f32[56, 56, 3, 3]", arg552_1: "f32[56]", arg553_1: "f32[56]", arg554_1: "f32[56]", arg555_1: "f32[56]", arg556_1: "f32[1024, 448, 1, 1]", arg557_1: "f32[1024]", arg558_1: "f32[1024]", arg559_1: "f32[1024]", arg560_1: "f32[1024]", arg561_1: "f32[448, 1024, 1, 1]", arg562_1: "f32[448]", arg563_1: "f32[448]", arg564_1: "f32[448]", arg565_1: "f32[448]", arg566_1: "f32[56, 56, 3, 3]", arg567_1: "f32[56]", arg568_1: "f32[56]", arg569_1: "f32[56]", arg570_1: "f32[56]", arg571_1: "f32[56, 56, 3, 3]", arg572_1: "f32[56]", arg573_1: "f32[56]", arg574_1: "f32[56]", arg575_1: "f32[56]", arg576_1: "f32[56, 56, 3, 3]", arg577_1: "f32[56]", arg578_1: "f32[56]", arg579_1: "f32[56]", arg580_1: "f32[56]", arg581_1: "f32[56, 56, 3, 3]", arg582_1: "f32[56]", arg583_1: "f32[56]", arg584_1: "f32[56]", arg585_1: "f32[56]", arg586_1: "f32[56, 56, 3, 3]", arg587_1: "f32[56]", arg588_1: "f32[56]", arg589_1: "f32[56]", arg590_1: "f32[56]", arg591_1: "f32[56, 56, 3, 3]", arg592_1: "f32[56]", arg593_1: "f32[56]", arg594_1: "f32[56]", arg595_1: "f32[56]", arg596_1: "f32[56, 56, 3, 3]", arg597_1: "f32[56]", arg598_1: "f32[56]", arg599_1: "f32[56]", arg600_1: "f32[56]", arg601_1: "f32[1024, 448, 1, 1]", arg602_1: "f32[1024]", arg603_1: "f32[1024]", arg604_1: "f32[1024]", arg605_1: "f32[1024]", arg606_1: "f32[896, 1024, 1, 1]", arg607_1: "f32[896]", arg608_1: "f32[896]", arg609_1: "f32[896]", arg610_1: "f32[896]", arg611_1: "f32[112, 112, 3, 3]", arg612_1: "f32[112]", arg613_1: "f32[112]", arg614_1: "f32[112]", arg615_1: "f32[112]", arg616_1: "f32[112, 112, 3, 3]", arg617_1: "f32[112]", arg618_1: "f32[112]", arg619_1: "f32[112]", arg620_1: "f32[112]", arg621_1: "f32[112, 112, 3, 3]", arg622_1: "f32[112]", arg623_1: "f32[112]", arg624_1: "f32[112]", arg625_1: "f32[112]", arg626_1: "f32[112, 112, 3, 3]", arg627_1: "f32[112]", arg628_1: "f32[112]", arg629_1: "f32[112]", arg630_1: "f32[112]", arg631_1: "f32[112, 112, 3, 3]", arg632_1: "f32[112]", arg633_1: "f32[112]", arg634_1: "f32[112]", arg635_1: "f32[112]", arg636_1: "f32[112, 112, 3, 3]", arg637_1: "f32[112]", arg638_1: "f32[112]", arg639_1: "f32[112]", arg640_1: "f32[112]", arg641_1: "f32[112, 112, 3, 3]", arg642_1: "f32[112]", arg643_1: "f32[112]", arg644_1: "f32[112]", arg645_1: "f32[112]", arg646_1: "f32[2048, 896, 1, 1]", arg647_1: "f32[2048]", arg648_1: "f32[2048]", arg649_1: "f32[2048]", arg650_1: "f32[2048]", arg651_1: "f32[2048, 1024, 1, 1]", arg652_1: "f32[2048]", arg653_1: "f32[2048]", arg654_1: "f32[2048]", arg655_1: "f32[2048]", arg656_1: "f32[896, 2048, 1, 1]", arg657_1: "f32[896]", arg658_1: "f32[896]", arg659_1: "f32[896]", arg660_1: "f32[896]", arg661_1: "f32[112, 112, 3, 3]", arg662_1: "f32[112]", arg663_1: "f32[112]", arg664_1: "f32[112]", arg665_1: "f32[112]", arg666_1: "f32[112, 112, 3, 3]", arg667_1: "f32[112]", arg668_1: "f32[112]", arg669_1: "f32[112]", arg670_1: "f32[112]", arg671_1: "f32[112, 112, 3, 3]", arg672_1: "f32[112]", arg673_1: "f32[112]", arg674_1: "f32[112]", arg675_1: "f32[112]", arg676_1: "f32[112, 112, 3, 3]", arg677_1: "f32[112]", arg678_1: "f32[112]", arg679_1: "f32[112]", arg680_1: "f32[112]", arg681_1: "f32[112, 112, 3, 3]", arg682_1: "f32[112]", arg683_1: "f32[112]", arg684_1: "f32[112]", arg685_1: "f32[112]", arg686_1: "f32[112, 112, 3, 3]", arg687_1: "f32[112]", arg688_1: "f32[112]", arg689_1: "f32[112]", arg690_1: "f32[112]", arg691_1: "f32[112, 112, 3, 3]", arg692_1: "f32[112]", arg693_1: "f32[112]", arg694_1: "f32[112]", arg695_1: "f32[112]", arg696_1: "f32[2048, 896, 1, 1]", arg697_1: "f32[2048]", arg698_1: "f32[2048]", arg699_1: "f32[2048]", arg700_1: "f32[2048]", arg701_1: "f32[896, 2048, 1, 1]", arg702_1: "f32[896]", arg703_1: "f32[896]", arg704_1: "f32[896]", arg705_1: "f32[896]", arg706_1: "f32[112, 112, 3, 3]", arg707_1: "f32[112]", arg708_1: "f32[112]", arg709_1: "f32[112]", arg710_1: "f32[112]", arg711_1: "f32[112, 112, 3, 3]", arg712_1: "f32[112]", arg713_1: "f32[112]", arg714_1: "f32[112]", arg715_1: "f32[112]", arg716_1: "f32[112, 112, 3, 3]", arg717_1: "f32[112]", arg718_1: "f32[112]", arg719_1: "f32[112]", arg720_1: "f32[112]", arg721_1: "f32[112, 112, 3, 3]", arg722_1: "f32[112]", arg723_1: "f32[112]", arg724_1: "f32[112]", arg725_1: "f32[112]", arg726_1: "f32[112, 112, 3, 3]", arg727_1: "f32[112]", arg728_1: "f32[112]", arg729_1: "f32[112]", arg730_1: "f32[112]", arg731_1: "f32[112, 112, 3, 3]", arg732_1: "f32[112]", arg733_1: "f32[112]", arg734_1: "f32[112]", arg735_1: "f32[112]", arg736_1: "f32[112, 112, 3, 3]", arg737_1: "f32[112]", arg738_1: "f32[112]", arg739_1: "f32[112]", arg740_1: "f32[112]", arg741_1: "f32[2048, 896, 1, 1]", arg742_1: "f32[2048]", arg743_1: "f32[2048]", arg744_1: "f32[2048]", arg745_1: "f32[2048]", arg746_1: "f32[1000, 2048]", arg747_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:615 in forward_features, code: x = self.conv1(x)
        convolution_149: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:616 in forward_features, code: x = self.bn1(x)
        add_386: "f32[64]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_149: "f32[64]" = torch.ops.aten.sqrt.default(add_386);  add_386 = None
        reciprocal_149: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_149);  sqrt_149 = None
        mul_447: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_149, 1);  reciprocal_149 = None
        unsqueeze_1192: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_1193: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, -1);  unsqueeze_1192 = None
        unsqueeze_1194: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_447, -1);  mul_447 = None
        unsqueeze_1195: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, -1);  unsqueeze_1194 = None
        sub_149: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_149, unsqueeze_1193);  convolution_149 = unsqueeze_1193 = None
        mul_448: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_1195);  sub_149 = unsqueeze_1195 = None
        unsqueeze_1196: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_1197: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, -1);  unsqueeze_1196 = None
        mul_449: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_1197);  mul_448 = unsqueeze_1197 = None
        unsqueeze_1198: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_1199: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, -1);  unsqueeze_1198 = None
        add_387: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_449, unsqueeze_1199);  mul_449 = unsqueeze_1199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:617 in forward_features, code: x = self.act1(x)
        relu_145: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_387);  add_387 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:618 in forward_features, code: x = self.maxpool(x)
        _low_memory_max_pool2d_with_offsets_1 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_145, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu_145 = None
        getitem_1154: "f32[8, 64, 56, 56]" = _low_memory_max_pool2d_with_offsets_1[0];  _low_memory_max_pool2d_with_offsets_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_150: "f32[8, 112, 56, 56]" = torch.ops.aten.convolution.default(getitem_1154, arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_388: "f32[112]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_150: "f32[112]" = torch.ops.aten.sqrt.default(add_388);  add_388 = None
        reciprocal_150: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_150);  sqrt_150 = None
        mul_450: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_150, 1);  reciprocal_150 = None
        unsqueeze_1200: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_1201: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, -1);  unsqueeze_1200 = None
        unsqueeze_1202: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_450, -1);  mul_450 = None
        unsqueeze_1203: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, -1);  unsqueeze_1202 = None
        sub_150: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_1201);  convolution_150 = unsqueeze_1201 = None
        mul_451: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_1203);  sub_150 = unsqueeze_1203 = None
        unsqueeze_1204: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_1205: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, -1);  unsqueeze_1204 = None
        mul_452: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(mul_451, unsqueeze_1205);  mul_451 = unsqueeze_1205 = None
        unsqueeze_1206: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_1207: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, -1);  unsqueeze_1206 = None
        add_389: "f32[8, 112, 56, 56]" = torch.ops.aten.add.Tensor(mul_452, unsqueeze_1207);  mul_452 = unsqueeze_1207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_146: "f32[8, 112, 56, 56]" = torch.ops.aten.relu.default(add_389);  add_389 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_145 = torch.ops.aten.split.Tensor(relu_146, 14, 1)
        getitem_1164: "f32[8, 14, 56, 56]" = split_145[0];  split_145 = None
        convolution_151: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_1164, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1164 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_390: "f32[14]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_151: "f32[14]" = torch.ops.aten.sqrt.default(add_390);  add_390 = None
        reciprocal_151: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_151);  sqrt_151 = None
        mul_453: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_151, 1);  reciprocal_151 = None
        unsqueeze_1208: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_1209: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, -1);  unsqueeze_1208 = None
        unsqueeze_1210: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_453, -1);  mul_453 = None
        unsqueeze_1211: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, -1);  unsqueeze_1210 = None
        sub_151: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_1209);  convolution_151 = unsqueeze_1209 = None
        mul_454: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_1211);  sub_151 = unsqueeze_1211 = None
        unsqueeze_1212: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_1213: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, -1);  unsqueeze_1212 = None
        mul_455: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_454, unsqueeze_1213);  mul_454 = unsqueeze_1213 = None
        unsqueeze_1214: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_1215: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, -1);  unsqueeze_1214 = None
        add_391: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_455, unsqueeze_1215);  mul_455 = unsqueeze_1215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_147: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_391);  add_391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_146 = torch.ops.aten.split.Tensor(relu_146, 14, 1)
        getitem_1173: "f32[8, 14, 56, 56]" = split_146[1];  split_146 = None
        convolution_152: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_1173, arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1173 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_392: "f32[14]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_152: "f32[14]" = torch.ops.aten.sqrt.default(add_392);  add_392 = None
        reciprocal_152: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_152);  sqrt_152 = None
        mul_456: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_152, 1);  reciprocal_152 = None
        unsqueeze_1216: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_1217: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, -1);  unsqueeze_1216 = None
        unsqueeze_1218: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_456, -1);  mul_456 = None
        unsqueeze_1219: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, -1);  unsqueeze_1218 = None
        sub_152: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_1217);  convolution_152 = unsqueeze_1217 = None
        mul_457: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_1219);  sub_152 = unsqueeze_1219 = None
        unsqueeze_1220: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_1221: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, -1);  unsqueeze_1220 = None
        mul_458: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_457, unsqueeze_1221);  mul_457 = unsqueeze_1221 = None
        unsqueeze_1222: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_1223: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, -1);  unsqueeze_1222 = None
        add_393: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_458, unsqueeze_1223);  mul_458 = unsqueeze_1223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_148: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_393);  add_393 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_147 = torch.ops.aten.split.Tensor(relu_146, 14, 1)
        getitem_1182: "f32[8, 14, 56, 56]" = split_147[2];  split_147 = None
        convolution_153: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_1182, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1182 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_394: "f32[14]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_153: "f32[14]" = torch.ops.aten.sqrt.default(add_394);  add_394 = None
        reciprocal_153: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_153);  sqrt_153 = None
        mul_459: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_153, 1);  reciprocal_153 = None
        unsqueeze_1224: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_1225: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, -1);  unsqueeze_1224 = None
        unsqueeze_1226: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_459, -1);  mul_459 = None
        unsqueeze_1227: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, -1);  unsqueeze_1226 = None
        sub_153: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_153, unsqueeze_1225);  convolution_153 = unsqueeze_1225 = None
        mul_460: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_1227);  sub_153 = unsqueeze_1227 = None
        unsqueeze_1228: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_1229: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, -1);  unsqueeze_1228 = None
        mul_461: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_460, unsqueeze_1229);  mul_460 = unsqueeze_1229 = None
        unsqueeze_1230: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_1231: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, -1);  unsqueeze_1230 = None
        add_395: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_1231);  mul_461 = unsqueeze_1231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_149: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_395);  add_395 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_148 = torch.ops.aten.split.Tensor(relu_146, 14, 1)
        getitem_1191: "f32[8, 14, 56, 56]" = split_148[3];  split_148 = None
        convolution_154: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_1191, arg26_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1191 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_396: "f32[14]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_154: "f32[14]" = torch.ops.aten.sqrt.default(add_396);  add_396 = None
        reciprocal_154: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_154);  sqrt_154 = None
        mul_462: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_154, 1);  reciprocal_154 = None
        unsqueeze_1232: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_1233: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, -1);  unsqueeze_1232 = None
        unsqueeze_1234: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_462, -1);  mul_462 = None
        unsqueeze_1235: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, -1);  unsqueeze_1234 = None
        sub_154: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_1233);  convolution_154 = unsqueeze_1233 = None
        mul_463: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_1235);  sub_154 = unsqueeze_1235 = None
        unsqueeze_1236: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_1237: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, -1);  unsqueeze_1236 = None
        mul_464: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_463, unsqueeze_1237);  mul_463 = unsqueeze_1237 = None
        unsqueeze_1238: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_1239: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, -1);  unsqueeze_1238 = None
        add_397: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_464, unsqueeze_1239);  mul_464 = unsqueeze_1239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_150: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_397);  add_397 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_149 = torch.ops.aten.split.Tensor(relu_146, 14, 1)
        getitem_1200: "f32[8, 14, 56, 56]" = split_149[4];  split_149 = None
        convolution_155: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_1200, arg31_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1200 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_398: "f32[14]" = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_155: "f32[14]" = torch.ops.aten.sqrt.default(add_398);  add_398 = None
        reciprocal_155: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_155);  sqrt_155 = None
        mul_465: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_155, 1);  reciprocal_155 = None
        unsqueeze_1240: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_1241: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, -1);  unsqueeze_1240 = None
        unsqueeze_1242: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_465, -1);  mul_465 = None
        unsqueeze_1243: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, -1);  unsqueeze_1242 = None
        sub_155: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_1241);  convolution_155 = unsqueeze_1241 = None
        mul_466: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_1243);  sub_155 = unsqueeze_1243 = None
        unsqueeze_1244: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_1245: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, -1);  unsqueeze_1244 = None
        mul_467: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_466, unsqueeze_1245);  mul_466 = unsqueeze_1245 = None
        unsqueeze_1246: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_1247: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, -1);  unsqueeze_1246 = None
        add_399: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_467, unsqueeze_1247);  mul_467 = unsqueeze_1247 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_151: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_399);  add_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_150 = torch.ops.aten.split.Tensor(relu_146, 14, 1)
        getitem_1209: "f32[8, 14, 56, 56]" = split_150[5];  split_150 = None
        convolution_156: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_1209, arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1209 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_400: "f32[14]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_156: "f32[14]" = torch.ops.aten.sqrt.default(add_400);  add_400 = None
        reciprocal_156: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_156);  sqrt_156 = None
        mul_468: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_156, 1);  reciprocal_156 = None
        unsqueeze_1248: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_1249: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, -1);  unsqueeze_1248 = None
        unsqueeze_1250: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_468, -1);  mul_468 = None
        unsqueeze_1251: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, -1);  unsqueeze_1250 = None
        sub_156: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_156, unsqueeze_1249);  convolution_156 = unsqueeze_1249 = None
        mul_469: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_1251);  sub_156 = unsqueeze_1251 = None
        unsqueeze_1252: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_1253: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, -1);  unsqueeze_1252 = None
        mul_470: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_1253);  mul_469 = unsqueeze_1253 = None
        unsqueeze_1254: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_1255: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, -1);  unsqueeze_1254 = None
        add_401: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_470, unsqueeze_1255);  mul_470 = unsqueeze_1255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_152: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_401);  add_401 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_151 = torch.ops.aten.split.Tensor(relu_146, 14, 1)
        getitem_1218: "f32[8, 14, 56, 56]" = split_151[6];  split_151 = None
        convolution_157: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_1218, arg41_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1218 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_402: "f32[14]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_157: "f32[14]" = torch.ops.aten.sqrt.default(add_402);  add_402 = None
        reciprocal_157: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_157);  sqrt_157 = None
        mul_471: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_157, 1);  reciprocal_157 = None
        unsqueeze_1256: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_1257: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, -1);  unsqueeze_1256 = None
        unsqueeze_1258: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_471, -1);  mul_471 = None
        unsqueeze_1259: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, -1);  unsqueeze_1258 = None
        sub_157: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_1257);  convolution_157 = unsqueeze_1257 = None
        mul_472: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_1259);  sub_157 = unsqueeze_1259 = None
        unsqueeze_1260: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_1261: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, -1);  unsqueeze_1260 = None
        mul_473: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_472, unsqueeze_1261);  mul_472 = unsqueeze_1261 = None
        unsqueeze_1262: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_1263: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, -1);  unsqueeze_1262 = None
        add_403: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_473, unsqueeze_1263);  mul_473 = unsqueeze_1263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_153: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_403);  add_403 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:99 in forward, code: spo.append(self.pool(spx[-1]))
        split_152 = torch.ops.aten.split.Tensor(relu_146, 14, 1);  relu_146 = None
        getitem_1227: "f32[8, 14, 56, 56]" = split_152[7];  split_152 = None
        avg_pool2d_4: "f32[8, 14, 56, 56]" = torch.ops.aten.avg_pool2d.default(getitem_1227, [3, 3], [1, 1], [1, 1]);  getitem_1227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_16: "f32[8, 112, 56, 56]" = torch.ops.aten.cat.default([relu_147, relu_148, relu_149, relu_150, relu_151, relu_152, relu_153, avg_pool2d_4], 1);  relu_147 = relu_148 = relu_149 = relu_150 = relu_151 = relu_152 = relu_153 = avg_pool2d_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_158: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_16, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_16 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_404: "f32[256]" = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_158: "f32[256]" = torch.ops.aten.sqrt.default(add_404);  add_404 = None
        reciprocal_158: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_158);  sqrt_158 = None
        mul_474: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_158, 1);  reciprocal_158 = None
        unsqueeze_1264: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_1265: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, -1);  unsqueeze_1264 = None
        unsqueeze_1266: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_474, -1);  mul_474 = None
        unsqueeze_1267: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, -1);  unsqueeze_1266 = None
        sub_158: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_158, unsqueeze_1265);  convolution_158 = unsqueeze_1265 = None
        mul_475: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_1267);  sub_158 = unsqueeze_1267 = None
        unsqueeze_1268: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_1269: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, -1);  unsqueeze_1268 = None
        mul_476: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_475, unsqueeze_1269);  mul_475 = unsqueeze_1269 = None
        unsqueeze_1270: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_1271: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, -1);  unsqueeze_1270 = None
        add_405: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_476, unsqueeze_1271);  mul_476 = unsqueeze_1271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:111 in forward, code: shortcut = self.downsample(x)
        convolution_159: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_1154, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_1154 = arg51_1 = None
        add_406: "f32[256]" = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_159: "f32[256]" = torch.ops.aten.sqrt.default(add_406);  add_406 = None
        reciprocal_159: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_159);  sqrt_159 = None
        mul_477: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_159, 1);  reciprocal_159 = None
        unsqueeze_1272: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_1273: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, -1);  unsqueeze_1272 = None
        unsqueeze_1274: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_477, -1);  mul_477 = None
        unsqueeze_1275: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, -1);  unsqueeze_1274 = None
        sub_159: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_159, unsqueeze_1273);  convolution_159 = unsqueeze_1273 = None
        mul_478: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_1275);  sub_159 = unsqueeze_1275 = None
        unsqueeze_1276: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_1277: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, -1);  unsqueeze_1276 = None
        mul_479: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_478, unsqueeze_1277);  mul_478 = unsqueeze_1277 = None
        unsqueeze_1278: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_1279: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, -1);  unsqueeze_1278 = None
        add_407: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_479, unsqueeze_1279);  mul_479 = unsqueeze_1279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_408: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_405, add_407);  add_405 = add_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_154: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_408);  add_408 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_160: "f32[8, 112, 56, 56]" = torch.ops.aten.convolution.default(relu_154, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_409: "f32[112]" = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_160: "f32[112]" = torch.ops.aten.sqrt.default(add_409);  add_409 = None
        reciprocal_160: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_160);  sqrt_160 = None
        mul_480: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_160, 1);  reciprocal_160 = None
        unsqueeze_1280: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_1281: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, -1);  unsqueeze_1280 = None
        unsqueeze_1282: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_480, -1);  mul_480 = None
        unsqueeze_1283: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, -1);  unsqueeze_1282 = None
        sub_160: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_1281);  convolution_160 = unsqueeze_1281 = None
        mul_481: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_1283);  sub_160 = unsqueeze_1283 = None
        unsqueeze_1284: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_1285: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, -1);  unsqueeze_1284 = None
        mul_482: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(mul_481, unsqueeze_1285);  mul_481 = unsqueeze_1285 = None
        unsqueeze_1286: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_1287: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, -1);  unsqueeze_1286 = None
        add_410: "f32[8, 112, 56, 56]" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_1287);  mul_482 = unsqueeze_1287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_155: "f32[8, 112, 56, 56]" = torch.ops.aten.relu.default(add_410);  add_410 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_154 = torch.ops.aten.split.Tensor(relu_155, 14, 1)
        getitem_1236: "f32[8, 14, 56, 56]" = split_154[0];  split_154 = None
        convolution_161: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_1236, arg61_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1236 = arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_411: "f32[14]" = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_161: "f32[14]" = torch.ops.aten.sqrt.default(add_411);  add_411 = None
        reciprocal_161: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_161);  sqrt_161 = None
        mul_483: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_161, 1);  reciprocal_161 = None
        unsqueeze_1288: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_1289: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, -1);  unsqueeze_1288 = None
        unsqueeze_1290: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_483, -1);  mul_483 = None
        unsqueeze_1291: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, -1);  unsqueeze_1290 = None
        sub_161: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_1289);  convolution_161 = unsqueeze_1289 = None
        mul_484: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_1291);  sub_161 = unsqueeze_1291 = None
        unsqueeze_1292: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_1293: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, -1);  unsqueeze_1292 = None
        mul_485: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_484, unsqueeze_1293);  mul_484 = unsqueeze_1293 = None
        unsqueeze_1294: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_1295: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, -1);  unsqueeze_1294 = None
        add_412: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_485, unsqueeze_1295);  mul_485 = unsqueeze_1295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_156: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_412);  add_412 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_155 = torch.ops.aten.split.Tensor(relu_155, 14, 1)
        getitem_1245: "f32[8, 14, 56, 56]" = split_155[1];  split_155 = None
        add_413: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_156, getitem_1245);  getitem_1245 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_162: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_413, arg66_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_413 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_414: "f32[14]" = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_162: "f32[14]" = torch.ops.aten.sqrt.default(add_414);  add_414 = None
        reciprocal_162: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_162);  sqrt_162 = None
        mul_486: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_162, 1);  reciprocal_162 = None
        unsqueeze_1296: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_1297: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, -1);  unsqueeze_1296 = None
        unsqueeze_1298: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_486, -1);  mul_486 = None
        unsqueeze_1299: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, -1);  unsqueeze_1298 = None
        sub_162: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_162, unsqueeze_1297);  convolution_162 = unsqueeze_1297 = None
        mul_487: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_1299);  sub_162 = unsqueeze_1299 = None
        unsqueeze_1300: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_1301: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, -1);  unsqueeze_1300 = None
        mul_488: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_487, unsqueeze_1301);  mul_487 = unsqueeze_1301 = None
        unsqueeze_1302: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_1303: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, -1);  unsqueeze_1302 = None
        add_415: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_488, unsqueeze_1303);  mul_488 = unsqueeze_1303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_157: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_415);  add_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_156 = torch.ops.aten.split.Tensor(relu_155, 14, 1)
        getitem_1254: "f32[8, 14, 56, 56]" = split_156[2];  split_156 = None
        add_416: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_157, getitem_1254);  getitem_1254 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_163: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_416, arg71_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_416 = arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_417: "f32[14]" = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_163: "f32[14]" = torch.ops.aten.sqrt.default(add_417);  add_417 = None
        reciprocal_163: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_163);  sqrt_163 = None
        mul_489: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_163, 1);  reciprocal_163 = None
        unsqueeze_1304: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_1305: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, -1);  unsqueeze_1304 = None
        unsqueeze_1306: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_489, -1);  mul_489 = None
        unsqueeze_1307: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, -1);  unsqueeze_1306 = None
        sub_163: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_163, unsqueeze_1305);  convolution_163 = unsqueeze_1305 = None
        mul_490: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_1307);  sub_163 = unsqueeze_1307 = None
        unsqueeze_1308: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_1309: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, -1);  unsqueeze_1308 = None
        mul_491: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_1309);  mul_490 = unsqueeze_1309 = None
        unsqueeze_1310: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_1311: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, -1);  unsqueeze_1310 = None
        add_418: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_491, unsqueeze_1311);  mul_491 = unsqueeze_1311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_158: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_418);  add_418 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_157 = torch.ops.aten.split.Tensor(relu_155, 14, 1)
        getitem_1263: "f32[8, 14, 56, 56]" = split_157[3];  split_157 = None
        add_419: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_158, getitem_1263);  getitem_1263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_164: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_419, arg76_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_419 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_420: "f32[14]" = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_164: "f32[14]" = torch.ops.aten.sqrt.default(add_420);  add_420 = None
        reciprocal_164: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_164);  sqrt_164 = None
        mul_492: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_164, 1);  reciprocal_164 = None
        unsqueeze_1312: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_1313: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, -1);  unsqueeze_1312 = None
        unsqueeze_1314: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_492, -1);  mul_492 = None
        unsqueeze_1315: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, -1);  unsqueeze_1314 = None
        sub_164: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_164, unsqueeze_1313);  convolution_164 = unsqueeze_1313 = None
        mul_493: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_1315);  sub_164 = unsqueeze_1315 = None
        unsqueeze_1316: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_1317: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, -1);  unsqueeze_1316 = None
        mul_494: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_493, unsqueeze_1317);  mul_493 = unsqueeze_1317 = None
        unsqueeze_1318: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_1319: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, -1);  unsqueeze_1318 = None
        add_421: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_494, unsqueeze_1319);  mul_494 = unsqueeze_1319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_159: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_421);  add_421 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_158 = torch.ops.aten.split.Tensor(relu_155, 14, 1)
        getitem_1272: "f32[8, 14, 56, 56]" = split_158[4];  split_158 = None
        add_422: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_159, getitem_1272);  getitem_1272 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_165: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_422, arg81_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_422 = arg81_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_423: "f32[14]" = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_165: "f32[14]" = torch.ops.aten.sqrt.default(add_423);  add_423 = None
        reciprocal_165: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_165);  sqrt_165 = None
        mul_495: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_165, 1);  reciprocal_165 = None
        unsqueeze_1320: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_1321: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, -1);  unsqueeze_1320 = None
        unsqueeze_1322: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_495, -1);  mul_495 = None
        unsqueeze_1323: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, -1);  unsqueeze_1322 = None
        sub_165: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_165, unsqueeze_1321);  convolution_165 = unsqueeze_1321 = None
        mul_496: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_1323);  sub_165 = unsqueeze_1323 = None
        unsqueeze_1324: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_1325: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, -1);  unsqueeze_1324 = None
        mul_497: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_496, unsqueeze_1325);  mul_496 = unsqueeze_1325 = None
        unsqueeze_1326: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_1327: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, -1);  unsqueeze_1326 = None
        add_424: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_497, unsqueeze_1327);  mul_497 = unsqueeze_1327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_160: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_424);  add_424 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_159 = torch.ops.aten.split.Tensor(relu_155, 14, 1)
        getitem_1281: "f32[8, 14, 56, 56]" = split_159[5];  split_159 = None
        add_425: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_160, getitem_1281);  getitem_1281 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_166: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_425, arg86_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_425 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_426: "f32[14]" = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_166: "f32[14]" = torch.ops.aten.sqrt.default(add_426);  add_426 = None
        reciprocal_166: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_166);  sqrt_166 = None
        mul_498: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_166, 1);  reciprocal_166 = None
        unsqueeze_1328: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_1329: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, -1);  unsqueeze_1328 = None
        unsqueeze_1330: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_498, -1);  mul_498 = None
        unsqueeze_1331: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, -1);  unsqueeze_1330 = None
        sub_166: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_1329);  convolution_166 = unsqueeze_1329 = None
        mul_499: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_1331);  sub_166 = unsqueeze_1331 = None
        unsqueeze_1332: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_1333: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, -1);  unsqueeze_1332 = None
        mul_500: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_1333);  mul_499 = unsqueeze_1333 = None
        unsqueeze_1334: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_1335: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, -1);  unsqueeze_1334 = None
        add_427: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_500, unsqueeze_1335);  mul_500 = unsqueeze_1335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_161: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_427);  add_427 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_160 = torch.ops.aten.split.Tensor(relu_155, 14, 1)
        getitem_1290: "f32[8, 14, 56, 56]" = split_160[6];  split_160 = None
        add_428: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_161, getitem_1290);  getitem_1290 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_167: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_428, arg91_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_428 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_429: "f32[14]" = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_167: "f32[14]" = torch.ops.aten.sqrt.default(add_429);  add_429 = None
        reciprocal_167: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_167);  sqrt_167 = None
        mul_501: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_167, 1);  reciprocal_167 = None
        unsqueeze_1336: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_1337: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, -1);  unsqueeze_1336 = None
        unsqueeze_1338: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_501, -1);  mul_501 = None
        unsqueeze_1339: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, -1);  unsqueeze_1338 = None
        sub_167: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_1337);  convolution_167 = unsqueeze_1337 = None
        mul_502: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_1339);  sub_167 = unsqueeze_1339 = None
        unsqueeze_1340: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_1341: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, -1);  unsqueeze_1340 = None
        mul_503: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_502, unsqueeze_1341);  mul_502 = unsqueeze_1341 = None
        unsqueeze_1342: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_1343: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, -1);  unsqueeze_1342 = None
        add_430: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_1343);  mul_503 = unsqueeze_1343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_162: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_430);  add_430 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_161 = torch.ops.aten.split.Tensor(relu_155, 14, 1);  relu_155 = None
        getitem_1299: "f32[8, 14, 56, 56]" = split_161[7];  split_161 = None
        cat_17: "f32[8, 112, 56, 56]" = torch.ops.aten.cat.default([relu_156, relu_157, relu_158, relu_159, relu_160, relu_161, relu_162, getitem_1299], 1);  relu_156 = relu_157 = relu_158 = relu_159 = relu_160 = relu_161 = relu_162 = getitem_1299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_168: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_17, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_17 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_431: "f32[256]" = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_168: "f32[256]" = torch.ops.aten.sqrt.default(add_431);  add_431 = None
        reciprocal_168: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_168);  sqrt_168 = None
        mul_504: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_168, 1);  reciprocal_168 = None
        unsqueeze_1344: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_1345: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, -1);  unsqueeze_1344 = None
        unsqueeze_1346: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_504, -1);  mul_504 = None
        unsqueeze_1347: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, -1);  unsqueeze_1346 = None
        sub_168: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_168, unsqueeze_1345);  convolution_168 = unsqueeze_1345 = None
        mul_505: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_1347);  sub_168 = unsqueeze_1347 = None
        unsqueeze_1348: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_1349: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, -1);  unsqueeze_1348 = None
        mul_506: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_505, unsqueeze_1349);  mul_505 = unsqueeze_1349 = None
        unsqueeze_1350: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_1351: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, -1);  unsqueeze_1350 = None
        add_432: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_506, unsqueeze_1351);  mul_506 = unsqueeze_1351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_433: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_432, relu_154);  add_432 = relu_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_163: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_433);  add_433 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_169: "f32[8, 112, 56, 56]" = torch.ops.aten.convolution.default(relu_163, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_434: "f32[112]" = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_169: "f32[112]" = torch.ops.aten.sqrt.default(add_434);  add_434 = None
        reciprocal_169: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_169);  sqrt_169 = None
        mul_507: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_169, 1);  reciprocal_169 = None
        unsqueeze_1352: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_1353: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, -1);  unsqueeze_1352 = None
        unsqueeze_1354: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_507, -1);  mul_507 = None
        unsqueeze_1355: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, -1);  unsqueeze_1354 = None
        sub_169: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_169, unsqueeze_1353);  convolution_169 = unsqueeze_1353 = None
        mul_508: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_1355);  sub_169 = unsqueeze_1355 = None
        unsqueeze_1356: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_1357: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, -1);  unsqueeze_1356 = None
        mul_509: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(mul_508, unsqueeze_1357);  mul_508 = unsqueeze_1357 = None
        unsqueeze_1358: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_1359: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, -1);  unsqueeze_1358 = None
        add_435: "f32[8, 112, 56, 56]" = torch.ops.aten.add.Tensor(mul_509, unsqueeze_1359);  mul_509 = unsqueeze_1359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_164: "f32[8, 112, 56, 56]" = torch.ops.aten.relu.default(add_435);  add_435 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_163 = torch.ops.aten.split.Tensor(relu_164, 14, 1)
        getitem_1308: "f32[8, 14, 56, 56]" = split_163[0];  split_163 = None
        convolution_170: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(getitem_1308, arg106_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1308 = arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_436: "f32[14]" = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_170: "f32[14]" = torch.ops.aten.sqrt.default(add_436);  add_436 = None
        reciprocal_170: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_170);  sqrt_170 = None
        mul_510: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_170, 1);  reciprocal_170 = None
        unsqueeze_1360: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_1361: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, -1);  unsqueeze_1360 = None
        unsqueeze_1362: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_510, -1);  mul_510 = None
        unsqueeze_1363: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, -1);  unsqueeze_1362 = None
        sub_170: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_170, unsqueeze_1361);  convolution_170 = unsqueeze_1361 = None
        mul_511: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_1363);  sub_170 = unsqueeze_1363 = None
        unsqueeze_1364: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_1365: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, -1);  unsqueeze_1364 = None
        mul_512: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_511, unsqueeze_1365);  mul_511 = unsqueeze_1365 = None
        unsqueeze_1366: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_1367: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, -1);  unsqueeze_1366 = None
        add_437: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_512, unsqueeze_1367);  mul_512 = unsqueeze_1367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_165: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_437);  add_437 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_164 = torch.ops.aten.split.Tensor(relu_164, 14, 1)
        getitem_1317: "f32[8, 14, 56, 56]" = split_164[1];  split_164 = None
        add_438: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_165, getitem_1317);  getitem_1317 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_171: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_438, arg111_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_438 = arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_439: "f32[14]" = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_171: "f32[14]" = torch.ops.aten.sqrt.default(add_439);  add_439 = None
        reciprocal_171: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_171);  sqrt_171 = None
        mul_513: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_171, 1);  reciprocal_171 = None
        unsqueeze_1368: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_1369: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, -1);  unsqueeze_1368 = None
        unsqueeze_1370: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_513, -1);  mul_513 = None
        unsqueeze_1371: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, -1);  unsqueeze_1370 = None
        sub_171: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_171, unsqueeze_1369);  convolution_171 = unsqueeze_1369 = None
        mul_514: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_1371);  sub_171 = unsqueeze_1371 = None
        unsqueeze_1372: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_1373: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, -1);  unsqueeze_1372 = None
        mul_515: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_514, unsqueeze_1373);  mul_514 = unsqueeze_1373 = None
        unsqueeze_1374: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_1375: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, -1);  unsqueeze_1374 = None
        add_440: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_515, unsqueeze_1375);  mul_515 = unsqueeze_1375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_166: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_440);  add_440 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_165 = torch.ops.aten.split.Tensor(relu_164, 14, 1)
        getitem_1326: "f32[8, 14, 56, 56]" = split_165[2];  split_165 = None
        add_441: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_166, getitem_1326);  getitem_1326 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_172: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_441, arg116_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_441 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_442: "f32[14]" = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
        sqrt_172: "f32[14]" = torch.ops.aten.sqrt.default(add_442);  add_442 = None
        reciprocal_172: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_172);  sqrt_172 = None
        mul_516: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_172, 1);  reciprocal_172 = None
        unsqueeze_1376: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_1377: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, -1);  unsqueeze_1376 = None
        unsqueeze_1378: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_516, -1);  mul_516 = None
        unsqueeze_1379: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, -1);  unsqueeze_1378 = None
        sub_172: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_172, unsqueeze_1377);  convolution_172 = unsqueeze_1377 = None
        mul_517: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_1379);  sub_172 = unsqueeze_1379 = None
        unsqueeze_1380: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_1381: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, -1);  unsqueeze_1380 = None
        mul_518: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_517, unsqueeze_1381);  mul_517 = unsqueeze_1381 = None
        unsqueeze_1382: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_1383: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, -1);  unsqueeze_1382 = None
        add_443: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_518, unsqueeze_1383);  mul_518 = unsqueeze_1383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_167: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_443);  add_443 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_166 = torch.ops.aten.split.Tensor(relu_164, 14, 1)
        getitem_1335: "f32[8, 14, 56, 56]" = split_166[3];  split_166 = None
        add_444: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_167, getitem_1335);  getitem_1335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_173: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_444, arg121_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_444 = arg121_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_445: "f32[14]" = torch.ops.aten.add.Tensor(arg123_1, 1e-05);  arg123_1 = None
        sqrt_173: "f32[14]" = torch.ops.aten.sqrt.default(add_445);  add_445 = None
        reciprocal_173: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_173);  sqrt_173 = None
        mul_519: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_173, 1);  reciprocal_173 = None
        unsqueeze_1384: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_1385: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, -1);  unsqueeze_1384 = None
        unsqueeze_1386: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_519, -1);  mul_519 = None
        unsqueeze_1387: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, -1);  unsqueeze_1386 = None
        sub_173: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_173, unsqueeze_1385);  convolution_173 = unsqueeze_1385 = None
        mul_520: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_1387);  sub_173 = unsqueeze_1387 = None
        unsqueeze_1388: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_1389: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, -1);  unsqueeze_1388 = None
        mul_521: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_520, unsqueeze_1389);  mul_520 = unsqueeze_1389 = None
        unsqueeze_1390: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_1391: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, -1);  unsqueeze_1390 = None
        add_446: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_521, unsqueeze_1391);  mul_521 = unsqueeze_1391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_168: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_446);  add_446 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_167 = torch.ops.aten.split.Tensor(relu_164, 14, 1)
        getitem_1344: "f32[8, 14, 56, 56]" = split_167[4];  split_167 = None
        add_447: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_168, getitem_1344);  getitem_1344 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_174: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_447, arg126_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_447 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_448: "f32[14]" = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_174: "f32[14]" = torch.ops.aten.sqrt.default(add_448);  add_448 = None
        reciprocal_174: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_174);  sqrt_174 = None
        mul_522: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_174, 1);  reciprocal_174 = None
        unsqueeze_1392: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_1393: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1392, -1);  unsqueeze_1392 = None
        unsqueeze_1394: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_522, -1);  mul_522 = None
        unsqueeze_1395: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, -1);  unsqueeze_1394 = None
        sub_174: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_174, unsqueeze_1393);  convolution_174 = unsqueeze_1393 = None
        mul_523: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_1395);  sub_174 = unsqueeze_1395 = None
        unsqueeze_1396: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_1397: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1396, -1);  unsqueeze_1396 = None
        mul_524: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_523, unsqueeze_1397);  mul_523 = unsqueeze_1397 = None
        unsqueeze_1398: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_1399: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, -1);  unsqueeze_1398 = None
        add_449: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_524, unsqueeze_1399);  mul_524 = unsqueeze_1399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_169: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_449);  add_449 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_168 = torch.ops.aten.split.Tensor(relu_164, 14, 1)
        getitem_1353: "f32[8, 14, 56, 56]" = split_168[5];  split_168 = None
        add_450: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_169, getitem_1353);  getitem_1353 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_175: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_450, arg131_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_450 = arg131_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_451: "f32[14]" = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_175: "f32[14]" = torch.ops.aten.sqrt.default(add_451);  add_451 = None
        reciprocal_175: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_175);  sqrt_175 = None
        mul_525: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_175, 1);  reciprocal_175 = None
        unsqueeze_1400: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_1401: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, -1);  unsqueeze_1400 = None
        unsqueeze_1402: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_525, -1);  mul_525 = None
        unsqueeze_1403: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1402, -1);  unsqueeze_1402 = None
        sub_175: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_175, unsqueeze_1401);  convolution_175 = unsqueeze_1401 = None
        mul_526: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_1403);  sub_175 = unsqueeze_1403 = None
        unsqueeze_1404: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_1405: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1404, -1);  unsqueeze_1404 = None
        mul_527: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_526, unsqueeze_1405);  mul_526 = unsqueeze_1405 = None
        unsqueeze_1406: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_1407: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, -1);  unsqueeze_1406 = None
        add_452: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_527, unsqueeze_1407);  mul_527 = unsqueeze_1407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_170: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_452);  add_452 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_169 = torch.ops.aten.split.Tensor(relu_164, 14, 1)
        getitem_1362: "f32[8, 14, 56, 56]" = split_169[6];  split_169 = None
        add_453: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(relu_170, getitem_1362);  getitem_1362 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_176: "f32[8, 14, 56, 56]" = torch.ops.aten.convolution.default(add_453, arg136_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_453 = arg136_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_454: "f32[14]" = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
        sqrt_176: "f32[14]" = torch.ops.aten.sqrt.default(add_454);  add_454 = None
        reciprocal_176: "f32[14]" = torch.ops.aten.reciprocal.default(sqrt_176);  sqrt_176 = None
        mul_528: "f32[14]" = torch.ops.aten.mul.Tensor(reciprocal_176, 1);  reciprocal_176 = None
        unsqueeze_1408: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_1409: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1408, -1);  unsqueeze_1408 = None
        unsqueeze_1410: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(mul_528, -1);  mul_528 = None
        unsqueeze_1411: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, -1);  unsqueeze_1410 = None
        sub_176: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_176, unsqueeze_1409);  convolution_176 = unsqueeze_1409 = None
        mul_529: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_1411);  sub_176 = unsqueeze_1411 = None
        unsqueeze_1412: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_1413: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, -1);  unsqueeze_1412 = None
        mul_530: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(mul_529, unsqueeze_1413);  mul_529 = unsqueeze_1413 = None
        unsqueeze_1414: "f32[14, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_1415: "f32[14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1414, -1);  unsqueeze_1414 = None
        add_455: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(mul_530, unsqueeze_1415);  mul_530 = unsqueeze_1415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_171: "f32[8, 14, 56, 56]" = torch.ops.aten.relu.default(add_455);  add_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_170 = torch.ops.aten.split.Tensor(relu_164, 14, 1);  relu_164 = None
        getitem_1371: "f32[8, 14, 56, 56]" = split_170[7];  split_170 = None
        cat_18: "f32[8, 112, 56, 56]" = torch.ops.aten.cat.default([relu_165, relu_166, relu_167, relu_168, relu_169, relu_170, relu_171, getitem_1371], 1);  relu_165 = relu_166 = relu_167 = relu_168 = relu_169 = relu_170 = relu_171 = getitem_1371 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_177: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_18, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_18 = arg141_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_456: "f32[256]" = torch.ops.aten.add.Tensor(arg143_1, 1e-05);  arg143_1 = None
        sqrt_177: "f32[256]" = torch.ops.aten.sqrt.default(add_456);  add_456 = None
        reciprocal_177: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_177);  sqrt_177 = None
        mul_531: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_177, 1);  reciprocal_177 = None
        unsqueeze_1416: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_1417: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1416, -1);  unsqueeze_1416 = None
        unsqueeze_1418: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_531, -1);  mul_531 = None
        unsqueeze_1419: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, -1);  unsqueeze_1418 = None
        sub_177: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_177, unsqueeze_1417);  convolution_177 = unsqueeze_1417 = None
        mul_532: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_1419);  sub_177 = unsqueeze_1419 = None
        unsqueeze_1420: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_1421: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1420, -1);  unsqueeze_1420 = None
        mul_533: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_1421);  mul_532 = unsqueeze_1421 = None
        unsqueeze_1422: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_1423: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, -1);  unsqueeze_1422 = None
        add_457: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_533, unsqueeze_1423);  mul_533 = unsqueeze_1423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_458: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_457, relu_163);  add_457 = relu_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_172: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_458);  add_458 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_178: "f32[8, 224, 56, 56]" = torch.ops.aten.convolution.default(relu_172, arg146_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_459: "f32[224]" = torch.ops.aten.add.Tensor(arg148_1, 1e-05);  arg148_1 = None
        sqrt_178: "f32[224]" = torch.ops.aten.sqrt.default(add_459);  add_459 = None
        reciprocal_178: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_178);  sqrt_178 = None
        mul_534: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_178, 1);  reciprocal_178 = None
        unsqueeze_1424: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_1425: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, -1);  unsqueeze_1424 = None
        unsqueeze_1426: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_534, -1);  mul_534 = None
        unsqueeze_1427: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1426, -1);  unsqueeze_1426 = None
        sub_178: "f32[8, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_178, unsqueeze_1425);  convolution_178 = unsqueeze_1425 = None
        mul_535: "f32[8, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_1427);  sub_178 = unsqueeze_1427 = None
        unsqueeze_1428: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_1429: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1428, -1);  unsqueeze_1428 = None
        mul_536: "f32[8, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_535, unsqueeze_1429);  mul_535 = unsqueeze_1429 = None
        unsqueeze_1430: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_1431: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1430, -1);  unsqueeze_1430 = None
        add_460: "f32[8, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_536, unsqueeze_1431);  mul_536 = unsqueeze_1431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_173: "f32[8, 224, 56, 56]" = torch.ops.aten.relu.default(add_460);  add_460 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_172 = torch.ops.aten.split.Tensor(relu_173, 28, 1)
        getitem_1380: "f32[8, 28, 56, 56]" = split_172[0];  split_172 = None
        convolution_179: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_1380, arg151_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1380 = arg151_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_461: "f32[28]" = torch.ops.aten.add.Tensor(arg153_1, 1e-05);  arg153_1 = None
        sqrt_179: "f32[28]" = torch.ops.aten.sqrt.default(add_461);  add_461 = None
        reciprocal_179: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_179);  sqrt_179 = None
        mul_537: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_179, 1);  reciprocal_179 = None
        unsqueeze_1432: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_1433: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, -1);  unsqueeze_1432 = None
        unsqueeze_1434: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_537, -1);  mul_537 = None
        unsqueeze_1435: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, -1);  unsqueeze_1434 = None
        sub_179: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_179, unsqueeze_1433);  convolution_179 = unsqueeze_1433 = None
        mul_538: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_1435);  sub_179 = unsqueeze_1435 = None
        unsqueeze_1436: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_1437: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1436, -1);  unsqueeze_1436 = None
        mul_539: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_538, unsqueeze_1437);  mul_538 = unsqueeze_1437 = None
        unsqueeze_1438: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_1439: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, -1);  unsqueeze_1438 = None
        add_462: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_539, unsqueeze_1439);  mul_539 = unsqueeze_1439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_174: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_462);  add_462 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_173 = torch.ops.aten.split.Tensor(relu_173, 28, 1)
        getitem_1389: "f32[8, 28, 56, 56]" = split_173[1];  split_173 = None
        convolution_180: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_1389, arg156_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1389 = arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_463: "f32[28]" = torch.ops.aten.add.Tensor(arg158_1, 1e-05);  arg158_1 = None
        sqrt_180: "f32[28]" = torch.ops.aten.sqrt.default(add_463);  add_463 = None
        reciprocal_180: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_180);  sqrt_180 = None
        mul_540: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_180, 1);  reciprocal_180 = None
        unsqueeze_1440: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_1441: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1440, -1);  unsqueeze_1440 = None
        unsqueeze_1442: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_540, -1);  mul_540 = None
        unsqueeze_1443: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1442, -1);  unsqueeze_1442 = None
        sub_180: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_180, unsqueeze_1441);  convolution_180 = unsqueeze_1441 = None
        mul_541: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_1443);  sub_180 = unsqueeze_1443 = None
        unsqueeze_1444: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_1445: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, -1);  unsqueeze_1444 = None
        mul_542: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_541, unsqueeze_1445);  mul_541 = unsqueeze_1445 = None
        unsqueeze_1446: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_1447: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, -1);  unsqueeze_1446 = None
        add_464: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_542, unsqueeze_1447);  mul_542 = unsqueeze_1447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_175: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_464);  add_464 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_174 = torch.ops.aten.split.Tensor(relu_173, 28, 1)
        getitem_1398: "f32[8, 28, 56, 56]" = split_174[2];  split_174 = None
        convolution_181: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_1398, arg161_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1398 = arg161_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_465: "f32[28]" = torch.ops.aten.add.Tensor(arg163_1, 1e-05);  arg163_1 = None
        sqrt_181: "f32[28]" = torch.ops.aten.sqrt.default(add_465);  add_465 = None
        reciprocal_181: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_181);  sqrt_181 = None
        mul_543: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_181, 1);  reciprocal_181 = None
        unsqueeze_1448: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_1449: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, -1);  unsqueeze_1448 = None
        unsqueeze_1450: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_543, -1);  mul_543 = None
        unsqueeze_1451: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, -1);  unsqueeze_1450 = None
        sub_181: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_181, unsqueeze_1449);  convolution_181 = unsqueeze_1449 = None
        mul_544: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_1451);  sub_181 = unsqueeze_1451 = None
        unsqueeze_1452: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_1453: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1452, -1);  unsqueeze_1452 = None
        mul_545: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_544, unsqueeze_1453);  mul_544 = unsqueeze_1453 = None
        unsqueeze_1454: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_1455: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, -1);  unsqueeze_1454 = None
        add_466: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_1455);  mul_545 = unsqueeze_1455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_176: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_466);  add_466 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_175 = torch.ops.aten.split.Tensor(relu_173, 28, 1)
        getitem_1407: "f32[8, 28, 56, 56]" = split_175[3];  split_175 = None
        convolution_182: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_1407, arg166_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1407 = arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_467: "f32[28]" = torch.ops.aten.add.Tensor(arg168_1, 1e-05);  arg168_1 = None
        sqrt_182: "f32[28]" = torch.ops.aten.sqrt.default(add_467);  add_467 = None
        reciprocal_182: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_182);  sqrt_182 = None
        mul_546: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_182, 1);  reciprocal_182 = None
        unsqueeze_1456: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_1457: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, -1);  unsqueeze_1456 = None
        unsqueeze_1458: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_546, -1);  mul_546 = None
        unsqueeze_1459: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, -1);  unsqueeze_1458 = None
        sub_182: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_182, unsqueeze_1457);  convolution_182 = unsqueeze_1457 = None
        mul_547: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_1459);  sub_182 = unsqueeze_1459 = None
        unsqueeze_1460: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_1461: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, -1);  unsqueeze_1460 = None
        mul_548: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_547, unsqueeze_1461);  mul_547 = unsqueeze_1461 = None
        unsqueeze_1462: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_1463: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, -1);  unsqueeze_1462 = None
        add_468: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_548, unsqueeze_1463);  mul_548 = unsqueeze_1463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_177: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_468);  add_468 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_176 = torch.ops.aten.split.Tensor(relu_173, 28, 1)
        getitem_1416: "f32[8, 28, 56, 56]" = split_176[4];  split_176 = None
        convolution_183: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_1416, arg171_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1416 = arg171_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_469: "f32[28]" = torch.ops.aten.add.Tensor(arg173_1, 1e-05);  arg173_1 = None
        sqrt_183: "f32[28]" = torch.ops.aten.sqrt.default(add_469);  add_469 = None
        reciprocal_183: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_183);  sqrt_183 = None
        mul_549: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_183, 1);  reciprocal_183 = None
        unsqueeze_1464: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_1465: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1464, -1);  unsqueeze_1464 = None
        unsqueeze_1466: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_549, -1);  mul_549 = None
        unsqueeze_1467: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, -1);  unsqueeze_1466 = None
        sub_183: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_183, unsqueeze_1465);  convolution_183 = unsqueeze_1465 = None
        mul_550: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_1467);  sub_183 = unsqueeze_1467 = None
        unsqueeze_1468: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_1469: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, -1);  unsqueeze_1468 = None
        mul_551: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_550, unsqueeze_1469);  mul_550 = unsqueeze_1469 = None
        unsqueeze_1470: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_1471: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, -1);  unsqueeze_1470 = None
        add_470: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_551, unsqueeze_1471);  mul_551 = unsqueeze_1471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_178: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_470);  add_470 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_177 = torch.ops.aten.split.Tensor(relu_173, 28, 1)
        getitem_1425: "f32[8, 28, 56, 56]" = split_177[5];  split_177 = None
        convolution_184: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_1425, arg176_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1425 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_471: "f32[28]" = torch.ops.aten.add.Tensor(arg178_1, 1e-05);  arg178_1 = None
        sqrt_184: "f32[28]" = torch.ops.aten.sqrt.default(add_471);  add_471 = None
        reciprocal_184: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_184);  sqrt_184 = None
        mul_552: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_184, 1);  reciprocal_184 = None
        unsqueeze_1472: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_1473: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, -1);  unsqueeze_1472 = None
        unsqueeze_1474: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_552, -1);  mul_552 = None
        unsqueeze_1475: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, -1);  unsqueeze_1474 = None
        sub_184: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_184, unsqueeze_1473);  convolution_184 = unsqueeze_1473 = None
        mul_553: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_1475);  sub_184 = unsqueeze_1475 = None
        unsqueeze_1476: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_1477: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1476, -1);  unsqueeze_1476 = None
        mul_554: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_1477);  mul_553 = unsqueeze_1477 = None
        unsqueeze_1478: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_1479: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, -1);  unsqueeze_1478 = None
        add_472: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_554, unsqueeze_1479);  mul_554 = unsqueeze_1479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_179: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_472);  add_472 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_178 = torch.ops.aten.split.Tensor(relu_173, 28, 1)
        getitem_1434: "f32[8, 28, 56, 56]" = split_178[6];  split_178 = None
        convolution_185: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_1434, arg181_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1434 = arg181_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_473: "f32[28]" = torch.ops.aten.add.Tensor(arg183_1, 1e-05);  arg183_1 = None
        sqrt_185: "f32[28]" = torch.ops.aten.sqrt.default(add_473);  add_473 = None
        reciprocal_185: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_185);  sqrt_185 = None
        mul_555: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_185, 1);  reciprocal_185 = None
        unsqueeze_1480: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_1481: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1480, -1);  unsqueeze_1480 = None
        unsqueeze_1482: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_555, -1);  mul_555 = None
        unsqueeze_1483: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, -1);  unsqueeze_1482 = None
        sub_185: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_185, unsqueeze_1481);  convolution_185 = unsqueeze_1481 = None
        mul_556: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_1483);  sub_185 = unsqueeze_1483 = None
        unsqueeze_1484: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_1485: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, -1);  unsqueeze_1484 = None
        mul_557: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_556, unsqueeze_1485);  mul_556 = unsqueeze_1485 = None
        unsqueeze_1486: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_1487: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1486, -1);  unsqueeze_1486 = None
        add_474: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_557, unsqueeze_1487);  mul_557 = unsqueeze_1487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_180: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_474);  add_474 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:99 in forward, code: spo.append(self.pool(spx[-1]))
        split_179 = torch.ops.aten.split.Tensor(relu_173, 28, 1);  relu_173 = None
        getitem_1443: "f32[8, 28, 56, 56]" = split_179[7];  split_179 = None
        avg_pool2d_5: "f32[8, 28, 28, 28]" = torch.ops.aten.avg_pool2d.default(getitem_1443, [3, 3], [2, 2], [1, 1]);  getitem_1443 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_19: "f32[8, 224, 28, 28]" = torch.ops.aten.cat.default([relu_174, relu_175, relu_176, relu_177, relu_178, relu_179, relu_180, avg_pool2d_5], 1);  relu_174 = relu_175 = relu_176 = relu_177 = relu_178 = relu_179 = relu_180 = avg_pool2d_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_186: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_19, arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_19 = arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_475: "f32[512]" = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_186: "f32[512]" = torch.ops.aten.sqrt.default(add_475);  add_475 = None
        reciprocal_186: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_186);  sqrt_186 = None
        mul_558: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_186, 1);  reciprocal_186 = None
        unsqueeze_1488: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_1489: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1488, -1);  unsqueeze_1488 = None
        unsqueeze_1490: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_558, -1);  mul_558 = None
        unsqueeze_1491: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, -1);  unsqueeze_1490 = None
        sub_186: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_186, unsqueeze_1489);  convolution_186 = unsqueeze_1489 = None
        mul_559: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_1491);  sub_186 = unsqueeze_1491 = None
        unsqueeze_1492: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_1493: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1492, -1);  unsqueeze_1492 = None
        mul_560: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_559, unsqueeze_1493);  mul_559 = unsqueeze_1493 = None
        unsqueeze_1494: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_1495: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, -1);  unsqueeze_1494 = None
        add_476: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_560, unsqueeze_1495);  mul_560 = unsqueeze_1495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:111 in forward, code: shortcut = self.downsample(x)
        convolution_187: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_172, arg191_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_172 = arg191_1 = None
        add_477: "f32[512]" = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_187: "f32[512]" = torch.ops.aten.sqrt.default(add_477);  add_477 = None
        reciprocal_187: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_187);  sqrt_187 = None
        mul_561: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_187, 1);  reciprocal_187 = None
        unsqueeze_1496: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_1497: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1496, -1);  unsqueeze_1496 = None
        unsqueeze_1498: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_561, -1);  mul_561 = None
        unsqueeze_1499: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, -1);  unsqueeze_1498 = None
        sub_187: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_187, unsqueeze_1497);  convolution_187 = unsqueeze_1497 = None
        mul_562: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_1499);  sub_187 = unsqueeze_1499 = None
        unsqueeze_1500: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_1501: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1500, -1);  unsqueeze_1500 = None
        mul_563: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_562, unsqueeze_1501);  mul_562 = unsqueeze_1501 = None
        unsqueeze_1502: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_1503: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1502, -1);  unsqueeze_1502 = None
        add_478: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_563, unsqueeze_1503);  mul_563 = unsqueeze_1503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_479: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_476, add_478);  add_476 = add_478 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_181: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_479);  add_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_188: "f32[8, 224, 28, 28]" = torch.ops.aten.convolution.default(relu_181, arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_480: "f32[224]" = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_188: "f32[224]" = torch.ops.aten.sqrt.default(add_480);  add_480 = None
        reciprocal_188: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_188);  sqrt_188 = None
        mul_564: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_188, 1);  reciprocal_188 = None
        unsqueeze_1504: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_1505: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1504, -1);  unsqueeze_1504 = None
        unsqueeze_1506: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_564, -1);  mul_564 = None
        unsqueeze_1507: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, -1);  unsqueeze_1506 = None
        sub_188: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_188, unsqueeze_1505);  convolution_188 = unsqueeze_1505 = None
        mul_565: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_1507);  sub_188 = unsqueeze_1507 = None
        unsqueeze_1508: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_1509: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1508, -1);  unsqueeze_1508 = None
        mul_566: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(mul_565, unsqueeze_1509);  mul_565 = unsqueeze_1509 = None
        unsqueeze_1510: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_1511: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1510, -1);  unsqueeze_1510 = None
        add_481: "f32[8, 224, 28, 28]" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_1511);  mul_566 = unsqueeze_1511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_182: "f32[8, 224, 28, 28]" = torch.ops.aten.relu.default(add_481);  add_481 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_181 = torch.ops.aten.split.Tensor(relu_182, 28, 1)
        getitem_1452: "f32[8, 28, 28, 28]" = split_181[0];  split_181 = None
        convolution_189: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_1452, arg201_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1452 = arg201_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_482: "f32[28]" = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_189: "f32[28]" = torch.ops.aten.sqrt.default(add_482);  add_482 = None
        reciprocal_189: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_189);  sqrt_189 = None
        mul_567: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_189, 1);  reciprocal_189 = None
        unsqueeze_1512: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_1513: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1512, -1);  unsqueeze_1512 = None
        unsqueeze_1514: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_567, -1);  mul_567 = None
        unsqueeze_1515: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1514, -1);  unsqueeze_1514 = None
        sub_189: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_189, unsqueeze_1513);  convolution_189 = unsqueeze_1513 = None
        mul_568: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_1515);  sub_189 = unsqueeze_1515 = None
        unsqueeze_1516: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_1517: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1516, -1);  unsqueeze_1516 = None
        mul_569: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_568, unsqueeze_1517);  mul_568 = unsqueeze_1517 = None
        unsqueeze_1518: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_1519: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, -1);  unsqueeze_1518 = None
        add_483: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_569, unsqueeze_1519);  mul_569 = unsqueeze_1519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_183: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_483);  add_483 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_182 = torch.ops.aten.split.Tensor(relu_182, 28, 1)
        getitem_1461: "f32[8, 28, 28, 28]" = split_182[1];  split_182 = None
        add_484: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_183, getitem_1461);  getitem_1461 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_190: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_484, arg206_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_484 = arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_485: "f32[28]" = torch.ops.aten.add.Tensor(arg208_1, 1e-05);  arg208_1 = None
        sqrt_190: "f32[28]" = torch.ops.aten.sqrt.default(add_485);  add_485 = None
        reciprocal_190: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_190);  sqrt_190 = None
        mul_570: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_190, 1);  reciprocal_190 = None
        unsqueeze_1520: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_1521: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1520, -1);  unsqueeze_1520 = None
        unsqueeze_1522: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_570, -1);  mul_570 = None
        unsqueeze_1523: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1522, -1);  unsqueeze_1522 = None
        sub_190: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_190, unsqueeze_1521);  convolution_190 = unsqueeze_1521 = None
        mul_571: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_1523);  sub_190 = unsqueeze_1523 = None
        unsqueeze_1524: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_1525: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1524, -1);  unsqueeze_1524 = None
        mul_572: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_571, unsqueeze_1525);  mul_571 = unsqueeze_1525 = None
        unsqueeze_1526: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_1527: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1526, -1);  unsqueeze_1526 = None
        add_486: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_572, unsqueeze_1527);  mul_572 = unsqueeze_1527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_184: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_486);  add_486 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_183 = torch.ops.aten.split.Tensor(relu_182, 28, 1)
        getitem_1470: "f32[8, 28, 28, 28]" = split_183[2];  split_183 = None
        add_487: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_184, getitem_1470);  getitem_1470 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_191: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_487, arg211_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_487 = arg211_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_488: "f32[28]" = torch.ops.aten.add.Tensor(arg213_1, 1e-05);  arg213_1 = None
        sqrt_191: "f32[28]" = torch.ops.aten.sqrt.default(add_488);  add_488 = None
        reciprocal_191: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_191);  sqrt_191 = None
        mul_573: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_191, 1);  reciprocal_191 = None
        unsqueeze_1528: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_1529: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1528, -1);  unsqueeze_1528 = None
        unsqueeze_1530: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_573, -1);  mul_573 = None
        unsqueeze_1531: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, -1);  unsqueeze_1530 = None
        sub_191: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_191, unsqueeze_1529);  convolution_191 = unsqueeze_1529 = None
        mul_574: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_1531);  sub_191 = unsqueeze_1531 = None
        unsqueeze_1532: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_1533: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1532, -1);  unsqueeze_1532 = None
        mul_575: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_1533);  mul_574 = unsqueeze_1533 = None
        unsqueeze_1534: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_1535: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1534, -1);  unsqueeze_1534 = None
        add_489: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_575, unsqueeze_1535);  mul_575 = unsqueeze_1535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_185: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_489);  add_489 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_184 = torch.ops.aten.split.Tensor(relu_182, 28, 1)
        getitem_1479: "f32[8, 28, 28, 28]" = split_184[3];  split_184 = None
        add_490: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_185, getitem_1479);  getitem_1479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_192: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_490, arg216_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_490 = arg216_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_491: "f32[28]" = torch.ops.aten.add.Tensor(arg218_1, 1e-05);  arg218_1 = None
        sqrt_192: "f32[28]" = torch.ops.aten.sqrt.default(add_491);  add_491 = None
        reciprocal_192: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_192);  sqrt_192 = None
        mul_576: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_192, 1);  reciprocal_192 = None
        unsqueeze_1536: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_1537: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1536, -1);  unsqueeze_1536 = None
        unsqueeze_1538: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_576, -1);  mul_576 = None
        unsqueeze_1539: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1538, -1);  unsqueeze_1538 = None
        sub_192: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_192, unsqueeze_1537);  convolution_192 = unsqueeze_1537 = None
        mul_577: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_1539);  sub_192 = unsqueeze_1539 = None
        unsqueeze_1540: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_1541: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1540, -1);  unsqueeze_1540 = None
        mul_578: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_577, unsqueeze_1541);  mul_577 = unsqueeze_1541 = None
        unsqueeze_1542: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_1543: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, -1);  unsqueeze_1542 = None
        add_492: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_578, unsqueeze_1543);  mul_578 = unsqueeze_1543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_186: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_492);  add_492 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_185 = torch.ops.aten.split.Tensor(relu_182, 28, 1)
        getitem_1488: "f32[8, 28, 28, 28]" = split_185[4];  split_185 = None
        add_493: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_186, getitem_1488);  getitem_1488 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_193: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_493, arg221_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_493 = arg221_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_494: "f32[28]" = torch.ops.aten.add.Tensor(arg223_1, 1e-05);  arg223_1 = None
        sqrt_193: "f32[28]" = torch.ops.aten.sqrt.default(add_494);  add_494 = None
        reciprocal_193: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_193);  sqrt_193 = None
        mul_579: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_193, 1);  reciprocal_193 = None
        unsqueeze_1544: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_1545: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1544, -1);  unsqueeze_1544 = None
        unsqueeze_1546: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_579, -1);  mul_579 = None
        unsqueeze_1547: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1546, -1);  unsqueeze_1546 = None
        sub_193: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_193, unsqueeze_1545);  convolution_193 = unsqueeze_1545 = None
        mul_580: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_1547);  sub_193 = unsqueeze_1547 = None
        unsqueeze_1548: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_1549: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1548, -1);  unsqueeze_1548 = None
        mul_581: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_580, unsqueeze_1549);  mul_580 = unsqueeze_1549 = None
        unsqueeze_1550: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_1551: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1550, -1);  unsqueeze_1550 = None
        add_495: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_581, unsqueeze_1551);  mul_581 = unsqueeze_1551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_187: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_495);  add_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_186 = torch.ops.aten.split.Tensor(relu_182, 28, 1)
        getitem_1497: "f32[8, 28, 28, 28]" = split_186[5];  split_186 = None
        add_496: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_187, getitem_1497);  getitem_1497 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_194: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_496, arg226_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_496 = arg226_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_497: "f32[28]" = torch.ops.aten.add.Tensor(arg228_1, 1e-05);  arg228_1 = None
        sqrt_194: "f32[28]" = torch.ops.aten.sqrt.default(add_497);  add_497 = None
        reciprocal_194: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_194);  sqrt_194 = None
        mul_582: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_194, 1);  reciprocal_194 = None
        unsqueeze_1552: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_1553: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1552, -1);  unsqueeze_1552 = None
        unsqueeze_1554: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_582, -1);  mul_582 = None
        unsqueeze_1555: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, -1);  unsqueeze_1554 = None
        sub_194: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_194, unsqueeze_1553);  convolution_194 = unsqueeze_1553 = None
        mul_583: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_1555);  sub_194 = unsqueeze_1555 = None
        unsqueeze_1556: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_1557: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1556, -1);  unsqueeze_1556 = None
        mul_584: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_583, unsqueeze_1557);  mul_583 = unsqueeze_1557 = None
        unsqueeze_1558: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_1559: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1558, -1);  unsqueeze_1558 = None
        add_498: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_584, unsqueeze_1559);  mul_584 = unsqueeze_1559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_188: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_498);  add_498 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_187 = torch.ops.aten.split.Tensor(relu_182, 28, 1)
        getitem_1506: "f32[8, 28, 28, 28]" = split_187[6];  split_187 = None
        add_499: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_188, getitem_1506);  getitem_1506 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_195: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_499, arg231_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_499 = arg231_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_500: "f32[28]" = torch.ops.aten.add.Tensor(arg233_1, 1e-05);  arg233_1 = None
        sqrt_195: "f32[28]" = torch.ops.aten.sqrt.default(add_500);  add_500 = None
        reciprocal_195: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_195);  sqrt_195 = None
        mul_585: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_195, 1);  reciprocal_195 = None
        unsqueeze_1560: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_1561: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1560, -1);  unsqueeze_1560 = None
        unsqueeze_1562: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_585, -1);  mul_585 = None
        unsqueeze_1563: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1562, -1);  unsqueeze_1562 = None
        sub_195: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_195, unsqueeze_1561);  convolution_195 = unsqueeze_1561 = None
        mul_586: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_1563);  sub_195 = unsqueeze_1563 = None
        unsqueeze_1564: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_1565: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1564, -1);  unsqueeze_1564 = None
        mul_587: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_586, unsqueeze_1565);  mul_586 = unsqueeze_1565 = None
        unsqueeze_1566: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_1567: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, -1);  unsqueeze_1566 = None
        add_501: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_1567);  mul_587 = unsqueeze_1567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_189: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_501);  add_501 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_188 = torch.ops.aten.split.Tensor(relu_182, 28, 1);  relu_182 = None
        getitem_1515: "f32[8, 28, 28, 28]" = split_188[7];  split_188 = None
        cat_20: "f32[8, 224, 28, 28]" = torch.ops.aten.cat.default([relu_183, relu_184, relu_185, relu_186, relu_187, relu_188, relu_189, getitem_1515], 1);  relu_183 = relu_184 = relu_185 = relu_186 = relu_187 = relu_188 = relu_189 = getitem_1515 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_196: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_20, arg236_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_20 = arg236_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_502: "f32[512]" = torch.ops.aten.add.Tensor(arg238_1, 1e-05);  arg238_1 = None
        sqrt_196: "f32[512]" = torch.ops.aten.sqrt.default(add_502);  add_502 = None
        reciprocal_196: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_196);  sqrt_196 = None
        mul_588: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_196, 1);  reciprocal_196 = None
        unsqueeze_1568: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_1569: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1568, -1);  unsqueeze_1568 = None
        unsqueeze_1570: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_588, -1);  mul_588 = None
        unsqueeze_1571: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1570, -1);  unsqueeze_1570 = None
        sub_196: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_196, unsqueeze_1569);  convolution_196 = unsqueeze_1569 = None
        mul_589: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_1571);  sub_196 = unsqueeze_1571 = None
        unsqueeze_1572: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_1573: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1572, -1);  unsqueeze_1572 = None
        mul_590: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_589, unsqueeze_1573);  mul_589 = unsqueeze_1573 = None
        unsqueeze_1574: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_1575: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1574, -1);  unsqueeze_1574 = None
        add_503: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_590, unsqueeze_1575);  mul_590 = unsqueeze_1575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_504: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_503, relu_181);  add_503 = relu_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_190: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_504);  add_504 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_197: "f32[8, 224, 28, 28]" = torch.ops.aten.convolution.default(relu_190, arg241_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg241_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_505: "f32[224]" = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_197: "f32[224]" = torch.ops.aten.sqrt.default(add_505);  add_505 = None
        reciprocal_197: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_197);  sqrt_197 = None
        mul_591: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_197, 1);  reciprocal_197 = None
        unsqueeze_1576: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_1577: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1576, -1);  unsqueeze_1576 = None
        unsqueeze_1578: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_591, -1);  mul_591 = None
        unsqueeze_1579: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, -1);  unsqueeze_1578 = None
        sub_197: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_197, unsqueeze_1577);  convolution_197 = unsqueeze_1577 = None
        mul_592: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_1579);  sub_197 = unsqueeze_1579 = None
        unsqueeze_1580: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_1581: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1580, -1);  unsqueeze_1580 = None
        mul_593: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(mul_592, unsqueeze_1581);  mul_592 = unsqueeze_1581 = None
        unsqueeze_1582: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_1583: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1582, -1);  unsqueeze_1582 = None
        add_506: "f32[8, 224, 28, 28]" = torch.ops.aten.add.Tensor(mul_593, unsqueeze_1583);  mul_593 = unsqueeze_1583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_191: "f32[8, 224, 28, 28]" = torch.ops.aten.relu.default(add_506);  add_506 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_190 = torch.ops.aten.split.Tensor(relu_191, 28, 1)
        getitem_1524: "f32[8, 28, 28, 28]" = split_190[0];  split_190 = None
        convolution_198: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_1524, arg246_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1524 = arg246_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_507: "f32[28]" = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
        sqrt_198: "f32[28]" = torch.ops.aten.sqrt.default(add_507);  add_507 = None
        reciprocal_198: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_198);  sqrt_198 = None
        mul_594: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_198, 1);  reciprocal_198 = None
        unsqueeze_1584: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_1585: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1584, -1);  unsqueeze_1584 = None
        unsqueeze_1586: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_594, -1);  mul_594 = None
        unsqueeze_1587: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1586, -1);  unsqueeze_1586 = None
        sub_198: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_198, unsqueeze_1585);  convolution_198 = unsqueeze_1585 = None
        mul_595: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_1587);  sub_198 = unsqueeze_1587 = None
        unsqueeze_1588: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_1589: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1588, -1);  unsqueeze_1588 = None
        mul_596: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_1589);  mul_595 = unsqueeze_1589 = None
        unsqueeze_1590: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_1591: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, -1);  unsqueeze_1590 = None
        add_508: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_596, unsqueeze_1591);  mul_596 = unsqueeze_1591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_192: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_508);  add_508 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_191 = torch.ops.aten.split.Tensor(relu_191, 28, 1)
        getitem_1533: "f32[8, 28, 28, 28]" = split_191[1];  split_191 = None
        add_509: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_192, getitem_1533);  getitem_1533 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_199: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_509, arg251_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_509 = arg251_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_510: "f32[28]" = torch.ops.aten.add.Tensor(arg253_1, 1e-05);  arg253_1 = None
        sqrt_199: "f32[28]" = torch.ops.aten.sqrt.default(add_510);  add_510 = None
        reciprocal_199: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_199);  sqrt_199 = None
        mul_597: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_199, 1);  reciprocal_199 = None
        unsqueeze_1592: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_1593: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1592, -1);  unsqueeze_1592 = None
        unsqueeze_1594: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_597, -1);  mul_597 = None
        unsqueeze_1595: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1594, -1);  unsqueeze_1594 = None
        sub_199: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_199, unsqueeze_1593);  convolution_199 = unsqueeze_1593 = None
        mul_598: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_1595);  sub_199 = unsqueeze_1595 = None
        unsqueeze_1596: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_1597: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1596, -1);  unsqueeze_1596 = None
        mul_599: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_598, unsqueeze_1597);  mul_598 = unsqueeze_1597 = None
        unsqueeze_1598: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_1599: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1598, -1);  unsqueeze_1598 = None
        add_511: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_599, unsqueeze_1599);  mul_599 = unsqueeze_1599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_193: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_511);  add_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_192 = torch.ops.aten.split.Tensor(relu_191, 28, 1)
        getitem_1542: "f32[8, 28, 28, 28]" = split_192[2];  split_192 = None
        add_512: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_193, getitem_1542);  getitem_1542 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_200: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_512, arg256_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_512 = arg256_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_513: "f32[28]" = torch.ops.aten.add.Tensor(arg258_1, 1e-05);  arg258_1 = None
        sqrt_200: "f32[28]" = torch.ops.aten.sqrt.default(add_513);  add_513 = None
        reciprocal_200: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_200);  sqrt_200 = None
        mul_600: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_200, 1);  reciprocal_200 = None
        unsqueeze_1600: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_1601: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1600, -1);  unsqueeze_1600 = None
        unsqueeze_1602: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_600, -1);  mul_600 = None
        unsqueeze_1603: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, -1);  unsqueeze_1602 = None
        sub_200: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_200, unsqueeze_1601);  convolution_200 = unsqueeze_1601 = None
        mul_601: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_1603);  sub_200 = unsqueeze_1603 = None
        unsqueeze_1604: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_1605: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1604, -1);  unsqueeze_1604 = None
        mul_602: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_601, unsqueeze_1605);  mul_601 = unsqueeze_1605 = None
        unsqueeze_1606: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_1607: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1606, -1);  unsqueeze_1606 = None
        add_514: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_602, unsqueeze_1607);  mul_602 = unsqueeze_1607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_194: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_514);  add_514 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_193 = torch.ops.aten.split.Tensor(relu_191, 28, 1)
        getitem_1551: "f32[8, 28, 28, 28]" = split_193[3];  split_193 = None
        add_515: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_194, getitem_1551);  getitem_1551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_201: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_515, arg261_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_515 = arg261_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_516: "f32[28]" = torch.ops.aten.add.Tensor(arg263_1, 1e-05);  arg263_1 = None
        sqrt_201: "f32[28]" = torch.ops.aten.sqrt.default(add_516);  add_516 = None
        reciprocal_201: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_201);  sqrt_201 = None
        mul_603: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_201, 1);  reciprocal_201 = None
        unsqueeze_1608: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_1609: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1608, -1);  unsqueeze_1608 = None
        unsqueeze_1610: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_603, -1);  mul_603 = None
        unsqueeze_1611: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1610, -1);  unsqueeze_1610 = None
        sub_201: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_201, unsqueeze_1609);  convolution_201 = unsqueeze_1609 = None
        mul_604: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_1611);  sub_201 = unsqueeze_1611 = None
        unsqueeze_1612: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_1613: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1612, -1);  unsqueeze_1612 = None
        mul_605: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_604, unsqueeze_1613);  mul_604 = unsqueeze_1613 = None
        unsqueeze_1614: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_1615: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, -1);  unsqueeze_1614 = None
        add_517: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_605, unsqueeze_1615);  mul_605 = unsqueeze_1615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_195: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_517);  add_517 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_194 = torch.ops.aten.split.Tensor(relu_191, 28, 1)
        getitem_1560: "f32[8, 28, 28, 28]" = split_194[4];  split_194 = None
        add_518: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_195, getitem_1560);  getitem_1560 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_202: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_518, arg266_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_518 = arg266_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_519: "f32[28]" = torch.ops.aten.add.Tensor(arg268_1, 1e-05);  arg268_1 = None
        sqrt_202: "f32[28]" = torch.ops.aten.sqrt.default(add_519);  add_519 = None
        reciprocal_202: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_202);  sqrt_202 = None
        mul_606: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_202, 1);  reciprocal_202 = None
        unsqueeze_1616: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_1617: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1616, -1);  unsqueeze_1616 = None
        unsqueeze_1618: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_606, -1);  mul_606 = None
        unsqueeze_1619: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1618, -1);  unsqueeze_1618 = None
        sub_202: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_202, unsqueeze_1617);  convolution_202 = unsqueeze_1617 = None
        mul_607: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_1619);  sub_202 = unsqueeze_1619 = None
        unsqueeze_1620: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_1621: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1620, -1);  unsqueeze_1620 = None
        mul_608: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_607, unsqueeze_1621);  mul_607 = unsqueeze_1621 = None
        unsqueeze_1622: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_1623: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1622, -1);  unsqueeze_1622 = None
        add_520: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_608, unsqueeze_1623);  mul_608 = unsqueeze_1623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_196: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_520);  add_520 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_195 = torch.ops.aten.split.Tensor(relu_191, 28, 1)
        getitem_1569: "f32[8, 28, 28, 28]" = split_195[5];  split_195 = None
        add_521: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_196, getitem_1569);  getitem_1569 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_203: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_521, arg271_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_521 = arg271_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_522: "f32[28]" = torch.ops.aten.add.Tensor(arg273_1, 1e-05);  arg273_1 = None
        sqrt_203: "f32[28]" = torch.ops.aten.sqrt.default(add_522);  add_522 = None
        reciprocal_203: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_203);  sqrt_203 = None
        mul_609: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_203, 1);  reciprocal_203 = None
        unsqueeze_1624: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_1625: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1624, -1);  unsqueeze_1624 = None
        unsqueeze_1626: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_609, -1);  mul_609 = None
        unsqueeze_1627: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, -1);  unsqueeze_1626 = None
        sub_203: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_203, unsqueeze_1625);  convolution_203 = unsqueeze_1625 = None
        mul_610: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_1627);  sub_203 = unsqueeze_1627 = None
        unsqueeze_1628: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_1629: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1628, -1);  unsqueeze_1628 = None
        mul_611: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_610, unsqueeze_1629);  mul_610 = unsqueeze_1629 = None
        unsqueeze_1630: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_1631: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1630, -1);  unsqueeze_1630 = None
        add_523: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_611, unsqueeze_1631);  mul_611 = unsqueeze_1631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_197: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_523);  add_523 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_196 = torch.ops.aten.split.Tensor(relu_191, 28, 1)
        getitem_1578: "f32[8, 28, 28, 28]" = split_196[6];  split_196 = None
        add_524: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_197, getitem_1578);  getitem_1578 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_204: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_524, arg276_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_524 = arg276_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_525: "f32[28]" = torch.ops.aten.add.Tensor(arg278_1, 1e-05);  arg278_1 = None
        sqrt_204: "f32[28]" = torch.ops.aten.sqrt.default(add_525);  add_525 = None
        reciprocal_204: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_204);  sqrt_204 = None
        mul_612: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_204, 1);  reciprocal_204 = None
        unsqueeze_1632: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_1633: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1632, -1);  unsqueeze_1632 = None
        unsqueeze_1634: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_612, -1);  mul_612 = None
        unsqueeze_1635: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1634, -1);  unsqueeze_1634 = None
        sub_204: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_204, unsqueeze_1633);  convolution_204 = unsqueeze_1633 = None
        mul_613: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_1635);  sub_204 = unsqueeze_1635 = None
        unsqueeze_1636: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_1637: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1636, -1);  unsqueeze_1636 = None
        mul_614: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_613, unsqueeze_1637);  mul_613 = unsqueeze_1637 = None
        unsqueeze_1638: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_1639: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, -1);  unsqueeze_1638 = None
        add_526: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_614, unsqueeze_1639);  mul_614 = unsqueeze_1639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_198: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_526);  add_526 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_197 = torch.ops.aten.split.Tensor(relu_191, 28, 1);  relu_191 = None
        getitem_1587: "f32[8, 28, 28, 28]" = split_197[7];  split_197 = None
        cat_21: "f32[8, 224, 28, 28]" = torch.ops.aten.cat.default([relu_192, relu_193, relu_194, relu_195, relu_196, relu_197, relu_198, getitem_1587], 1);  relu_192 = relu_193 = relu_194 = relu_195 = relu_196 = relu_197 = relu_198 = getitem_1587 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_205: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_21, arg281_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_21 = arg281_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_527: "f32[512]" = torch.ops.aten.add.Tensor(arg283_1, 1e-05);  arg283_1 = None
        sqrt_205: "f32[512]" = torch.ops.aten.sqrt.default(add_527);  add_527 = None
        reciprocal_205: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_205);  sqrt_205 = None
        mul_615: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_205, 1);  reciprocal_205 = None
        unsqueeze_1640: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_1641: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1640, -1);  unsqueeze_1640 = None
        unsqueeze_1642: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_615, -1);  mul_615 = None
        unsqueeze_1643: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1642, -1);  unsqueeze_1642 = None
        sub_205: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_205, unsqueeze_1641);  convolution_205 = unsqueeze_1641 = None
        mul_616: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_1643);  sub_205 = unsqueeze_1643 = None
        unsqueeze_1644: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_1645: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1644, -1);  unsqueeze_1644 = None
        mul_617: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_1645);  mul_616 = unsqueeze_1645 = None
        unsqueeze_1646: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_1647: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1646, -1);  unsqueeze_1646 = None
        add_528: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_617, unsqueeze_1647);  mul_617 = unsqueeze_1647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_529: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_528, relu_190);  add_528 = relu_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_199: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_529);  add_529 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_206: "f32[8, 224, 28, 28]" = torch.ops.aten.convolution.default(relu_199, arg286_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg286_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_530: "f32[224]" = torch.ops.aten.add.Tensor(arg288_1, 1e-05);  arg288_1 = None
        sqrt_206: "f32[224]" = torch.ops.aten.sqrt.default(add_530);  add_530 = None
        reciprocal_206: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_206);  sqrt_206 = None
        mul_618: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_206, 1);  reciprocal_206 = None
        unsqueeze_1648: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_1649: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1648, -1);  unsqueeze_1648 = None
        unsqueeze_1650: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_618, -1);  mul_618 = None
        unsqueeze_1651: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, -1);  unsqueeze_1650 = None
        sub_206: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_206, unsqueeze_1649);  convolution_206 = unsqueeze_1649 = None
        mul_619: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_1651);  sub_206 = unsqueeze_1651 = None
        unsqueeze_1652: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_1653: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1652, -1);  unsqueeze_1652 = None
        mul_620: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(mul_619, unsqueeze_1653);  mul_619 = unsqueeze_1653 = None
        unsqueeze_1654: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_1655: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1654, -1);  unsqueeze_1654 = None
        add_531: "f32[8, 224, 28, 28]" = torch.ops.aten.add.Tensor(mul_620, unsqueeze_1655);  mul_620 = unsqueeze_1655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_200: "f32[8, 224, 28, 28]" = torch.ops.aten.relu.default(add_531);  add_531 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_199 = torch.ops.aten.split.Tensor(relu_200, 28, 1)
        getitem_1596: "f32[8, 28, 28, 28]" = split_199[0];  split_199 = None
        convolution_207: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_1596, arg291_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1596 = arg291_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_532: "f32[28]" = torch.ops.aten.add.Tensor(arg293_1, 1e-05);  arg293_1 = None
        sqrt_207: "f32[28]" = torch.ops.aten.sqrt.default(add_532);  add_532 = None
        reciprocal_207: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_207);  sqrt_207 = None
        mul_621: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_207, 1);  reciprocal_207 = None
        unsqueeze_1656: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_1657: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1656, -1);  unsqueeze_1656 = None
        unsqueeze_1658: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_621, -1);  mul_621 = None
        unsqueeze_1659: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1658, -1);  unsqueeze_1658 = None
        sub_207: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_207, unsqueeze_1657);  convolution_207 = unsqueeze_1657 = None
        mul_622: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_1659);  sub_207 = unsqueeze_1659 = None
        unsqueeze_1660: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_1661: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1660, -1);  unsqueeze_1660 = None
        mul_623: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_622, unsqueeze_1661);  mul_622 = unsqueeze_1661 = None
        unsqueeze_1662: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_1663: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, -1);  unsqueeze_1662 = None
        add_533: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_623, unsqueeze_1663);  mul_623 = unsqueeze_1663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_201: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_533);  add_533 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_200 = torch.ops.aten.split.Tensor(relu_200, 28, 1)
        getitem_1605: "f32[8, 28, 28, 28]" = split_200[1];  split_200 = None
        add_534: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_201, getitem_1605);  getitem_1605 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_208: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_534, arg296_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_534 = arg296_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_535: "f32[28]" = torch.ops.aten.add.Tensor(arg298_1, 1e-05);  arg298_1 = None
        sqrt_208: "f32[28]" = torch.ops.aten.sqrt.default(add_535);  add_535 = None
        reciprocal_208: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_208);  sqrt_208 = None
        mul_624: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_208, 1);  reciprocal_208 = None
        unsqueeze_1664: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg297_1, -1);  arg297_1 = None
        unsqueeze_1665: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1664, -1);  unsqueeze_1664 = None
        unsqueeze_1666: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_624, -1);  mul_624 = None
        unsqueeze_1667: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1666, -1);  unsqueeze_1666 = None
        sub_208: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_208, unsqueeze_1665);  convolution_208 = unsqueeze_1665 = None
        mul_625: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_1667);  sub_208 = unsqueeze_1667 = None
        unsqueeze_1668: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_1669: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1668, -1);  unsqueeze_1668 = None
        mul_626: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_625, unsqueeze_1669);  mul_625 = unsqueeze_1669 = None
        unsqueeze_1670: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_1671: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1670, -1);  unsqueeze_1670 = None
        add_536: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_626, unsqueeze_1671);  mul_626 = unsqueeze_1671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_202: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_536);  add_536 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_201 = torch.ops.aten.split.Tensor(relu_200, 28, 1)
        getitem_1614: "f32[8, 28, 28, 28]" = split_201[2];  split_201 = None
        add_537: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_202, getitem_1614);  getitem_1614 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_209: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_537, arg301_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_537 = arg301_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_538: "f32[28]" = torch.ops.aten.add.Tensor(arg303_1, 1e-05);  arg303_1 = None
        sqrt_209: "f32[28]" = torch.ops.aten.sqrt.default(add_538);  add_538 = None
        reciprocal_209: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_209);  sqrt_209 = None
        mul_627: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_209, 1);  reciprocal_209 = None
        unsqueeze_1672: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_1673: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1672, -1);  unsqueeze_1672 = None
        unsqueeze_1674: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_627, -1);  mul_627 = None
        unsqueeze_1675: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1674, -1);  unsqueeze_1674 = None
        sub_209: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_209, unsqueeze_1673);  convolution_209 = unsqueeze_1673 = None
        mul_628: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_1675);  sub_209 = unsqueeze_1675 = None
        unsqueeze_1676: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_1677: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1676, -1);  unsqueeze_1676 = None
        mul_629: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_628, unsqueeze_1677);  mul_628 = unsqueeze_1677 = None
        unsqueeze_1678: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_1679: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1678, -1);  unsqueeze_1678 = None
        add_539: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_629, unsqueeze_1679);  mul_629 = unsqueeze_1679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_203: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_539);  add_539 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_202 = torch.ops.aten.split.Tensor(relu_200, 28, 1)
        getitem_1623: "f32[8, 28, 28, 28]" = split_202[3];  split_202 = None
        add_540: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_203, getitem_1623);  getitem_1623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_210: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_540, arg306_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_540 = arg306_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_541: "f32[28]" = torch.ops.aten.add.Tensor(arg308_1, 1e-05);  arg308_1 = None
        sqrt_210: "f32[28]" = torch.ops.aten.sqrt.default(add_541);  add_541 = None
        reciprocal_210: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_210);  sqrt_210 = None
        mul_630: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_210, 1);  reciprocal_210 = None
        unsqueeze_1680: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg307_1, -1);  arg307_1 = None
        unsqueeze_1681: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1680, -1);  unsqueeze_1680 = None
        unsqueeze_1682: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_630, -1);  mul_630 = None
        unsqueeze_1683: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1682, -1);  unsqueeze_1682 = None
        sub_210: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_210, unsqueeze_1681);  convolution_210 = unsqueeze_1681 = None
        mul_631: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_1683);  sub_210 = unsqueeze_1683 = None
        unsqueeze_1684: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_1685: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1684, -1);  unsqueeze_1684 = None
        mul_632: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_631, unsqueeze_1685);  mul_631 = unsqueeze_1685 = None
        unsqueeze_1686: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
        unsqueeze_1687: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1686, -1);  unsqueeze_1686 = None
        add_542: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_632, unsqueeze_1687);  mul_632 = unsqueeze_1687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_204: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_542);  add_542 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_203 = torch.ops.aten.split.Tensor(relu_200, 28, 1)
        getitem_1632: "f32[8, 28, 28, 28]" = split_203[4];  split_203 = None
        add_543: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_204, getitem_1632);  getitem_1632 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_211: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_543, arg311_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_543 = arg311_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_544: "f32[28]" = torch.ops.aten.add.Tensor(arg313_1, 1e-05);  arg313_1 = None
        sqrt_211: "f32[28]" = torch.ops.aten.sqrt.default(add_544);  add_544 = None
        reciprocal_211: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_211);  sqrt_211 = None
        mul_633: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_211, 1);  reciprocal_211 = None
        unsqueeze_1688: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_1689: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1688, -1);  unsqueeze_1688 = None
        unsqueeze_1690: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_633, -1);  mul_633 = None
        unsqueeze_1691: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1690, -1);  unsqueeze_1690 = None
        sub_211: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_211, unsqueeze_1689);  convolution_211 = unsqueeze_1689 = None
        mul_634: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_1691);  sub_211 = unsqueeze_1691 = None
        unsqueeze_1692: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_1693: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1692, -1);  unsqueeze_1692 = None
        mul_635: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_634, unsqueeze_1693);  mul_634 = unsqueeze_1693 = None
        unsqueeze_1694: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg315_1, -1);  arg315_1 = None
        unsqueeze_1695: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1694, -1);  unsqueeze_1694 = None
        add_545: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_635, unsqueeze_1695);  mul_635 = unsqueeze_1695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_205: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_545);  add_545 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_204 = torch.ops.aten.split.Tensor(relu_200, 28, 1)
        getitem_1641: "f32[8, 28, 28, 28]" = split_204[5];  split_204 = None
        add_546: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_205, getitem_1641);  getitem_1641 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_212: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_546, arg316_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_546 = arg316_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_547: "f32[28]" = torch.ops.aten.add.Tensor(arg318_1, 1e-05);  arg318_1 = None
        sqrt_212: "f32[28]" = torch.ops.aten.sqrt.default(add_547);  add_547 = None
        reciprocal_212: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_212);  sqrt_212 = None
        mul_636: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_212, 1);  reciprocal_212 = None
        unsqueeze_1696: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
        unsqueeze_1697: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1696, -1);  unsqueeze_1696 = None
        unsqueeze_1698: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_636, -1);  mul_636 = None
        unsqueeze_1699: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1698, -1);  unsqueeze_1698 = None
        sub_212: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_212, unsqueeze_1697);  convolution_212 = unsqueeze_1697 = None
        mul_637: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_1699);  sub_212 = unsqueeze_1699 = None
        unsqueeze_1700: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_1701: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1700, -1);  unsqueeze_1700 = None
        mul_638: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_1701);  mul_637 = unsqueeze_1701 = None
        unsqueeze_1702: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_1703: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1702, -1);  unsqueeze_1702 = None
        add_548: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_638, unsqueeze_1703);  mul_638 = unsqueeze_1703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_206: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_548);  add_548 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_205 = torch.ops.aten.split.Tensor(relu_200, 28, 1)
        getitem_1650: "f32[8, 28, 28, 28]" = split_205[6];  split_205 = None
        add_549: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(relu_206, getitem_1650);  getitem_1650 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_213: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(add_549, arg321_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_549 = arg321_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_550: "f32[28]" = torch.ops.aten.add.Tensor(arg323_1, 1e-05);  arg323_1 = None
        sqrt_213: "f32[28]" = torch.ops.aten.sqrt.default(add_550);  add_550 = None
        reciprocal_213: "f32[28]" = torch.ops.aten.reciprocal.default(sqrt_213);  sqrt_213 = None
        mul_639: "f32[28]" = torch.ops.aten.mul.Tensor(reciprocal_213, 1);  reciprocal_213 = None
        unsqueeze_1704: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_1705: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1704, -1);  unsqueeze_1704 = None
        unsqueeze_1706: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(mul_639, -1);  mul_639 = None
        unsqueeze_1707: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1706, -1);  unsqueeze_1706 = None
        sub_213: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_213, unsqueeze_1705);  convolution_213 = unsqueeze_1705 = None
        mul_640: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_1707);  sub_213 = unsqueeze_1707 = None
        unsqueeze_1708: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_1709: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1708, -1);  unsqueeze_1708 = None
        mul_641: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(mul_640, unsqueeze_1709);  mul_640 = unsqueeze_1709 = None
        unsqueeze_1710: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_1711: "f32[28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1710, -1);  unsqueeze_1710 = None
        add_551: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(mul_641, unsqueeze_1711);  mul_641 = unsqueeze_1711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_207: "f32[8, 28, 28, 28]" = torch.ops.aten.relu.default(add_551);  add_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_206 = torch.ops.aten.split.Tensor(relu_200, 28, 1);  relu_200 = None
        getitem_1659: "f32[8, 28, 28, 28]" = split_206[7];  split_206 = None
        cat_22: "f32[8, 224, 28, 28]" = torch.ops.aten.cat.default([relu_201, relu_202, relu_203, relu_204, relu_205, relu_206, relu_207, getitem_1659], 1);  relu_201 = relu_202 = relu_203 = relu_204 = relu_205 = relu_206 = relu_207 = getitem_1659 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_214: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_22, arg326_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_22 = arg326_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_552: "f32[512]" = torch.ops.aten.add.Tensor(arg328_1, 1e-05);  arg328_1 = None
        sqrt_214: "f32[512]" = torch.ops.aten.sqrt.default(add_552);  add_552 = None
        reciprocal_214: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_214);  sqrt_214 = None
        mul_642: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_214, 1);  reciprocal_214 = None
        unsqueeze_1712: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg327_1, -1);  arg327_1 = None
        unsqueeze_1713: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1712, -1);  unsqueeze_1712 = None
        unsqueeze_1714: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_642, -1);  mul_642 = None
        unsqueeze_1715: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1714, -1);  unsqueeze_1714 = None
        sub_214: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_214, unsqueeze_1713);  convolution_214 = unsqueeze_1713 = None
        mul_643: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_1715);  sub_214 = unsqueeze_1715 = None
        unsqueeze_1716: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_1717: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1716, -1);  unsqueeze_1716 = None
        mul_644: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_643, unsqueeze_1717);  mul_643 = unsqueeze_1717 = None
        unsqueeze_1718: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_1719: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1718, -1);  unsqueeze_1718 = None
        add_553: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_644, unsqueeze_1719);  mul_644 = unsqueeze_1719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_554: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_553, relu_199);  add_553 = relu_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_208: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_554);  add_554 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_215: "f32[8, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_208, arg331_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg331_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_555: "f32[448]" = torch.ops.aten.add.Tensor(arg333_1, 1e-05);  arg333_1 = None
        sqrt_215: "f32[448]" = torch.ops.aten.sqrt.default(add_555);  add_555 = None
        reciprocal_215: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_215);  sqrt_215 = None
        mul_645: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_215, 1);  reciprocal_215 = None
        unsqueeze_1720: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg332_1, -1);  arg332_1 = None
        unsqueeze_1721: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1720, -1);  unsqueeze_1720 = None
        unsqueeze_1722: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_645, -1);  mul_645 = None
        unsqueeze_1723: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1722, -1);  unsqueeze_1722 = None
        sub_215: "f32[8, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_215, unsqueeze_1721);  convolution_215 = unsqueeze_1721 = None
        mul_646: "f32[8, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_1723);  sub_215 = unsqueeze_1723 = None
        unsqueeze_1724: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_1725: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1724, -1);  unsqueeze_1724 = None
        mul_647: "f32[8, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_646, unsqueeze_1725);  mul_646 = unsqueeze_1725 = None
        unsqueeze_1726: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
        unsqueeze_1727: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1726, -1);  unsqueeze_1726 = None
        add_556: "f32[8, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_647, unsqueeze_1727);  mul_647 = unsqueeze_1727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_209: "f32[8, 448, 28, 28]" = torch.ops.aten.relu.default(add_556);  add_556 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_208 = torch.ops.aten.split.Tensor(relu_209, 56, 1)
        getitem_1668: "f32[8, 56, 28, 28]" = split_208[0];  split_208 = None
        convolution_216: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_1668, arg336_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1668 = arg336_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_557: "f32[56]" = torch.ops.aten.add.Tensor(arg338_1, 1e-05);  arg338_1 = None
        sqrt_216: "f32[56]" = torch.ops.aten.sqrt.default(add_557);  add_557 = None
        reciprocal_216: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_216);  sqrt_216 = None
        mul_648: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_216, 1);  reciprocal_216 = None
        unsqueeze_1728: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg337_1, -1);  arg337_1 = None
        unsqueeze_1729: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1728, -1);  unsqueeze_1728 = None
        unsqueeze_1730: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_648, -1);  mul_648 = None
        unsqueeze_1731: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1730, -1);  unsqueeze_1730 = None
        sub_216: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_216, unsqueeze_1729);  convolution_216 = unsqueeze_1729 = None
        mul_649: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_1731);  sub_216 = unsqueeze_1731 = None
        unsqueeze_1732: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_1733: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1732, -1);  unsqueeze_1732 = None
        mul_650: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_649, unsqueeze_1733);  mul_649 = unsqueeze_1733 = None
        unsqueeze_1734: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg340_1, -1);  arg340_1 = None
        unsqueeze_1735: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1734, -1);  unsqueeze_1734 = None
        add_558: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_650, unsqueeze_1735);  mul_650 = unsqueeze_1735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_210: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_558);  add_558 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_209 = torch.ops.aten.split.Tensor(relu_209, 56, 1)
        getitem_1677: "f32[8, 56, 28, 28]" = split_209[1];  split_209 = None
        convolution_217: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_1677, arg341_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1677 = arg341_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_559: "f32[56]" = torch.ops.aten.add.Tensor(arg343_1, 1e-05);  arg343_1 = None
        sqrt_217: "f32[56]" = torch.ops.aten.sqrt.default(add_559);  add_559 = None
        reciprocal_217: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_217);  sqrt_217 = None
        mul_651: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_217, 1);  reciprocal_217 = None
        unsqueeze_1736: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_1737: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1736, -1);  unsqueeze_1736 = None
        unsqueeze_1738: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_651, -1);  mul_651 = None
        unsqueeze_1739: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1738, -1);  unsqueeze_1738 = None
        sub_217: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_217, unsqueeze_1737);  convolution_217 = unsqueeze_1737 = None
        mul_652: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_1739);  sub_217 = unsqueeze_1739 = None
        unsqueeze_1740: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_1741: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1740, -1);  unsqueeze_1740 = None
        mul_653: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_652, unsqueeze_1741);  mul_652 = unsqueeze_1741 = None
        unsqueeze_1742: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_1743: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1742, -1);  unsqueeze_1742 = None
        add_560: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_653, unsqueeze_1743);  mul_653 = unsqueeze_1743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_211: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_560);  add_560 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_210 = torch.ops.aten.split.Tensor(relu_209, 56, 1)
        getitem_1686: "f32[8, 56, 28, 28]" = split_210[2];  split_210 = None
        convolution_218: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_1686, arg346_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1686 = arg346_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_561: "f32[56]" = torch.ops.aten.add.Tensor(arg348_1, 1e-05);  arg348_1 = None
        sqrt_218: "f32[56]" = torch.ops.aten.sqrt.default(add_561);  add_561 = None
        reciprocal_218: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_218);  sqrt_218 = None
        mul_654: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_218, 1);  reciprocal_218 = None
        unsqueeze_1744: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
        unsqueeze_1745: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1744, -1);  unsqueeze_1744 = None
        unsqueeze_1746: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_654, -1);  mul_654 = None
        unsqueeze_1747: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1746, -1);  unsqueeze_1746 = None
        sub_218: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_218, unsqueeze_1745);  convolution_218 = unsqueeze_1745 = None
        mul_655: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_1747);  sub_218 = unsqueeze_1747 = None
        unsqueeze_1748: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_1749: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1748, -1);  unsqueeze_1748 = None
        mul_656: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_655, unsqueeze_1749);  mul_655 = unsqueeze_1749 = None
        unsqueeze_1750: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
        unsqueeze_1751: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1750, -1);  unsqueeze_1750 = None
        add_562: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_656, unsqueeze_1751);  mul_656 = unsqueeze_1751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_212: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_562);  add_562 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_211 = torch.ops.aten.split.Tensor(relu_209, 56, 1)
        getitem_1695: "f32[8, 56, 28, 28]" = split_211[3];  split_211 = None
        convolution_219: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_1695, arg351_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1695 = arg351_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_563: "f32[56]" = torch.ops.aten.add.Tensor(arg353_1, 1e-05);  arg353_1 = None
        sqrt_219: "f32[56]" = torch.ops.aten.sqrt.default(add_563);  add_563 = None
        reciprocal_219: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_219);  sqrt_219 = None
        mul_657: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_219, 1);  reciprocal_219 = None
        unsqueeze_1752: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
        unsqueeze_1753: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1752, -1);  unsqueeze_1752 = None
        unsqueeze_1754: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_657, -1);  mul_657 = None
        unsqueeze_1755: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1754, -1);  unsqueeze_1754 = None
        sub_219: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_219, unsqueeze_1753);  convolution_219 = unsqueeze_1753 = None
        mul_658: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_1755);  sub_219 = unsqueeze_1755 = None
        unsqueeze_1756: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_1757: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1756, -1);  unsqueeze_1756 = None
        mul_659: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_1757);  mul_658 = unsqueeze_1757 = None
        unsqueeze_1758: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
        unsqueeze_1759: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1758, -1);  unsqueeze_1758 = None
        add_564: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_659, unsqueeze_1759);  mul_659 = unsqueeze_1759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_213: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_564);  add_564 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_212 = torch.ops.aten.split.Tensor(relu_209, 56, 1)
        getitem_1704: "f32[8, 56, 28, 28]" = split_212[4];  split_212 = None
        convolution_220: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_1704, arg356_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1704 = arg356_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_565: "f32[56]" = torch.ops.aten.add.Tensor(arg358_1, 1e-05);  arg358_1 = None
        sqrt_220: "f32[56]" = torch.ops.aten.sqrt.default(add_565);  add_565 = None
        reciprocal_220: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_220);  sqrt_220 = None
        mul_660: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_220, 1);  reciprocal_220 = None
        unsqueeze_1760: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg357_1, -1);  arg357_1 = None
        unsqueeze_1761: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1760, -1);  unsqueeze_1760 = None
        unsqueeze_1762: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_660, -1);  mul_660 = None
        unsqueeze_1763: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1762, -1);  unsqueeze_1762 = None
        sub_220: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_220, unsqueeze_1761);  convolution_220 = unsqueeze_1761 = None
        mul_661: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_1763);  sub_220 = unsqueeze_1763 = None
        unsqueeze_1764: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_1765: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1764, -1);  unsqueeze_1764 = None
        mul_662: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_661, unsqueeze_1765);  mul_661 = unsqueeze_1765 = None
        unsqueeze_1766: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_1767: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1766, -1);  unsqueeze_1766 = None
        add_566: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_662, unsqueeze_1767);  mul_662 = unsqueeze_1767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_214: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_566);  add_566 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_213 = torch.ops.aten.split.Tensor(relu_209, 56, 1)
        getitem_1713: "f32[8, 56, 28, 28]" = split_213[5];  split_213 = None
        convolution_221: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_1713, arg361_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1713 = arg361_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_567: "f32[56]" = torch.ops.aten.add.Tensor(arg363_1, 1e-05);  arg363_1 = None
        sqrt_221: "f32[56]" = torch.ops.aten.sqrt.default(add_567);  add_567 = None
        reciprocal_221: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_221);  sqrt_221 = None
        mul_663: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_221, 1);  reciprocal_221 = None
        unsqueeze_1768: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg362_1, -1);  arg362_1 = None
        unsqueeze_1769: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1768, -1);  unsqueeze_1768 = None
        unsqueeze_1770: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_663, -1);  mul_663 = None
        unsqueeze_1771: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1770, -1);  unsqueeze_1770 = None
        sub_221: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_221, unsqueeze_1769);  convolution_221 = unsqueeze_1769 = None
        mul_664: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_1771);  sub_221 = unsqueeze_1771 = None
        unsqueeze_1772: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
        unsqueeze_1773: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1772, -1);  unsqueeze_1772 = None
        mul_665: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_664, unsqueeze_1773);  mul_664 = unsqueeze_1773 = None
        unsqueeze_1774: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
        unsqueeze_1775: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1774, -1);  unsqueeze_1774 = None
        add_568: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_665, unsqueeze_1775);  mul_665 = unsqueeze_1775 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_215: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_568);  add_568 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_214 = torch.ops.aten.split.Tensor(relu_209, 56, 1)
        getitem_1722: "f32[8, 56, 28, 28]" = split_214[6];  split_214 = None
        convolution_222: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_1722, arg366_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1722 = arg366_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_569: "f32[56]" = torch.ops.aten.add.Tensor(arg368_1, 1e-05);  arg368_1 = None
        sqrt_222: "f32[56]" = torch.ops.aten.sqrt.default(add_569);  add_569 = None
        reciprocal_222: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_222);  sqrt_222 = None
        mul_666: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_222, 1);  reciprocal_222 = None
        unsqueeze_1776: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg367_1, -1);  arg367_1 = None
        unsqueeze_1777: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1776, -1);  unsqueeze_1776 = None
        unsqueeze_1778: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_666, -1);  mul_666 = None
        unsqueeze_1779: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1778, -1);  unsqueeze_1778 = None
        sub_222: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_222, unsqueeze_1777);  convolution_222 = unsqueeze_1777 = None
        mul_667: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_1779);  sub_222 = unsqueeze_1779 = None
        unsqueeze_1780: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg369_1, -1);  arg369_1 = None
        unsqueeze_1781: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1780, -1);  unsqueeze_1780 = None
        mul_668: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_667, unsqueeze_1781);  mul_667 = unsqueeze_1781 = None
        unsqueeze_1782: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg370_1, -1);  arg370_1 = None
        unsqueeze_1783: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1782, -1);  unsqueeze_1782 = None
        add_570: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_668, unsqueeze_1783);  mul_668 = unsqueeze_1783 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_216: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_570);  add_570 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:99 in forward, code: spo.append(self.pool(spx[-1]))
        split_215 = torch.ops.aten.split.Tensor(relu_209, 56, 1);  relu_209 = None
        getitem_1731: "f32[8, 56, 28, 28]" = split_215[7];  split_215 = None
        avg_pool2d_6: "f32[8, 56, 14, 14]" = torch.ops.aten.avg_pool2d.default(getitem_1731, [3, 3], [2, 2], [1, 1]);  getitem_1731 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_23: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([relu_210, relu_211, relu_212, relu_213, relu_214, relu_215, relu_216, avg_pool2d_6], 1);  relu_210 = relu_211 = relu_212 = relu_213 = relu_214 = relu_215 = relu_216 = avg_pool2d_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_223: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_23, arg371_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_23 = arg371_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_571: "f32[1024]" = torch.ops.aten.add.Tensor(arg373_1, 1e-05);  arg373_1 = None
        sqrt_223: "f32[1024]" = torch.ops.aten.sqrt.default(add_571);  add_571 = None
        reciprocal_223: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_223);  sqrt_223 = None
        mul_669: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_223, 1);  reciprocal_223 = None
        unsqueeze_1784: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg372_1, -1);  arg372_1 = None
        unsqueeze_1785: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1784, -1);  unsqueeze_1784 = None
        unsqueeze_1786: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_669, -1);  mul_669 = None
        unsqueeze_1787: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1786, -1);  unsqueeze_1786 = None
        sub_223: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_223, unsqueeze_1785);  convolution_223 = unsqueeze_1785 = None
        mul_670: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_1787);  sub_223 = unsqueeze_1787 = None
        unsqueeze_1788: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
        unsqueeze_1789: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1788, -1);  unsqueeze_1788 = None
        mul_671: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_670, unsqueeze_1789);  mul_670 = unsqueeze_1789 = None
        unsqueeze_1790: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg375_1, -1);  arg375_1 = None
        unsqueeze_1791: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1790, -1);  unsqueeze_1790 = None
        add_572: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_671, unsqueeze_1791);  mul_671 = unsqueeze_1791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:111 in forward, code: shortcut = self.downsample(x)
        convolution_224: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_208, arg376_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_208 = arg376_1 = None
        add_573: "f32[1024]" = torch.ops.aten.add.Tensor(arg378_1, 1e-05);  arg378_1 = None
        sqrt_224: "f32[1024]" = torch.ops.aten.sqrt.default(add_573);  add_573 = None
        reciprocal_224: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_224);  sqrt_224 = None
        mul_672: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_224, 1);  reciprocal_224 = None
        unsqueeze_1792: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg377_1, -1);  arg377_1 = None
        unsqueeze_1793: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1792, -1);  unsqueeze_1792 = None
        unsqueeze_1794: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_672, -1);  mul_672 = None
        unsqueeze_1795: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1794, -1);  unsqueeze_1794 = None
        sub_224: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_224, unsqueeze_1793);  convolution_224 = unsqueeze_1793 = None
        mul_673: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_1795);  sub_224 = unsqueeze_1795 = None
        unsqueeze_1796: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_1797: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1796, -1);  unsqueeze_1796 = None
        mul_674: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_673, unsqueeze_1797);  mul_673 = unsqueeze_1797 = None
        unsqueeze_1798: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
        unsqueeze_1799: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1798, -1);  unsqueeze_1798 = None
        add_574: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_674, unsqueeze_1799);  mul_674 = unsqueeze_1799 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_575: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_572, add_574);  add_572 = add_574 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_217: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_575);  add_575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_225: "f32[8, 448, 14, 14]" = torch.ops.aten.convolution.default(relu_217, arg381_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg381_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_576: "f32[448]" = torch.ops.aten.add.Tensor(arg383_1, 1e-05);  arg383_1 = None
        sqrt_225: "f32[448]" = torch.ops.aten.sqrt.default(add_576);  add_576 = None
        reciprocal_225: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_225);  sqrt_225 = None
        mul_675: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_225, 1);  reciprocal_225 = None
        unsqueeze_1800: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg382_1, -1);  arg382_1 = None
        unsqueeze_1801: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1800, -1);  unsqueeze_1800 = None
        unsqueeze_1802: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_675, -1);  mul_675 = None
        unsqueeze_1803: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1802, -1);  unsqueeze_1802 = None
        sub_225: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_225, unsqueeze_1801);  convolution_225 = unsqueeze_1801 = None
        mul_676: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_1803);  sub_225 = unsqueeze_1803 = None
        unsqueeze_1804: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg384_1, -1);  arg384_1 = None
        unsqueeze_1805: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1804, -1);  unsqueeze_1804 = None
        mul_677: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(mul_676, unsqueeze_1805);  mul_676 = unsqueeze_1805 = None
        unsqueeze_1806: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
        unsqueeze_1807: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1806, -1);  unsqueeze_1806 = None
        add_577: "f32[8, 448, 14, 14]" = torch.ops.aten.add.Tensor(mul_677, unsqueeze_1807);  mul_677 = unsqueeze_1807 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_218: "f32[8, 448, 14, 14]" = torch.ops.aten.relu.default(add_577);  add_577 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_217 = torch.ops.aten.split.Tensor(relu_218, 56, 1)
        getitem_1740: "f32[8, 56, 14, 14]" = split_217[0];  split_217 = None
        convolution_226: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_1740, arg386_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1740 = arg386_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_578: "f32[56]" = torch.ops.aten.add.Tensor(arg388_1, 1e-05);  arg388_1 = None
        sqrt_226: "f32[56]" = torch.ops.aten.sqrt.default(add_578);  add_578 = None
        reciprocal_226: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_226);  sqrt_226 = None
        mul_678: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_226, 1);  reciprocal_226 = None
        unsqueeze_1808: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg387_1, -1);  arg387_1 = None
        unsqueeze_1809: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1808, -1);  unsqueeze_1808 = None
        unsqueeze_1810: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_678, -1);  mul_678 = None
        unsqueeze_1811: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1810, -1);  unsqueeze_1810 = None
        sub_226: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_226, unsqueeze_1809);  convolution_226 = unsqueeze_1809 = None
        mul_679: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_1811);  sub_226 = unsqueeze_1811 = None
        unsqueeze_1812: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg389_1, -1);  arg389_1 = None
        unsqueeze_1813: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1812, -1);  unsqueeze_1812 = None
        mul_680: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_679, unsqueeze_1813);  mul_679 = unsqueeze_1813 = None
        unsqueeze_1814: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg390_1, -1);  arg390_1 = None
        unsqueeze_1815: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1814, -1);  unsqueeze_1814 = None
        add_579: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_680, unsqueeze_1815);  mul_680 = unsqueeze_1815 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_219: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_579);  add_579 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_218 = torch.ops.aten.split.Tensor(relu_218, 56, 1)
        getitem_1749: "f32[8, 56, 14, 14]" = split_218[1];  split_218 = None
        add_580: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_219, getitem_1749);  getitem_1749 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_227: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_580, arg391_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_580 = arg391_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_581: "f32[56]" = torch.ops.aten.add.Tensor(arg393_1, 1e-05);  arg393_1 = None
        sqrt_227: "f32[56]" = torch.ops.aten.sqrt.default(add_581);  add_581 = None
        reciprocal_227: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_227);  sqrt_227 = None
        mul_681: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_227, 1);  reciprocal_227 = None
        unsqueeze_1816: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg392_1, -1);  arg392_1 = None
        unsqueeze_1817: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1816, -1);  unsqueeze_1816 = None
        unsqueeze_1818: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_681, -1);  mul_681 = None
        unsqueeze_1819: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1818, -1);  unsqueeze_1818 = None
        sub_227: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_227, unsqueeze_1817);  convolution_227 = unsqueeze_1817 = None
        mul_682: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_1819);  sub_227 = unsqueeze_1819 = None
        unsqueeze_1820: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
        unsqueeze_1821: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1820, -1);  unsqueeze_1820 = None
        mul_683: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_682, unsqueeze_1821);  mul_682 = unsqueeze_1821 = None
        unsqueeze_1822: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
        unsqueeze_1823: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1822, -1);  unsqueeze_1822 = None
        add_582: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_683, unsqueeze_1823);  mul_683 = unsqueeze_1823 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_220: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_582);  add_582 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_219 = torch.ops.aten.split.Tensor(relu_218, 56, 1)
        getitem_1758: "f32[8, 56, 14, 14]" = split_219[2];  split_219 = None
        add_583: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_220, getitem_1758);  getitem_1758 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_228: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_583, arg396_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_583 = arg396_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_584: "f32[56]" = torch.ops.aten.add.Tensor(arg398_1, 1e-05);  arg398_1 = None
        sqrt_228: "f32[56]" = torch.ops.aten.sqrt.default(add_584);  add_584 = None
        reciprocal_228: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_228);  sqrt_228 = None
        mul_684: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_228, 1);  reciprocal_228 = None
        unsqueeze_1824: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
        unsqueeze_1825: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1824, -1);  unsqueeze_1824 = None
        unsqueeze_1826: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_684, -1);  mul_684 = None
        unsqueeze_1827: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1826, -1);  unsqueeze_1826 = None
        sub_228: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_228, unsqueeze_1825);  convolution_228 = unsqueeze_1825 = None
        mul_685: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_1827);  sub_228 = unsqueeze_1827 = None
        unsqueeze_1828: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg399_1, -1);  arg399_1 = None
        unsqueeze_1829: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1828, -1);  unsqueeze_1828 = None
        mul_686: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_685, unsqueeze_1829);  mul_685 = unsqueeze_1829 = None
        unsqueeze_1830: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg400_1, -1);  arg400_1 = None
        unsqueeze_1831: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1830, -1);  unsqueeze_1830 = None
        add_585: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_686, unsqueeze_1831);  mul_686 = unsqueeze_1831 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_221: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_585);  add_585 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_220 = torch.ops.aten.split.Tensor(relu_218, 56, 1)
        getitem_1767: "f32[8, 56, 14, 14]" = split_220[3];  split_220 = None
        add_586: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_221, getitem_1767);  getitem_1767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_229: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_586, arg401_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_586 = arg401_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_587: "f32[56]" = torch.ops.aten.add.Tensor(arg403_1, 1e-05);  arg403_1 = None
        sqrt_229: "f32[56]" = torch.ops.aten.sqrt.default(add_587);  add_587 = None
        reciprocal_229: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_229);  sqrt_229 = None
        mul_687: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_229, 1);  reciprocal_229 = None
        unsqueeze_1832: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg402_1, -1);  arg402_1 = None
        unsqueeze_1833: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1832, -1);  unsqueeze_1832 = None
        unsqueeze_1834: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_687, -1);  mul_687 = None
        unsqueeze_1835: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1834, -1);  unsqueeze_1834 = None
        sub_229: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_229, unsqueeze_1833);  convolution_229 = unsqueeze_1833 = None
        mul_688: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_1835);  sub_229 = unsqueeze_1835 = None
        unsqueeze_1836: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
        unsqueeze_1837: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1836, -1);  unsqueeze_1836 = None
        mul_689: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_688, unsqueeze_1837);  mul_688 = unsqueeze_1837 = None
        unsqueeze_1838: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg405_1, -1);  arg405_1 = None
        unsqueeze_1839: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1838, -1);  unsqueeze_1838 = None
        add_588: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_689, unsqueeze_1839);  mul_689 = unsqueeze_1839 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_222: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_588);  add_588 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_221 = torch.ops.aten.split.Tensor(relu_218, 56, 1)
        getitem_1776: "f32[8, 56, 14, 14]" = split_221[4];  split_221 = None
        add_589: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_222, getitem_1776);  getitem_1776 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_230: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_589, arg406_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_589 = arg406_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_590: "f32[56]" = torch.ops.aten.add.Tensor(arg408_1, 1e-05);  arg408_1 = None
        sqrt_230: "f32[56]" = torch.ops.aten.sqrt.default(add_590);  add_590 = None
        reciprocal_230: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_230);  sqrt_230 = None
        mul_690: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_230, 1);  reciprocal_230 = None
        unsqueeze_1840: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg407_1, -1);  arg407_1 = None
        unsqueeze_1841: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1840, -1);  unsqueeze_1840 = None
        unsqueeze_1842: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_690, -1);  mul_690 = None
        unsqueeze_1843: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1842, -1);  unsqueeze_1842 = None
        sub_230: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_230, unsqueeze_1841);  convolution_230 = unsqueeze_1841 = None
        mul_691: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_1843);  sub_230 = unsqueeze_1843 = None
        unsqueeze_1844: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
        unsqueeze_1845: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1844, -1);  unsqueeze_1844 = None
        mul_692: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_691, unsqueeze_1845);  mul_691 = unsqueeze_1845 = None
        unsqueeze_1846: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg410_1, -1);  arg410_1 = None
        unsqueeze_1847: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1846, -1);  unsqueeze_1846 = None
        add_591: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_692, unsqueeze_1847);  mul_692 = unsqueeze_1847 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_223: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_591);  add_591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_222 = torch.ops.aten.split.Tensor(relu_218, 56, 1)
        getitem_1785: "f32[8, 56, 14, 14]" = split_222[5];  split_222 = None
        add_592: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_223, getitem_1785);  getitem_1785 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_231: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_592, arg411_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_592 = arg411_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_593: "f32[56]" = torch.ops.aten.add.Tensor(arg413_1, 1e-05);  arg413_1 = None
        sqrt_231: "f32[56]" = torch.ops.aten.sqrt.default(add_593);  add_593 = None
        reciprocal_231: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_231);  sqrt_231 = None
        mul_693: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_231, 1);  reciprocal_231 = None
        unsqueeze_1848: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg412_1, -1);  arg412_1 = None
        unsqueeze_1849: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1848, -1);  unsqueeze_1848 = None
        unsqueeze_1850: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_693, -1);  mul_693 = None
        unsqueeze_1851: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1850, -1);  unsqueeze_1850 = None
        sub_231: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_231, unsqueeze_1849);  convolution_231 = unsqueeze_1849 = None
        mul_694: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_1851);  sub_231 = unsqueeze_1851 = None
        unsqueeze_1852: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_1853: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1852, -1);  unsqueeze_1852 = None
        mul_695: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_694, unsqueeze_1853);  mul_694 = unsqueeze_1853 = None
        unsqueeze_1854: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg415_1, -1);  arg415_1 = None
        unsqueeze_1855: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1854, -1);  unsqueeze_1854 = None
        add_594: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_695, unsqueeze_1855);  mul_695 = unsqueeze_1855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_224: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_594);  add_594 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_223 = torch.ops.aten.split.Tensor(relu_218, 56, 1)
        getitem_1794: "f32[8, 56, 14, 14]" = split_223[6];  split_223 = None
        add_595: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_224, getitem_1794);  getitem_1794 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_232: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_595, arg416_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_595 = arg416_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_596: "f32[56]" = torch.ops.aten.add.Tensor(arg418_1, 1e-05);  arg418_1 = None
        sqrt_232: "f32[56]" = torch.ops.aten.sqrt.default(add_596);  add_596 = None
        reciprocal_232: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_232);  sqrt_232 = None
        mul_696: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_232, 1);  reciprocal_232 = None
        unsqueeze_1856: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg417_1, -1);  arg417_1 = None
        unsqueeze_1857: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1856, -1);  unsqueeze_1856 = None
        unsqueeze_1858: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_696, -1);  mul_696 = None
        unsqueeze_1859: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1858, -1);  unsqueeze_1858 = None
        sub_232: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_232, unsqueeze_1857);  convolution_232 = unsqueeze_1857 = None
        mul_697: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_1859);  sub_232 = unsqueeze_1859 = None
        unsqueeze_1860: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg419_1, -1);  arg419_1 = None
        unsqueeze_1861: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1860, -1);  unsqueeze_1860 = None
        mul_698: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_697, unsqueeze_1861);  mul_697 = unsqueeze_1861 = None
        unsqueeze_1862: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg420_1, -1);  arg420_1 = None
        unsqueeze_1863: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1862, -1);  unsqueeze_1862 = None
        add_597: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_698, unsqueeze_1863);  mul_698 = unsqueeze_1863 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_225: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_597);  add_597 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_224 = torch.ops.aten.split.Tensor(relu_218, 56, 1);  relu_218 = None
        getitem_1803: "f32[8, 56, 14, 14]" = split_224[7];  split_224 = None
        cat_24: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([relu_219, relu_220, relu_221, relu_222, relu_223, relu_224, relu_225, getitem_1803], 1);  relu_219 = relu_220 = relu_221 = relu_222 = relu_223 = relu_224 = relu_225 = getitem_1803 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_233: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_24, arg421_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_24 = arg421_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_598: "f32[1024]" = torch.ops.aten.add.Tensor(arg423_1, 1e-05);  arg423_1 = None
        sqrt_233: "f32[1024]" = torch.ops.aten.sqrt.default(add_598);  add_598 = None
        reciprocal_233: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_233);  sqrt_233 = None
        mul_699: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_233, 1);  reciprocal_233 = None
        unsqueeze_1864: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg422_1, -1);  arg422_1 = None
        unsqueeze_1865: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1864, -1);  unsqueeze_1864 = None
        unsqueeze_1866: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_699, -1);  mul_699 = None
        unsqueeze_1867: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1866, -1);  unsqueeze_1866 = None
        sub_233: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_233, unsqueeze_1865);  convolution_233 = unsqueeze_1865 = None
        mul_700: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_1867);  sub_233 = unsqueeze_1867 = None
        unsqueeze_1868: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg424_1, -1);  arg424_1 = None
        unsqueeze_1869: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1868, -1);  unsqueeze_1868 = None
        mul_701: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_700, unsqueeze_1869);  mul_700 = unsqueeze_1869 = None
        unsqueeze_1870: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg425_1, -1);  arg425_1 = None
        unsqueeze_1871: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1870, -1);  unsqueeze_1870 = None
        add_599: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_701, unsqueeze_1871);  mul_701 = unsqueeze_1871 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_600: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_599, relu_217);  add_599 = relu_217 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_226: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_600);  add_600 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_234: "f32[8, 448, 14, 14]" = torch.ops.aten.convolution.default(relu_226, arg426_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg426_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_601: "f32[448]" = torch.ops.aten.add.Tensor(arg428_1, 1e-05);  arg428_1 = None
        sqrt_234: "f32[448]" = torch.ops.aten.sqrt.default(add_601);  add_601 = None
        reciprocal_234: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_234);  sqrt_234 = None
        mul_702: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_234, 1);  reciprocal_234 = None
        unsqueeze_1872: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg427_1, -1);  arg427_1 = None
        unsqueeze_1873: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1872, -1);  unsqueeze_1872 = None
        unsqueeze_1874: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_702, -1);  mul_702 = None
        unsqueeze_1875: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1874, -1);  unsqueeze_1874 = None
        sub_234: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_234, unsqueeze_1873);  convolution_234 = unsqueeze_1873 = None
        mul_703: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_1875);  sub_234 = unsqueeze_1875 = None
        unsqueeze_1876: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg429_1, -1);  arg429_1 = None
        unsqueeze_1877: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1876, -1);  unsqueeze_1876 = None
        mul_704: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(mul_703, unsqueeze_1877);  mul_703 = unsqueeze_1877 = None
        unsqueeze_1878: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
        unsqueeze_1879: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1878, -1);  unsqueeze_1878 = None
        add_602: "f32[8, 448, 14, 14]" = torch.ops.aten.add.Tensor(mul_704, unsqueeze_1879);  mul_704 = unsqueeze_1879 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_227: "f32[8, 448, 14, 14]" = torch.ops.aten.relu.default(add_602);  add_602 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_226 = torch.ops.aten.split.Tensor(relu_227, 56, 1)
        getitem_1812: "f32[8, 56, 14, 14]" = split_226[0];  split_226 = None
        convolution_235: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_1812, arg431_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1812 = arg431_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_603: "f32[56]" = torch.ops.aten.add.Tensor(arg433_1, 1e-05);  arg433_1 = None
        sqrt_235: "f32[56]" = torch.ops.aten.sqrt.default(add_603);  add_603 = None
        reciprocal_235: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_235);  sqrt_235 = None
        mul_705: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_235, 1);  reciprocal_235 = None
        unsqueeze_1880: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg432_1, -1);  arg432_1 = None
        unsqueeze_1881: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1880, -1);  unsqueeze_1880 = None
        unsqueeze_1882: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_705, -1);  mul_705 = None
        unsqueeze_1883: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1882, -1);  unsqueeze_1882 = None
        sub_235: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_235, unsqueeze_1881);  convolution_235 = unsqueeze_1881 = None
        mul_706: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_1883);  sub_235 = unsqueeze_1883 = None
        unsqueeze_1884: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg434_1, -1);  arg434_1 = None
        unsqueeze_1885: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1884, -1);  unsqueeze_1884 = None
        mul_707: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_706, unsqueeze_1885);  mul_706 = unsqueeze_1885 = None
        unsqueeze_1886: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg435_1, -1);  arg435_1 = None
        unsqueeze_1887: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1886, -1);  unsqueeze_1886 = None
        add_604: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_707, unsqueeze_1887);  mul_707 = unsqueeze_1887 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_228: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_604);  add_604 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_227 = torch.ops.aten.split.Tensor(relu_227, 56, 1)
        getitem_1821: "f32[8, 56, 14, 14]" = split_227[1];  split_227 = None
        add_605: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_228, getitem_1821);  getitem_1821 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_236: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_605, arg436_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_605 = arg436_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_606: "f32[56]" = torch.ops.aten.add.Tensor(arg438_1, 1e-05);  arg438_1 = None
        sqrt_236: "f32[56]" = torch.ops.aten.sqrt.default(add_606);  add_606 = None
        reciprocal_236: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_236);  sqrt_236 = None
        mul_708: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_236, 1);  reciprocal_236 = None
        unsqueeze_1888: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg437_1, -1);  arg437_1 = None
        unsqueeze_1889: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1888, -1);  unsqueeze_1888 = None
        unsqueeze_1890: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_708, -1);  mul_708 = None
        unsqueeze_1891: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1890, -1);  unsqueeze_1890 = None
        sub_236: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_236, unsqueeze_1889);  convolution_236 = unsqueeze_1889 = None
        mul_709: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_1891);  sub_236 = unsqueeze_1891 = None
        unsqueeze_1892: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg439_1, -1);  arg439_1 = None
        unsqueeze_1893: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1892, -1);  unsqueeze_1892 = None
        mul_710: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_709, unsqueeze_1893);  mul_709 = unsqueeze_1893 = None
        unsqueeze_1894: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg440_1, -1);  arg440_1 = None
        unsqueeze_1895: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1894, -1);  unsqueeze_1894 = None
        add_607: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_710, unsqueeze_1895);  mul_710 = unsqueeze_1895 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_229: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_607);  add_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_228 = torch.ops.aten.split.Tensor(relu_227, 56, 1)
        getitem_1830: "f32[8, 56, 14, 14]" = split_228[2];  split_228 = None
        add_608: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_229, getitem_1830);  getitem_1830 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_237: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_608, arg441_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_608 = arg441_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_609: "f32[56]" = torch.ops.aten.add.Tensor(arg443_1, 1e-05);  arg443_1 = None
        sqrt_237: "f32[56]" = torch.ops.aten.sqrt.default(add_609);  add_609 = None
        reciprocal_237: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_237);  sqrt_237 = None
        mul_711: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_237, 1);  reciprocal_237 = None
        unsqueeze_1896: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg442_1, -1);  arg442_1 = None
        unsqueeze_1897: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1896, -1);  unsqueeze_1896 = None
        unsqueeze_1898: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_711, -1);  mul_711 = None
        unsqueeze_1899: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1898, -1);  unsqueeze_1898 = None
        sub_237: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_237, unsqueeze_1897);  convolution_237 = unsqueeze_1897 = None
        mul_712: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_1899);  sub_237 = unsqueeze_1899 = None
        unsqueeze_1900: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg444_1, -1);  arg444_1 = None
        unsqueeze_1901: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1900, -1);  unsqueeze_1900 = None
        mul_713: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_712, unsqueeze_1901);  mul_712 = unsqueeze_1901 = None
        unsqueeze_1902: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg445_1, -1);  arg445_1 = None
        unsqueeze_1903: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1902, -1);  unsqueeze_1902 = None
        add_610: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_713, unsqueeze_1903);  mul_713 = unsqueeze_1903 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_230: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_610);  add_610 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_229 = torch.ops.aten.split.Tensor(relu_227, 56, 1)
        getitem_1839: "f32[8, 56, 14, 14]" = split_229[3];  split_229 = None
        add_611: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_230, getitem_1839);  getitem_1839 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_238: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_611, arg446_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_611 = arg446_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_612: "f32[56]" = torch.ops.aten.add.Tensor(arg448_1, 1e-05);  arg448_1 = None
        sqrt_238: "f32[56]" = torch.ops.aten.sqrt.default(add_612);  add_612 = None
        reciprocal_238: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_238);  sqrt_238 = None
        mul_714: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_238, 1);  reciprocal_238 = None
        unsqueeze_1904: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg447_1, -1);  arg447_1 = None
        unsqueeze_1905: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1904, -1);  unsqueeze_1904 = None
        unsqueeze_1906: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_714, -1);  mul_714 = None
        unsqueeze_1907: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1906, -1);  unsqueeze_1906 = None
        sub_238: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_238, unsqueeze_1905);  convolution_238 = unsqueeze_1905 = None
        mul_715: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_1907);  sub_238 = unsqueeze_1907 = None
        unsqueeze_1908: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
        unsqueeze_1909: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1908, -1);  unsqueeze_1908 = None
        mul_716: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_715, unsqueeze_1909);  mul_715 = unsqueeze_1909 = None
        unsqueeze_1910: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg450_1, -1);  arg450_1 = None
        unsqueeze_1911: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1910, -1);  unsqueeze_1910 = None
        add_613: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_716, unsqueeze_1911);  mul_716 = unsqueeze_1911 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_231: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_613);  add_613 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_230 = torch.ops.aten.split.Tensor(relu_227, 56, 1)
        getitem_1848: "f32[8, 56, 14, 14]" = split_230[4];  split_230 = None
        add_614: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_231, getitem_1848);  getitem_1848 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_239: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_614, arg451_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_614 = arg451_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_615: "f32[56]" = torch.ops.aten.add.Tensor(arg453_1, 1e-05);  arg453_1 = None
        sqrt_239: "f32[56]" = torch.ops.aten.sqrt.default(add_615);  add_615 = None
        reciprocal_239: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_239);  sqrt_239 = None
        mul_717: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_239, 1);  reciprocal_239 = None
        unsqueeze_1912: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg452_1, -1);  arg452_1 = None
        unsqueeze_1913: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1912, -1);  unsqueeze_1912 = None
        unsqueeze_1914: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_717, -1);  mul_717 = None
        unsqueeze_1915: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1914, -1);  unsqueeze_1914 = None
        sub_239: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_239, unsqueeze_1913);  convolution_239 = unsqueeze_1913 = None
        mul_718: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_1915);  sub_239 = unsqueeze_1915 = None
        unsqueeze_1916: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
        unsqueeze_1917: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1916, -1);  unsqueeze_1916 = None
        mul_719: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_718, unsqueeze_1917);  mul_718 = unsqueeze_1917 = None
        unsqueeze_1918: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg455_1, -1);  arg455_1 = None
        unsqueeze_1919: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1918, -1);  unsqueeze_1918 = None
        add_616: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_719, unsqueeze_1919);  mul_719 = unsqueeze_1919 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_232: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_616);  add_616 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_231 = torch.ops.aten.split.Tensor(relu_227, 56, 1)
        getitem_1857: "f32[8, 56, 14, 14]" = split_231[5];  split_231 = None
        add_617: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_232, getitem_1857);  getitem_1857 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_240: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_617, arg456_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_617 = arg456_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_618: "f32[56]" = torch.ops.aten.add.Tensor(arg458_1, 1e-05);  arg458_1 = None
        sqrt_240: "f32[56]" = torch.ops.aten.sqrt.default(add_618);  add_618 = None
        reciprocal_240: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_240);  sqrt_240 = None
        mul_720: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_240, 1);  reciprocal_240 = None
        unsqueeze_1920: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg457_1, -1);  arg457_1 = None
        unsqueeze_1921: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1920, -1);  unsqueeze_1920 = None
        unsqueeze_1922: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_720, -1);  mul_720 = None
        unsqueeze_1923: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1922, -1);  unsqueeze_1922 = None
        sub_240: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_240, unsqueeze_1921);  convolution_240 = unsqueeze_1921 = None
        mul_721: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_1923);  sub_240 = unsqueeze_1923 = None
        unsqueeze_1924: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg459_1, -1);  arg459_1 = None
        unsqueeze_1925: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1924, -1);  unsqueeze_1924 = None
        mul_722: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_721, unsqueeze_1925);  mul_721 = unsqueeze_1925 = None
        unsqueeze_1926: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg460_1, -1);  arg460_1 = None
        unsqueeze_1927: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1926, -1);  unsqueeze_1926 = None
        add_619: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_722, unsqueeze_1927);  mul_722 = unsqueeze_1927 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_233: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_619);  add_619 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_232 = torch.ops.aten.split.Tensor(relu_227, 56, 1)
        getitem_1866: "f32[8, 56, 14, 14]" = split_232[6];  split_232 = None
        add_620: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_233, getitem_1866);  getitem_1866 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_241: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_620, arg461_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_620 = arg461_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_621: "f32[56]" = torch.ops.aten.add.Tensor(arg463_1, 1e-05);  arg463_1 = None
        sqrt_241: "f32[56]" = torch.ops.aten.sqrt.default(add_621);  add_621 = None
        reciprocal_241: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_241);  sqrt_241 = None
        mul_723: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_241, 1);  reciprocal_241 = None
        unsqueeze_1928: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg462_1, -1);  arg462_1 = None
        unsqueeze_1929: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1928, -1);  unsqueeze_1928 = None
        unsqueeze_1930: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_723, -1);  mul_723 = None
        unsqueeze_1931: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1930, -1);  unsqueeze_1930 = None
        sub_241: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_241, unsqueeze_1929);  convolution_241 = unsqueeze_1929 = None
        mul_724: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_1931);  sub_241 = unsqueeze_1931 = None
        unsqueeze_1932: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
        unsqueeze_1933: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1932, -1);  unsqueeze_1932 = None
        mul_725: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_724, unsqueeze_1933);  mul_724 = unsqueeze_1933 = None
        unsqueeze_1934: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg465_1, -1);  arg465_1 = None
        unsqueeze_1935: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1934, -1);  unsqueeze_1934 = None
        add_622: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_725, unsqueeze_1935);  mul_725 = unsqueeze_1935 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_234: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_622);  add_622 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_233 = torch.ops.aten.split.Tensor(relu_227, 56, 1);  relu_227 = None
        getitem_1875: "f32[8, 56, 14, 14]" = split_233[7];  split_233 = None
        cat_25: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([relu_228, relu_229, relu_230, relu_231, relu_232, relu_233, relu_234, getitem_1875], 1);  relu_228 = relu_229 = relu_230 = relu_231 = relu_232 = relu_233 = relu_234 = getitem_1875 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_242: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_25, arg466_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_25 = arg466_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_623: "f32[1024]" = torch.ops.aten.add.Tensor(arg468_1, 1e-05);  arg468_1 = None
        sqrt_242: "f32[1024]" = torch.ops.aten.sqrt.default(add_623);  add_623 = None
        reciprocal_242: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_242);  sqrt_242 = None
        mul_726: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_242, 1);  reciprocal_242 = None
        unsqueeze_1936: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg467_1, -1);  arg467_1 = None
        unsqueeze_1937: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1936, -1);  unsqueeze_1936 = None
        unsqueeze_1938: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_726, -1);  mul_726 = None
        unsqueeze_1939: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1938, -1);  unsqueeze_1938 = None
        sub_242: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_242, unsqueeze_1937);  convolution_242 = unsqueeze_1937 = None
        mul_727: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_1939);  sub_242 = unsqueeze_1939 = None
        unsqueeze_1940: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
        unsqueeze_1941: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1940, -1);  unsqueeze_1940 = None
        mul_728: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_727, unsqueeze_1941);  mul_727 = unsqueeze_1941 = None
        unsqueeze_1942: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg470_1, -1);  arg470_1 = None
        unsqueeze_1943: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1942, -1);  unsqueeze_1942 = None
        add_624: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_728, unsqueeze_1943);  mul_728 = unsqueeze_1943 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_625: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_624, relu_226);  add_624 = relu_226 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_235: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_625);  add_625 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_243: "f32[8, 448, 14, 14]" = torch.ops.aten.convolution.default(relu_235, arg471_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg471_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_626: "f32[448]" = torch.ops.aten.add.Tensor(arg473_1, 1e-05);  arg473_1 = None
        sqrt_243: "f32[448]" = torch.ops.aten.sqrt.default(add_626);  add_626 = None
        reciprocal_243: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_243);  sqrt_243 = None
        mul_729: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_243, 1);  reciprocal_243 = None
        unsqueeze_1944: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg472_1, -1);  arg472_1 = None
        unsqueeze_1945: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1944, -1);  unsqueeze_1944 = None
        unsqueeze_1946: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_729, -1);  mul_729 = None
        unsqueeze_1947: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1946, -1);  unsqueeze_1946 = None
        sub_243: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_243, unsqueeze_1945);  convolution_243 = unsqueeze_1945 = None
        mul_730: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_1947);  sub_243 = unsqueeze_1947 = None
        unsqueeze_1948: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg474_1, -1);  arg474_1 = None
        unsqueeze_1949: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1948, -1);  unsqueeze_1948 = None
        mul_731: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(mul_730, unsqueeze_1949);  mul_730 = unsqueeze_1949 = None
        unsqueeze_1950: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg475_1, -1);  arg475_1 = None
        unsqueeze_1951: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1950, -1);  unsqueeze_1950 = None
        add_627: "f32[8, 448, 14, 14]" = torch.ops.aten.add.Tensor(mul_731, unsqueeze_1951);  mul_731 = unsqueeze_1951 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_236: "f32[8, 448, 14, 14]" = torch.ops.aten.relu.default(add_627);  add_627 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_235 = torch.ops.aten.split.Tensor(relu_236, 56, 1)
        getitem_1884: "f32[8, 56, 14, 14]" = split_235[0];  split_235 = None
        convolution_244: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_1884, arg476_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1884 = arg476_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_628: "f32[56]" = torch.ops.aten.add.Tensor(arg478_1, 1e-05);  arg478_1 = None
        sqrt_244: "f32[56]" = torch.ops.aten.sqrt.default(add_628);  add_628 = None
        reciprocal_244: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_244);  sqrt_244 = None
        mul_732: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_244, 1);  reciprocal_244 = None
        unsqueeze_1952: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg477_1, -1);  arg477_1 = None
        unsqueeze_1953: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1952, -1);  unsqueeze_1952 = None
        unsqueeze_1954: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_732, -1);  mul_732 = None
        unsqueeze_1955: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1954, -1);  unsqueeze_1954 = None
        sub_244: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_244, unsqueeze_1953);  convolution_244 = unsqueeze_1953 = None
        mul_733: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_1955);  sub_244 = unsqueeze_1955 = None
        unsqueeze_1956: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg479_1, -1);  arg479_1 = None
        unsqueeze_1957: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1956, -1);  unsqueeze_1956 = None
        mul_734: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_733, unsqueeze_1957);  mul_733 = unsqueeze_1957 = None
        unsqueeze_1958: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg480_1, -1);  arg480_1 = None
        unsqueeze_1959: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1958, -1);  unsqueeze_1958 = None
        add_629: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_734, unsqueeze_1959);  mul_734 = unsqueeze_1959 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_237: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_629);  add_629 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_236 = torch.ops.aten.split.Tensor(relu_236, 56, 1)
        getitem_1893: "f32[8, 56, 14, 14]" = split_236[1];  split_236 = None
        add_630: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_237, getitem_1893);  getitem_1893 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_245: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_630, arg481_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_630 = arg481_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_631: "f32[56]" = torch.ops.aten.add.Tensor(arg483_1, 1e-05);  arg483_1 = None
        sqrt_245: "f32[56]" = torch.ops.aten.sqrt.default(add_631);  add_631 = None
        reciprocal_245: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_245);  sqrt_245 = None
        mul_735: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_245, 1);  reciprocal_245 = None
        unsqueeze_1960: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg482_1, -1);  arg482_1 = None
        unsqueeze_1961: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1960, -1);  unsqueeze_1960 = None
        unsqueeze_1962: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_735, -1);  mul_735 = None
        unsqueeze_1963: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1962, -1);  unsqueeze_1962 = None
        sub_245: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_245, unsqueeze_1961);  convolution_245 = unsqueeze_1961 = None
        mul_736: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_1963);  sub_245 = unsqueeze_1963 = None
        unsqueeze_1964: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg484_1, -1);  arg484_1 = None
        unsqueeze_1965: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1964, -1);  unsqueeze_1964 = None
        mul_737: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_736, unsqueeze_1965);  mul_736 = unsqueeze_1965 = None
        unsqueeze_1966: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg485_1, -1);  arg485_1 = None
        unsqueeze_1967: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1966, -1);  unsqueeze_1966 = None
        add_632: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_737, unsqueeze_1967);  mul_737 = unsqueeze_1967 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_238: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_632);  add_632 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_237 = torch.ops.aten.split.Tensor(relu_236, 56, 1)
        getitem_1902: "f32[8, 56, 14, 14]" = split_237[2];  split_237 = None
        add_633: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_238, getitem_1902);  getitem_1902 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_246: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_633, arg486_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_633 = arg486_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_634: "f32[56]" = torch.ops.aten.add.Tensor(arg488_1, 1e-05);  arg488_1 = None
        sqrt_246: "f32[56]" = torch.ops.aten.sqrt.default(add_634);  add_634 = None
        reciprocal_246: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_246);  sqrt_246 = None
        mul_738: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_246, 1);  reciprocal_246 = None
        unsqueeze_1968: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg487_1, -1);  arg487_1 = None
        unsqueeze_1969: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1968, -1);  unsqueeze_1968 = None
        unsqueeze_1970: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_738, -1);  mul_738 = None
        unsqueeze_1971: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1970, -1);  unsqueeze_1970 = None
        sub_246: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_246, unsqueeze_1969);  convolution_246 = unsqueeze_1969 = None
        mul_739: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_1971);  sub_246 = unsqueeze_1971 = None
        unsqueeze_1972: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg489_1, -1);  arg489_1 = None
        unsqueeze_1973: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1972, -1);  unsqueeze_1972 = None
        mul_740: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_739, unsqueeze_1973);  mul_739 = unsqueeze_1973 = None
        unsqueeze_1974: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg490_1, -1);  arg490_1 = None
        unsqueeze_1975: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1974, -1);  unsqueeze_1974 = None
        add_635: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_740, unsqueeze_1975);  mul_740 = unsqueeze_1975 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_239: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_635);  add_635 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_238 = torch.ops.aten.split.Tensor(relu_236, 56, 1)
        getitem_1911: "f32[8, 56, 14, 14]" = split_238[3];  split_238 = None
        add_636: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_239, getitem_1911);  getitem_1911 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_247: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_636, arg491_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_636 = arg491_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_637: "f32[56]" = torch.ops.aten.add.Tensor(arg493_1, 1e-05);  arg493_1 = None
        sqrt_247: "f32[56]" = torch.ops.aten.sqrt.default(add_637);  add_637 = None
        reciprocal_247: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_247);  sqrt_247 = None
        mul_741: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_247, 1);  reciprocal_247 = None
        unsqueeze_1976: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg492_1, -1);  arg492_1 = None
        unsqueeze_1977: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1976, -1);  unsqueeze_1976 = None
        unsqueeze_1978: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_741, -1);  mul_741 = None
        unsqueeze_1979: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1978, -1);  unsqueeze_1978 = None
        sub_247: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_247, unsqueeze_1977);  convolution_247 = unsqueeze_1977 = None
        mul_742: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_1979);  sub_247 = unsqueeze_1979 = None
        unsqueeze_1980: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg494_1, -1);  arg494_1 = None
        unsqueeze_1981: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1980, -1);  unsqueeze_1980 = None
        mul_743: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_742, unsqueeze_1981);  mul_742 = unsqueeze_1981 = None
        unsqueeze_1982: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg495_1, -1);  arg495_1 = None
        unsqueeze_1983: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1982, -1);  unsqueeze_1982 = None
        add_638: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_743, unsqueeze_1983);  mul_743 = unsqueeze_1983 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_240: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_638);  add_638 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_239 = torch.ops.aten.split.Tensor(relu_236, 56, 1)
        getitem_1920: "f32[8, 56, 14, 14]" = split_239[4];  split_239 = None
        add_639: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_240, getitem_1920);  getitem_1920 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_248: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_639, arg496_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_639 = arg496_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_640: "f32[56]" = torch.ops.aten.add.Tensor(arg498_1, 1e-05);  arg498_1 = None
        sqrt_248: "f32[56]" = torch.ops.aten.sqrt.default(add_640);  add_640 = None
        reciprocal_248: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_248);  sqrt_248 = None
        mul_744: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_248, 1);  reciprocal_248 = None
        unsqueeze_1984: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg497_1, -1);  arg497_1 = None
        unsqueeze_1985: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1984, -1);  unsqueeze_1984 = None
        unsqueeze_1986: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_744, -1);  mul_744 = None
        unsqueeze_1987: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1986, -1);  unsqueeze_1986 = None
        sub_248: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_248, unsqueeze_1985);  convolution_248 = unsqueeze_1985 = None
        mul_745: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_1987);  sub_248 = unsqueeze_1987 = None
        unsqueeze_1988: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg499_1, -1);  arg499_1 = None
        unsqueeze_1989: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1988, -1);  unsqueeze_1988 = None
        mul_746: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_745, unsqueeze_1989);  mul_745 = unsqueeze_1989 = None
        unsqueeze_1990: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg500_1, -1);  arg500_1 = None
        unsqueeze_1991: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1990, -1);  unsqueeze_1990 = None
        add_641: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_746, unsqueeze_1991);  mul_746 = unsqueeze_1991 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_241: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_641);  add_641 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_240 = torch.ops.aten.split.Tensor(relu_236, 56, 1)
        getitem_1929: "f32[8, 56, 14, 14]" = split_240[5];  split_240 = None
        add_642: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_241, getitem_1929);  getitem_1929 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_249: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_642, arg501_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_642 = arg501_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_643: "f32[56]" = torch.ops.aten.add.Tensor(arg503_1, 1e-05);  arg503_1 = None
        sqrt_249: "f32[56]" = torch.ops.aten.sqrt.default(add_643);  add_643 = None
        reciprocal_249: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_249);  sqrt_249 = None
        mul_747: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_249, 1);  reciprocal_249 = None
        unsqueeze_1992: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg502_1, -1);  arg502_1 = None
        unsqueeze_1993: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1992, -1);  unsqueeze_1992 = None
        unsqueeze_1994: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_747, -1);  mul_747 = None
        unsqueeze_1995: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1994, -1);  unsqueeze_1994 = None
        sub_249: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_249, unsqueeze_1993);  convolution_249 = unsqueeze_1993 = None
        mul_748: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_1995);  sub_249 = unsqueeze_1995 = None
        unsqueeze_1996: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg504_1, -1);  arg504_1 = None
        unsqueeze_1997: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1996, -1);  unsqueeze_1996 = None
        mul_749: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_748, unsqueeze_1997);  mul_748 = unsqueeze_1997 = None
        unsqueeze_1998: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg505_1, -1);  arg505_1 = None
        unsqueeze_1999: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1998, -1);  unsqueeze_1998 = None
        add_644: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_749, unsqueeze_1999);  mul_749 = unsqueeze_1999 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_242: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_644);  add_644 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_241 = torch.ops.aten.split.Tensor(relu_236, 56, 1)
        getitem_1938: "f32[8, 56, 14, 14]" = split_241[6];  split_241 = None
        add_645: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_242, getitem_1938);  getitem_1938 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_250: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_645, arg506_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_645 = arg506_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_646: "f32[56]" = torch.ops.aten.add.Tensor(arg508_1, 1e-05);  arg508_1 = None
        sqrt_250: "f32[56]" = torch.ops.aten.sqrt.default(add_646);  add_646 = None
        reciprocal_250: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_250);  sqrt_250 = None
        mul_750: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_250, 1);  reciprocal_250 = None
        unsqueeze_2000: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg507_1, -1);  arg507_1 = None
        unsqueeze_2001: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2000, -1);  unsqueeze_2000 = None
        unsqueeze_2002: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_750, -1);  mul_750 = None
        unsqueeze_2003: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2002, -1);  unsqueeze_2002 = None
        sub_250: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_250, unsqueeze_2001);  convolution_250 = unsqueeze_2001 = None
        mul_751: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_2003);  sub_250 = unsqueeze_2003 = None
        unsqueeze_2004: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg509_1, -1);  arg509_1 = None
        unsqueeze_2005: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2004, -1);  unsqueeze_2004 = None
        mul_752: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_751, unsqueeze_2005);  mul_751 = unsqueeze_2005 = None
        unsqueeze_2006: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg510_1, -1);  arg510_1 = None
        unsqueeze_2007: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2006, -1);  unsqueeze_2006 = None
        add_647: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_752, unsqueeze_2007);  mul_752 = unsqueeze_2007 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_243: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_647);  add_647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_242 = torch.ops.aten.split.Tensor(relu_236, 56, 1);  relu_236 = None
        getitem_1947: "f32[8, 56, 14, 14]" = split_242[7];  split_242 = None
        cat_26: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([relu_237, relu_238, relu_239, relu_240, relu_241, relu_242, relu_243, getitem_1947], 1);  relu_237 = relu_238 = relu_239 = relu_240 = relu_241 = relu_242 = relu_243 = getitem_1947 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_251: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_26, arg511_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_26 = arg511_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_648: "f32[1024]" = torch.ops.aten.add.Tensor(arg513_1, 1e-05);  arg513_1 = None
        sqrt_251: "f32[1024]" = torch.ops.aten.sqrt.default(add_648);  add_648 = None
        reciprocal_251: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_251);  sqrt_251 = None
        mul_753: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_251, 1);  reciprocal_251 = None
        unsqueeze_2008: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg512_1, -1);  arg512_1 = None
        unsqueeze_2009: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2008, -1);  unsqueeze_2008 = None
        unsqueeze_2010: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_753, -1);  mul_753 = None
        unsqueeze_2011: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2010, -1);  unsqueeze_2010 = None
        sub_251: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_251, unsqueeze_2009);  convolution_251 = unsqueeze_2009 = None
        mul_754: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_2011);  sub_251 = unsqueeze_2011 = None
        unsqueeze_2012: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg514_1, -1);  arg514_1 = None
        unsqueeze_2013: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2012, -1);  unsqueeze_2012 = None
        mul_755: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_754, unsqueeze_2013);  mul_754 = unsqueeze_2013 = None
        unsqueeze_2014: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg515_1, -1);  arg515_1 = None
        unsqueeze_2015: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2014, -1);  unsqueeze_2014 = None
        add_649: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_755, unsqueeze_2015);  mul_755 = unsqueeze_2015 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_650: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_649, relu_235);  add_649 = relu_235 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_244: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_650);  add_650 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_252: "f32[8, 448, 14, 14]" = torch.ops.aten.convolution.default(relu_244, arg516_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg516_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_651: "f32[448]" = torch.ops.aten.add.Tensor(arg518_1, 1e-05);  arg518_1 = None
        sqrt_252: "f32[448]" = torch.ops.aten.sqrt.default(add_651);  add_651 = None
        reciprocal_252: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_252);  sqrt_252 = None
        mul_756: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_252, 1);  reciprocal_252 = None
        unsqueeze_2016: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg517_1, -1);  arg517_1 = None
        unsqueeze_2017: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2016, -1);  unsqueeze_2016 = None
        unsqueeze_2018: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_756, -1);  mul_756 = None
        unsqueeze_2019: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2018, -1);  unsqueeze_2018 = None
        sub_252: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_252, unsqueeze_2017);  convolution_252 = unsqueeze_2017 = None
        mul_757: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_2019);  sub_252 = unsqueeze_2019 = None
        unsqueeze_2020: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg519_1, -1);  arg519_1 = None
        unsqueeze_2021: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2020, -1);  unsqueeze_2020 = None
        mul_758: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(mul_757, unsqueeze_2021);  mul_757 = unsqueeze_2021 = None
        unsqueeze_2022: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg520_1, -1);  arg520_1 = None
        unsqueeze_2023: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2022, -1);  unsqueeze_2022 = None
        add_652: "f32[8, 448, 14, 14]" = torch.ops.aten.add.Tensor(mul_758, unsqueeze_2023);  mul_758 = unsqueeze_2023 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_245: "f32[8, 448, 14, 14]" = torch.ops.aten.relu.default(add_652);  add_652 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_244 = torch.ops.aten.split.Tensor(relu_245, 56, 1)
        getitem_1956: "f32[8, 56, 14, 14]" = split_244[0];  split_244 = None
        convolution_253: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_1956, arg521_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_1956 = arg521_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_653: "f32[56]" = torch.ops.aten.add.Tensor(arg523_1, 1e-05);  arg523_1 = None
        sqrt_253: "f32[56]" = torch.ops.aten.sqrt.default(add_653);  add_653 = None
        reciprocal_253: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_253);  sqrt_253 = None
        mul_759: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_253, 1);  reciprocal_253 = None
        unsqueeze_2024: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg522_1, -1);  arg522_1 = None
        unsqueeze_2025: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2024, -1);  unsqueeze_2024 = None
        unsqueeze_2026: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_759, -1);  mul_759 = None
        unsqueeze_2027: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2026, -1);  unsqueeze_2026 = None
        sub_253: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_253, unsqueeze_2025);  convolution_253 = unsqueeze_2025 = None
        mul_760: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_2027);  sub_253 = unsqueeze_2027 = None
        unsqueeze_2028: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg524_1, -1);  arg524_1 = None
        unsqueeze_2029: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2028, -1);  unsqueeze_2028 = None
        mul_761: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_760, unsqueeze_2029);  mul_760 = unsqueeze_2029 = None
        unsqueeze_2030: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg525_1, -1);  arg525_1 = None
        unsqueeze_2031: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2030, -1);  unsqueeze_2030 = None
        add_654: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_761, unsqueeze_2031);  mul_761 = unsqueeze_2031 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_246: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_654);  add_654 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_245 = torch.ops.aten.split.Tensor(relu_245, 56, 1)
        getitem_1965: "f32[8, 56, 14, 14]" = split_245[1];  split_245 = None
        add_655: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_246, getitem_1965);  getitem_1965 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_254: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_655, arg526_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_655 = arg526_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_656: "f32[56]" = torch.ops.aten.add.Tensor(arg528_1, 1e-05);  arg528_1 = None
        sqrt_254: "f32[56]" = torch.ops.aten.sqrt.default(add_656);  add_656 = None
        reciprocal_254: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_254);  sqrt_254 = None
        mul_762: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_254, 1);  reciprocal_254 = None
        unsqueeze_2032: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg527_1, -1);  arg527_1 = None
        unsqueeze_2033: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2032, -1);  unsqueeze_2032 = None
        unsqueeze_2034: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_762, -1);  mul_762 = None
        unsqueeze_2035: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2034, -1);  unsqueeze_2034 = None
        sub_254: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_254, unsqueeze_2033);  convolution_254 = unsqueeze_2033 = None
        mul_763: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_2035);  sub_254 = unsqueeze_2035 = None
        unsqueeze_2036: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg529_1, -1);  arg529_1 = None
        unsqueeze_2037: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2036, -1);  unsqueeze_2036 = None
        mul_764: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_763, unsqueeze_2037);  mul_763 = unsqueeze_2037 = None
        unsqueeze_2038: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg530_1, -1);  arg530_1 = None
        unsqueeze_2039: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2038, -1);  unsqueeze_2038 = None
        add_657: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_764, unsqueeze_2039);  mul_764 = unsqueeze_2039 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_247: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_657);  add_657 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_246 = torch.ops.aten.split.Tensor(relu_245, 56, 1)
        getitem_1974: "f32[8, 56, 14, 14]" = split_246[2];  split_246 = None
        add_658: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_247, getitem_1974);  getitem_1974 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_255: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_658, arg531_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_658 = arg531_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_659: "f32[56]" = torch.ops.aten.add.Tensor(arg533_1, 1e-05);  arg533_1 = None
        sqrt_255: "f32[56]" = torch.ops.aten.sqrt.default(add_659);  add_659 = None
        reciprocal_255: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_255);  sqrt_255 = None
        mul_765: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_255, 1);  reciprocal_255 = None
        unsqueeze_2040: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg532_1, -1);  arg532_1 = None
        unsqueeze_2041: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2040, -1);  unsqueeze_2040 = None
        unsqueeze_2042: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_765, -1);  mul_765 = None
        unsqueeze_2043: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2042, -1);  unsqueeze_2042 = None
        sub_255: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_255, unsqueeze_2041);  convolution_255 = unsqueeze_2041 = None
        mul_766: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_2043);  sub_255 = unsqueeze_2043 = None
        unsqueeze_2044: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg534_1, -1);  arg534_1 = None
        unsqueeze_2045: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2044, -1);  unsqueeze_2044 = None
        mul_767: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_766, unsqueeze_2045);  mul_766 = unsqueeze_2045 = None
        unsqueeze_2046: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg535_1, -1);  arg535_1 = None
        unsqueeze_2047: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2046, -1);  unsqueeze_2046 = None
        add_660: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_767, unsqueeze_2047);  mul_767 = unsqueeze_2047 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_248: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_660);  add_660 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_247 = torch.ops.aten.split.Tensor(relu_245, 56, 1)
        getitem_1983: "f32[8, 56, 14, 14]" = split_247[3];  split_247 = None
        add_661: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_248, getitem_1983);  getitem_1983 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_256: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_661, arg536_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_661 = arg536_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_662: "f32[56]" = torch.ops.aten.add.Tensor(arg538_1, 1e-05);  arg538_1 = None
        sqrt_256: "f32[56]" = torch.ops.aten.sqrt.default(add_662);  add_662 = None
        reciprocal_256: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_256);  sqrt_256 = None
        mul_768: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_256, 1);  reciprocal_256 = None
        unsqueeze_2048: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg537_1, -1);  arg537_1 = None
        unsqueeze_2049: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2048, -1);  unsqueeze_2048 = None
        unsqueeze_2050: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_768, -1);  mul_768 = None
        unsqueeze_2051: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2050, -1);  unsqueeze_2050 = None
        sub_256: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_256, unsqueeze_2049);  convolution_256 = unsqueeze_2049 = None
        mul_769: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_2051);  sub_256 = unsqueeze_2051 = None
        unsqueeze_2052: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg539_1, -1);  arg539_1 = None
        unsqueeze_2053: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2052, -1);  unsqueeze_2052 = None
        mul_770: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_769, unsqueeze_2053);  mul_769 = unsqueeze_2053 = None
        unsqueeze_2054: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg540_1, -1);  arg540_1 = None
        unsqueeze_2055: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2054, -1);  unsqueeze_2054 = None
        add_663: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_770, unsqueeze_2055);  mul_770 = unsqueeze_2055 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_249: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_663);  add_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_248 = torch.ops.aten.split.Tensor(relu_245, 56, 1)
        getitem_1992: "f32[8, 56, 14, 14]" = split_248[4];  split_248 = None
        add_664: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_249, getitem_1992);  getitem_1992 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_257: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_664, arg541_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_664 = arg541_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_665: "f32[56]" = torch.ops.aten.add.Tensor(arg543_1, 1e-05);  arg543_1 = None
        sqrt_257: "f32[56]" = torch.ops.aten.sqrt.default(add_665);  add_665 = None
        reciprocal_257: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_257);  sqrt_257 = None
        mul_771: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_257, 1);  reciprocal_257 = None
        unsqueeze_2056: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg542_1, -1);  arg542_1 = None
        unsqueeze_2057: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2056, -1);  unsqueeze_2056 = None
        unsqueeze_2058: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_771, -1);  mul_771 = None
        unsqueeze_2059: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2058, -1);  unsqueeze_2058 = None
        sub_257: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_257, unsqueeze_2057);  convolution_257 = unsqueeze_2057 = None
        mul_772: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_2059);  sub_257 = unsqueeze_2059 = None
        unsqueeze_2060: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg544_1, -1);  arg544_1 = None
        unsqueeze_2061: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2060, -1);  unsqueeze_2060 = None
        mul_773: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_772, unsqueeze_2061);  mul_772 = unsqueeze_2061 = None
        unsqueeze_2062: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg545_1, -1);  arg545_1 = None
        unsqueeze_2063: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2062, -1);  unsqueeze_2062 = None
        add_666: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_773, unsqueeze_2063);  mul_773 = unsqueeze_2063 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_250: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_666);  add_666 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_249 = torch.ops.aten.split.Tensor(relu_245, 56, 1)
        getitem_2001: "f32[8, 56, 14, 14]" = split_249[5];  split_249 = None
        add_667: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_250, getitem_2001);  getitem_2001 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_258: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_667, arg546_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_667 = arg546_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_668: "f32[56]" = torch.ops.aten.add.Tensor(arg548_1, 1e-05);  arg548_1 = None
        sqrt_258: "f32[56]" = torch.ops.aten.sqrt.default(add_668);  add_668 = None
        reciprocal_258: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_258);  sqrt_258 = None
        mul_774: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_258, 1);  reciprocal_258 = None
        unsqueeze_2064: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg547_1, -1);  arg547_1 = None
        unsqueeze_2065: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2064, -1);  unsqueeze_2064 = None
        unsqueeze_2066: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_774, -1);  mul_774 = None
        unsqueeze_2067: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2066, -1);  unsqueeze_2066 = None
        sub_258: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_258, unsqueeze_2065);  convolution_258 = unsqueeze_2065 = None
        mul_775: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_2067);  sub_258 = unsqueeze_2067 = None
        unsqueeze_2068: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg549_1, -1);  arg549_1 = None
        unsqueeze_2069: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2068, -1);  unsqueeze_2068 = None
        mul_776: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_775, unsqueeze_2069);  mul_775 = unsqueeze_2069 = None
        unsqueeze_2070: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg550_1, -1);  arg550_1 = None
        unsqueeze_2071: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2070, -1);  unsqueeze_2070 = None
        add_669: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_776, unsqueeze_2071);  mul_776 = unsqueeze_2071 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_251: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_669);  add_669 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_250 = torch.ops.aten.split.Tensor(relu_245, 56, 1)
        getitem_2010: "f32[8, 56, 14, 14]" = split_250[6];  split_250 = None
        add_670: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_251, getitem_2010);  getitem_2010 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_259: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_670, arg551_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_670 = arg551_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_671: "f32[56]" = torch.ops.aten.add.Tensor(arg553_1, 1e-05);  arg553_1 = None
        sqrt_259: "f32[56]" = torch.ops.aten.sqrt.default(add_671);  add_671 = None
        reciprocal_259: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_259);  sqrt_259 = None
        mul_777: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_259, 1);  reciprocal_259 = None
        unsqueeze_2072: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg552_1, -1);  arg552_1 = None
        unsqueeze_2073: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2072, -1);  unsqueeze_2072 = None
        unsqueeze_2074: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_777, -1);  mul_777 = None
        unsqueeze_2075: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2074, -1);  unsqueeze_2074 = None
        sub_259: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_259, unsqueeze_2073);  convolution_259 = unsqueeze_2073 = None
        mul_778: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_2075);  sub_259 = unsqueeze_2075 = None
        unsqueeze_2076: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg554_1, -1);  arg554_1 = None
        unsqueeze_2077: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2076, -1);  unsqueeze_2076 = None
        mul_779: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_778, unsqueeze_2077);  mul_778 = unsqueeze_2077 = None
        unsqueeze_2078: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg555_1, -1);  arg555_1 = None
        unsqueeze_2079: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2078, -1);  unsqueeze_2078 = None
        add_672: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_779, unsqueeze_2079);  mul_779 = unsqueeze_2079 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_252: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_672);  add_672 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_251 = torch.ops.aten.split.Tensor(relu_245, 56, 1);  relu_245 = None
        getitem_2019: "f32[8, 56, 14, 14]" = split_251[7];  split_251 = None
        cat_27: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([relu_246, relu_247, relu_248, relu_249, relu_250, relu_251, relu_252, getitem_2019], 1);  relu_246 = relu_247 = relu_248 = relu_249 = relu_250 = relu_251 = relu_252 = getitem_2019 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_260: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_27, arg556_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_27 = arg556_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_673: "f32[1024]" = torch.ops.aten.add.Tensor(arg558_1, 1e-05);  arg558_1 = None
        sqrt_260: "f32[1024]" = torch.ops.aten.sqrt.default(add_673);  add_673 = None
        reciprocal_260: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_260);  sqrt_260 = None
        mul_780: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_260, 1);  reciprocal_260 = None
        unsqueeze_2080: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg557_1, -1);  arg557_1 = None
        unsqueeze_2081: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2080, -1);  unsqueeze_2080 = None
        unsqueeze_2082: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_780, -1);  mul_780 = None
        unsqueeze_2083: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2082, -1);  unsqueeze_2082 = None
        sub_260: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_260, unsqueeze_2081);  convolution_260 = unsqueeze_2081 = None
        mul_781: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_2083);  sub_260 = unsqueeze_2083 = None
        unsqueeze_2084: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg559_1, -1);  arg559_1 = None
        unsqueeze_2085: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2084, -1);  unsqueeze_2084 = None
        mul_782: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_781, unsqueeze_2085);  mul_781 = unsqueeze_2085 = None
        unsqueeze_2086: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg560_1, -1);  arg560_1 = None
        unsqueeze_2087: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2086, -1);  unsqueeze_2086 = None
        add_674: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_782, unsqueeze_2087);  mul_782 = unsqueeze_2087 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_675: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_674, relu_244);  add_674 = relu_244 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_253: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_675);  add_675 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_261: "f32[8, 448, 14, 14]" = torch.ops.aten.convolution.default(relu_253, arg561_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg561_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_676: "f32[448]" = torch.ops.aten.add.Tensor(arg563_1, 1e-05);  arg563_1 = None
        sqrt_261: "f32[448]" = torch.ops.aten.sqrt.default(add_676);  add_676 = None
        reciprocal_261: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_261);  sqrt_261 = None
        mul_783: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_261, 1);  reciprocal_261 = None
        unsqueeze_2088: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg562_1, -1);  arg562_1 = None
        unsqueeze_2089: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2088, -1);  unsqueeze_2088 = None
        unsqueeze_2090: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_783, -1);  mul_783 = None
        unsqueeze_2091: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2090, -1);  unsqueeze_2090 = None
        sub_261: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_261, unsqueeze_2089);  convolution_261 = unsqueeze_2089 = None
        mul_784: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_2091);  sub_261 = unsqueeze_2091 = None
        unsqueeze_2092: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg564_1, -1);  arg564_1 = None
        unsqueeze_2093: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2092, -1);  unsqueeze_2092 = None
        mul_785: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(mul_784, unsqueeze_2093);  mul_784 = unsqueeze_2093 = None
        unsqueeze_2094: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg565_1, -1);  arg565_1 = None
        unsqueeze_2095: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2094, -1);  unsqueeze_2094 = None
        add_677: "f32[8, 448, 14, 14]" = torch.ops.aten.add.Tensor(mul_785, unsqueeze_2095);  mul_785 = unsqueeze_2095 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_254: "f32[8, 448, 14, 14]" = torch.ops.aten.relu.default(add_677);  add_677 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_253 = torch.ops.aten.split.Tensor(relu_254, 56, 1)
        getitem_2028: "f32[8, 56, 14, 14]" = split_253[0];  split_253 = None
        convolution_262: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(getitem_2028, arg566_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2028 = arg566_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_678: "f32[56]" = torch.ops.aten.add.Tensor(arg568_1, 1e-05);  arg568_1 = None
        sqrt_262: "f32[56]" = torch.ops.aten.sqrt.default(add_678);  add_678 = None
        reciprocal_262: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_262);  sqrt_262 = None
        mul_786: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_262, 1);  reciprocal_262 = None
        unsqueeze_2096: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg567_1, -1);  arg567_1 = None
        unsqueeze_2097: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2096, -1);  unsqueeze_2096 = None
        unsqueeze_2098: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_786, -1);  mul_786 = None
        unsqueeze_2099: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2098, -1);  unsqueeze_2098 = None
        sub_262: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_262, unsqueeze_2097);  convolution_262 = unsqueeze_2097 = None
        mul_787: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_2099);  sub_262 = unsqueeze_2099 = None
        unsqueeze_2100: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg569_1, -1);  arg569_1 = None
        unsqueeze_2101: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2100, -1);  unsqueeze_2100 = None
        mul_788: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_787, unsqueeze_2101);  mul_787 = unsqueeze_2101 = None
        unsqueeze_2102: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg570_1, -1);  arg570_1 = None
        unsqueeze_2103: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2102, -1);  unsqueeze_2102 = None
        add_679: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_788, unsqueeze_2103);  mul_788 = unsqueeze_2103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_255: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_679);  add_679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_254 = torch.ops.aten.split.Tensor(relu_254, 56, 1)
        getitem_2037: "f32[8, 56, 14, 14]" = split_254[1];  split_254 = None
        add_680: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_255, getitem_2037);  getitem_2037 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_263: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_680, arg571_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_680 = arg571_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_681: "f32[56]" = torch.ops.aten.add.Tensor(arg573_1, 1e-05);  arg573_1 = None
        sqrt_263: "f32[56]" = torch.ops.aten.sqrt.default(add_681);  add_681 = None
        reciprocal_263: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_263);  sqrt_263 = None
        mul_789: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_263, 1);  reciprocal_263 = None
        unsqueeze_2104: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg572_1, -1);  arg572_1 = None
        unsqueeze_2105: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2104, -1);  unsqueeze_2104 = None
        unsqueeze_2106: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_789, -1);  mul_789 = None
        unsqueeze_2107: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2106, -1);  unsqueeze_2106 = None
        sub_263: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_263, unsqueeze_2105);  convolution_263 = unsqueeze_2105 = None
        mul_790: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_2107);  sub_263 = unsqueeze_2107 = None
        unsqueeze_2108: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg574_1, -1);  arg574_1 = None
        unsqueeze_2109: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2108, -1);  unsqueeze_2108 = None
        mul_791: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_790, unsqueeze_2109);  mul_790 = unsqueeze_2109 = None
        unsqueeze_2110: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg575_1, -1);  arg575_1 = None
        unsqueeze_2111: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2110, -1);  unsqueeze_2110 = None
        add_682: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_791, unsqueeze_2111);  mul_791 = unsqueeze_2111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_256: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_682);  add_682 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_255 = torch.ops.aten.split.Tensor(relu_254, 56, 1)
        getitem_2046: "f32[8, 56, 14, 14]" = split_255[2];  split_255 = None
        add_683: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_256, getitem_2046);  getitem_2046 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_264: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_683, arg576_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_683 = arg576_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_684: "f32[56]" = torch.ops.aten.add.Tensor(arg578_1, 1e-05);  arg578_1 = None
        sqrt_264: "f32[56]" = torch.ops.aten.sqrt.default(add_684);  add_684 = None
        reciprocal_264: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_264);  sqrt_264 = None
        mul_792: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_264, 1);  reciprocal_264 = None
        unsqueeze_2112: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg577_1, -1);  arg577_1 = None
        unsqueeze_2113: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2112, -1);  unsqueeze_2112 = None
        unsqueeze_2114: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_792, -1);  mul_792 = None
        unsqueeze_2115: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2114, -1);  unsqueeze_2114 = None
        sub_264: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_264, unsqueeze_2113);  convolution_264 = unsqueeze_2113 = None
        mul_793: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_2115);  sub_264 = unsqueeze_2115 = None
        unsqueeze_2116: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg579_1, -1);  arg579_1 = None
        unsqueeze_2117: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2116, -1);  unsqueeze_2116 = None
        mul_794: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_793, unsqueeze_2117);  mul_793 = unsqueeze_2117 = None
        unsqueeze_2118: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg580_1, -1);  arg580_1 = None
        unsqueeze_2119: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2118, -1);  unsqueeze_2118 = None
        add_685: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_794, unsqueeze_2119);  mul_794 = unsqueeze_2119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_257: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_685);  add_685 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_256 = torch.ops.aten.split.Tensor(relu_254, 56, 1)
        getitem_2055: "f32[8, 56, 14, 14]" = split_256[3];  split_256 = None
        add_686: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_257, getitem_2055);  getitem_2055 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_265: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_686, arg581_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_686 = arg581_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_687: "f32[56]" = torch.ops.aten.add.Tensor(arg583_1, 1e-05);  arg583_1 = None
        sqrt_265: "f32[56]" = torch.ops.aten.sqrt.default(add_687);  add_687 = None
        reciprocal_265: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_265);  sqrt_265 = None
        mul_795: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_265, 1);  reciprocal_265 = None
        unsqueeze_2120: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg582_1, -1);  arg582_1 = None
        unsqueeze_2121: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2120, -1);  unsqueeze_2120 = None
        unsqueeze_2122: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_795, -1);  mul_795 = None
        unsqueeze_2123: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2122, -1);  unsqueeze_2122 = None
        sub_265: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_265, unsqueeze_2121);  convolution_265 = unsqueeze_2121 = None
        mul_796: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_2123);  sub_265 = unsqueeze_2123 = None
        unsqueeze_2124: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg584_1, -1);  arg584_1 = None
        unsqueeze_2125: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2124, -1);  unsqueeze_2124 = None
        mul_797: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_796, unsqueeze_2125);  mul_796 = unsqueeze_2125 = None
        unsqueeze_2126: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg585_1, -1);  arg585_1 = None
        unsqueeze_2127: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2126, -1);  unsqueeze_2126 = None
        add_688: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_797, unsqueeze_2127);  mul_797 = unsqueeze_2127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_258: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_688);  add_688 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_257 = torch.ops.aten.split.Tensor(relu_254, 56, 1)
        getitem_2064: "f32[8, 56, 14, 14]" = split_257[4];  split_257 = None
        add_689: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_258, getitem_2064);  getitem_2064 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_266: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_689, arg586_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_689 = arg586_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_690: "f32[56]" = torch.ops.aten.add.Tensor(arg588_1, 1e-05);  arg588_1 = None
        sqrt_266: "f32[56]" = torch.ops.aten.sqrt.default(add_690);  add_690 = None
        reciprocal_266: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_266);  sqrt_266 = None
        mul_798: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_266, 1);  reciprocal_266 = None
        unsqueeze_2128: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg587_1, -1);  arg587_1 = None
        unsqueeze_2129: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2128, -1);  unsqueeze_2128 = None
        unsqueeze_2130: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_798, -1);  mul_798 = None
        unsqueeze_2131: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2130, -1);  unsqueeze_2130 = None
        sub_266: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_266, unsqueeze_2129);  convolution_266 = unsqueeze_2129 = None
        mul_799: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_2131);  sub_266 = unsqueeze_2131 = None
        unsqueeze_2132: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg589_1, -1);  arg589_1 = None
        unsqueeze_2133: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2132, -1);  unsqueeze_2132 = None
        mul_800: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_799, unsqueeze_2133);  mul_799 = unsqueeze_2133 = None
        unsqueeze_2134: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg590_1, -1);  arg590_1 = None
        unsqueeze_2135: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2134, -1);  unsqueeze_2134 = None
        add_691: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_800, unsqueeze_2135);  mul_800 = unsqueeze_2135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_259: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_691);  add_691 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_258 = torch.ops.aten.split.Tensor(relu_254, 56, 1)
        getitem_2073: "f32[8, 56, 14, 14]" = split_258[5];  split_258 = None
        add_692: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_259, getitem_2073);  getitem_2073 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_267: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_692, arg591_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_692 = arg591_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_693: "f32[56]" = torch.ops.aten.add.Tensor(arg593_1, 1e-05);  arg593_1 = None
        sqrt_267: "f32[56]" = torch.ops.aten.sqrt.default(add_693);  add_693 = None
        reciprocal_267: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_267);  sqrt_267 = None
        mul_801: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_267, 1);  reciprocal_267 = None
        unsqueeze_2136: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg592_1, -1);  arg592_1 = None
        unsqueeze_2137: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2136, -1);  unsqueeze_2136 = None
        unsqueeze_2138: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_801, -1);  mul_801 = None
        unsqueeze_2139: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2138, -1);  unsqueeze_2138 = None
        sub_267: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_267, unsqueeze_2137);  convolution_267 = unsqueeze_2137 = None
        mul_802: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_2139);  sub_267 = unsqueeze_2139 = None
        unsqueeze_2140: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg594_1, -1);  arg594_1 = None
        unsqueeze_2141: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2140, -1);  unsqueeze_2140 = None
        mul_803: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_802, unsqueeze_2141);  mul_802 = unsqueeze_2141 = None
        unsqueeze_2142: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg595_1, -1);  arg595_1 = None
        unsqueeze_2143: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2142, -1);  unsqueeze_2142 = None
        add_694: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_803, unsqueeze_2143);  mul_803 = unsqueeze_2143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_260: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_694);  add_694 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_259 = torch.ops.aten.split.Tensor(relu_254, 56, 1)
        getitem_2082: "f32[8, 56, 14, 14]" = split_259[6];  split_259 = None
        add_695: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(relu_260, getitem_2082);  getitem_2082 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_268: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_695, arg596_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_695 = arg596_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_696: "f32[56]" = torch.ops.aten.add.Tensor(arg598_1, 1e-05);  arg598_1 = None
        sqrt_268: "f32[56]" = torch.ops.aten.sqrt.default(add_696);  add_696 = None
        reciprocal_268: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_268);  sqrt_268 = None
        mul_804: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_268, 1);  reciprocal_268 = None
        unsqueeze_2144: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg597_1, -1);  arg597_1 = None
        unsqueeze_2145: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2144, -1);  unsqueeze_2144 = None
        unsqueeze_2146: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_804, -1);  mul_804 = None
        unsqueeze_2147: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2146, -1);  unsqueeze_2146 = None
        sub_268: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_268, unsqueeze_2145);  convolution_268 = unsqueeze_2145 = None
        mul_805: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_2147);  sub_268 = unsqueeze_2147 = None
        unsqueeze_2148: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg599_1, -1);  arg599_1 = None
        unsqueeze_2149: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2148, -1);  unsqueeze_2148 = None
        mul_806: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_805, unsqueeze_2149);  mul_805 = unsqueeze_2149 = None
        unsqueeze_2150: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg600_1, -1);  arg600_1 = None
        unsqueeze_2151: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2150, -1);  unsqueeze_2150 = None
        add_697: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_806, unsqueeze_2151);  mul_806 = unsqueeze_2151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_261: "f32[8, 56, 14, 14]" = torch.ops.aten.relu.default(add_697);  add_697 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_260 = torch.ops.aten.split.Tensor(relu_254, 56, 1);  relu_254 = None
        getitem_2091: "f32[8, 56, 14, 14]" = split_260[7];  split_260 = None
        cat_28: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([relu_255, relu_256, relu_257, relu_258, relu_259, relu_260, relu_261, getitem_2091], 1);  relu_255 = relu_256 = relu_257 = relu_258 = relu_259 = relu_260 = relu_261 = getitem_2091 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_269: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_28, arg601_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_28 = arg601_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_698: "f32[1024]" = torch.ops.aten.add.Tensor(arg603_1, 1e-05);  arg603_1 = None
        sqrt_269: "f32[1024]" = torch.ops.aten.sqrt.default(add_698);  add_698 = None
        reciprocal_269: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_269);  sqrt_269 = None
        mul_807: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_269, 1);  reciprocal_269 = None
        unsqueeze_2152: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg602_1, -1);  arg602_1 = None
        unsqueeze_2153: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2152, -1);  unsqueeze_2152 = None
        unsqueeze_2154: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_807, -1);  mul_807 = None
        unsqueeze_2155: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2154, -1);  unsqueeze_2154 = None
        sub_269: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_269, unsqueeze_2153);  convolution_269 = unsqueeze_2153 = None
        mul_808: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_2155);  sub_269 = unsqueeze_2155 = None
        unsqueeze_2156: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg604_1, -1);  arg604_1 = None
        unsqueeze_2157: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2156, -1);  unsqueeze_2156 = None
        mul_809: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_808, unsqueeze_2157);  mul_808 = unsqueeze_2157 = None
        unsqueeze_2158: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg605_1, -1);  arg605_1 = None
        unsqueeze_2159: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2158, -1);  unsqueeze_2158 = None
        add_699: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_809, unsqueeze_2159);  mul_809 = unsqueeze_2159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_700: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_699, relu_253);  add_699 = relu_253 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_262: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_700);  add_700 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_270: "f32[8, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_262, arg606_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg606_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_701: "f32[896]" = torch.ops.aten.add.Tensor(arg608_1, 1e-05);  arg608_1 = None
        sqrt_270: "f32[896]" = torch.ops.aten.sqrt.default(add_701);  add_701 = None
        reciprocal_270: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_270);  sqrt_270 = None
        mul_810: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_270, 1);  reciprocal_270 = None
        unsqueeze_2160: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg607_1, -1);  arg607_1 = None
        unsqueeze_2161: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2160, -1);  unsqueeze_2160 = None
        unsqueeze_2162: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_810, -1);  mul_810 = None
        unsqueeze_2163: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2162, -1);  unsqueeze_2162 = None
        sub_270: "f32[8, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_270, unsqueeze_2161);  convolution_270 = unsqueeze_2161 = None
        mul_811: "f32[8, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_2163);  sub_270 = unsqueeze_2163 = None
        unsqueeze_2164: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg609_1, -1);  arg609_1 = None
        unsqueeze_2165: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2164, -1);  unsqueeze_2164 = None
        mul_812: "f32[8, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_811, unsqueeze_2165);  mul_811 = unsqueeze_2165 = None
        unsqueeze_2166: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg610_1, -1);  arg610_1 = None
        unsqueeze_2167: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2166, -1);  unsqueeze_2166 = None
        add_702: "f32[8, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_812, unsqueeze_2167);  mul_812 = unsqueeze_2167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_263: "f32[8, 896, 14, 14]" = torch.ops.aten.relu.default(add_702);  add_702 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_262 = torch.ops.aten.split.Tensor(relu_263, 112, 1)
        getitem_2100: "f32[8, 112, 14, 14]" = split_262[0];  split_262 = None
        convolution_271: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_2100, arg611_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2100 = arg611_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_703: "f32[112]" = torch.ops.aten.add.Tensor(arg613_1, 1e-05);  arg613_1 = None
        sqrt_271: "f32[112]" = torch.ops.aten.sqrt.default(add_703);  add_703 = None
        reciprocal_271: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_271);  sqrt_271 = None
        mul_813: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_271, 1);  reciprocal_271 = None
        unsqueeze_2168: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg612_1, -1);  arg612_1 = None
        unsqueeze_2169: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2168, -1);  unsqueeze_2168 = None
        unsqueeze_2170: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_813, -1);  mul_813 = None
        unsqueeze_2171: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2170, -1);  unsqueeze_2170 = None
        sub_271: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_271, unsqueeze_2169);  convolution_271 = unsqueeze_2169 = None
        mul_814: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_2171);  sub_271 = unsqueeze_2171 = None
        unsqueeze_2172: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg614_1, -1);  arg614_1 = None
        unsqueeze_2173: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2172, -1);  unsqueeze_2172 = None
        mul_815: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_814, unsqueeze_2173);  mul_814 = unsqueeze_2173 = None
        unsqueeze_2174: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg615_1, -1);  arg615_1 = None
        unsqueeze_2175: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2174, -1);  unsqueeze_2174 = None
        add_704: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_815, unsqueeze_2175);  mul_815 = unsqueeze_2175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_264: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_704);  add_704 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_263 = torch.ops.aten.split.Tensor(relu_263, 112, 1)
        getitem_2109: "f32[8, 112, 14, 14]" = split_263[1];  split_263 = None
        convolution_272: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_2109, arg616_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2109 = arg616_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_705: "f32[112]" = torch.ops.aten.add.Tensor(arg618_1, 1e-05);  arg618_1 = None
        sqrt_272: "f32[112]" = torch.ops.aten.sqrt.default(add_705);  add_705 = None
        reciprocal_272: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_272);  sqrt_272 = None
        mul_816: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_272, 1);  reciprocal_272 = None
        unsqueeze_2176: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg617_1, -1);  arg617_1 = None
        unsqueeze_2177: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2176, -1);  unsqueeze_2176 = None
        unsqueeze_2178: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_816, -1);  mul_816 = None
        unsqueeze_2179: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2178, -1);  unsqueeze_2178 = None
        sub_272: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_272, unsqueeze_2177);  convolution_272 = unsqueeze_2177 = None
        mul_817: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_2179);  sub_272 = unsqueeze_2179 = None
        unsqueeze_2180: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg619_1, -1);  arg619_1 = None
        unsqueeze_2181: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2180, -1);  unsqueeze_2180 = None
        mul_818: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_817, unsqueeze_2181);  mul_817 = unsqueeze_2181 = None
        unsqueeze_2182: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg620_1, -1);  arg620_1 = None
        unsqueeze_2183: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2182, -1);  unsqueeze_2182 = None
        add_706: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_818, unsqueeze_2183);  mul_818 = unsqueeze_2183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_265: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_706);  add_706 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_264 = torch.ops.aten.split.Tensor(relu_263, 112, 1)
        getitem_2118: "f32[8, 112, 14, 14]" = split_264[2];  split_264 = None
        convolution_273: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_2118, arg621_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2118 = arg621_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_707: "f32[112]" = torch.ops.aten.add.Tensor(arg623_1, 1e-05);  arg623_1 = None
        sqrt_273: "f32[112]" = torch.ops.aten.sqrt.default(add_707);  add_707 = None
        reciprocal_273: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_273);  sqrt_273 = None
        mul_819: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_273, 1);  reciprocal_273 = None
        unsqueeze_2184: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg622_1, -1);  arg622_1 = None
        unsqueeze_2185: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2184, -1);  unsqueeze_2184 = None
        unsqueeze_2186: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_819, -1);  mul_819 = None
        unsqueeze_2187: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2186, -1);  unsqueeze_2186 = None
        sub_273: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_273, unsqueeze_2185);  convolution_273 = unsqueeze_2185 = None
        mul_820: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_2187);  sub_273 = unsqueeze_2187 = None
        unsqueeze_2188: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg624_1, -1);  arg624_1 = None
        unsqueeze_2189: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2188, -1);  unsqueeze_2188 = None
        mul_821: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_820, unsqueeze_2189);  mul_820 = unsqueeze_2189 = None
        unsqueeze_2190: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg625_1, -1);  arg625_1 = None
        unsqueeze_2191: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2190, -1);  unsqueeze_2190 = None
        add_708: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_821, unsqueeze_2191);  mul_821 = unsqueeze_2191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_266: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_708);  add_708 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_265 = torch.ops.aten.split.Tensor(relu_263, 112, 1)
        getitem_2127: "f32[8, 112, 14, 14]" = split_265[3];  split_265 = None
        convolution_274: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_2127, arg626_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2127 = arg626_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_709: "f32[112]" = torch.ops.aten.add.Tensor(arg628_1, 1e-05);  arg628_1 = None
        sqrt_274: "f32[112]" = torch.ops.aten.sqrt.default(add_709);  add_709 = None
        reciprocal_274: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_274);  sqrt_274 = None
        mul_822: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_274, 1);  reciprocal_274 = None
        unsqueeze_2192: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg627_1, -1);  arg627_1 = None
        unsqueeze_2193: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2192, -1);  unsqueeze_2192 = None
        unsqueeze_2194: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_822, -1);  mul_822 = None
        unsqueeze_2195: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2194, -1);  unsqueeze_2194 = None
        sub_274: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_274, unsqueeze_2193);  convolution_274 = unsqueeze_2193 = None
        mul_823: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_2195);  sub_274 = unsqueeze_2195 = None
        unsqueeze_2196: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg629_1, -1);  arg629_1 = None
        unsqueeze_2197: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2196, -1);  unsqueeze_2196 = None
        mul_824: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_823, unsqueeze_2197);  mul_823 = unsqueeze_2197 = None
        unsqueeze_2198: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg630_1, -1);  arg630_1 = None
        unsqueeze_2199: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2198, -1);  unsqueeze_2198 = None
        add_710: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_824, unsqueeze_2199);  mul_824 = unsqueeze_2199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_267: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_710);  add_710 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_266 = torch.ops.aten.split.Tensor(relu_263, 112, 1)
        getitem_2136: "f32[8, 112, 14, 14]" = split_266[4];  split_266 = None
        convolution_275: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_2136, arg631_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2136 = arg631_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_711: "f32[112]" = torch.ops.aten.add.Tensor(arg633_1, 1e-05);  arg633_1 = None
        sqrt_275: "f32[112]" = torch.ops.aten.sqrt.default(add_711);  add_711 = None
        reciprocal_275: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_275);  sqrt_275 = None
        mul_825: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_275, 1);  reciprocal_275 = None
        unsqueeze_2200: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg632_1, -1);  arg632_1 = None
        unsqueeze_2201: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2200, -1);  unsqueeze_2200 = None
        unsqueeze_2202: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_825, -1);  mul_825 = None
        unsqueeze_2203: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2202, -1);  unsqueeze_2202 = None
        sub_275: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_275, unsqueeze_2201);  convolution_275 = unsqueeze_2201 = None
        mul_826: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_2203);  sub_275 = unsqueeze_2203 = None
        unsqueeze_2204: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg634_1, -1);  arg634_1 = None
        unsqueeze_2205: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2204, -1);  unsqueeze_2204 = None
        mul_827: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_826, unsqueeze_2205);  mul_826 = unsqueeze_2205 = None
        unsqueeze_2206: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg635_1, -1);  arg635_1 = None
        unsqueeze_2207: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2206, -1);  unsqueeze_2206 = None
        add_712: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_827, unsqueeze_2207);  mul_827 = unsqueeze_2207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_268: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_712);  add_712 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_267 = torch.ops.aten.split.Tensor(relu_263, 112, 1)
        getitem_2145: "f32[8, 112, 14, 14]" = split_267[5];  split_267 = None
        convolution_276: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_2145, arg636_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2145 = arg636_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_713: "f32[112]" = torch.ops.aten.add.Tensor(arg638_1, 1e-05);  arg638_1 = None
        sqrt_276: "f32[112]" = torch.ops.aten.sqrt.default(add_713);  add_713 = None
        reciprocal_276: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_276);  sqrt_276 = None
        mul_828: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_276, 1);  reciprocal_276 = None
        unsqueeze_2208: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg637_1, -1);  arg637_1 = None
        unsqueeze_2209: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2208, -1);  unsqueeze_2208 = None
        unsqueeze_2210: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_828, -1);  mul_828 = None
        unsqueeze_2211: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2210, -1);  unsqueeze_2210 = None
        sub_276: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_276, unsqueeze_2209);  convolution_276 = unsqueeze_2209 = None
        mul_829: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_2211);  sub_276 = unsqueeze_2211 = None
        unsqueeze_2212: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg639_1, -1);  arg639_1 = None
        unsqueeze_2213: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2212, -1);  unsqueeze_2212 = None
        mul_830: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_829, unsqueeze_2213);  mul_829 = unsqueeze_2213 = None
        unsqueeze_2214: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg640_1, -1);  arg640_1 = None
        unsqueeze_2215: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2214, -1);  unsqueeze_2214 = None
        add_714: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_830, unsqueeze_2215);  mul_830 = unsqueeze_2215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_269: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_714);  add_714 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_268 = torch.ops.aten.split.Tensor(relu_263, 112, 1)
        getitem_2154: "f32[8, 112, 14, 14]" = split_268[6];  split_268 = None
        convolution_277: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_2154, arg641_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2154 = arg641_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_715: "f32[112]" = torch.ops.aten.add.Tensor(arg643_1, 1e-05);  arg643_1 = None
        sqrt_277: "f32[112]" = torch.ops.aten.sqrt.default(add_715);  add_715 = None
        reciprocal_277: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_277);  sqrt_277 = None
        mul_831: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_277, 1);  reciprocal_277 = None
        unsqueeze_2216: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg642_1, -1);  arg642_1 = None
        unsqueeze_2217: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2216, -1);  unsqueeze_2216 = None
        unsqueeze_2218: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_831, -1);  mul_831 = None
        unsqueeze_2219: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2218, -1);  unsqueeze_2218 = None
        sub_277: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_277, unsqueeze_2217);  convolution_277 = unsqueeze_2217 = None
        mul_832: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_2219);  sub_277 = unsqueeze_2219 = None
        unsqueeze_2220: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg644_1, -1);  arg644_1 = None
        unsqueeze_2221: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2220, -1);  unsqueeze_2220 = None
        mul_833: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_832, unsqueeze_2221);  mul_832 = unsqueeze_2221 = None
        unsqueeze_2222: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg645_1, -1);  arg645_1 = None
        unsqueeze_2223: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2222, -1);  unsqueeze_2222 = None
        add_716: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_833, unsqueeze_2223);  mul_833 = unsqueeze_2223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_270: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_716);  add_716 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:99 in forward, code: spo.append(self.pool(spx[-1]))
        split_269 = torch.ops.aten.split.Tensor(relu_263, 112, 1);  relu_263 = None
        getitem_2163: "f32[8, 112, 14, 14]" = split_269[7];  split_269 = None
        avg_pool2d_7: "f32[8, 112, 7, 7]" = torch.ops.aten.avg_pool2d.default(getitem_2163, [3, 3], [2, 2], [1, 1]);  getitem_2163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        cat_29: "f32[8, 896, 7, 7]" = torch.ops.aten.cat.default([relu_264, relu_265, relu_266, relu_267, relu_268, relu_269, relu_270, avg_pool2d_7], 1);  relu_264 = relu_265 = relu_266 = relu_267 = relu_268 = relu_269 = relu_270 = avg_pool2d_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_278: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_29, arg646_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_29 = arg646_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_717: "f32[2048]" = torch.ops.aten.add.Tensor(arg648_1, 1e-05);  arg648_1 = None
        sqrt_278: "f32[2048]" = torch.ops.aten.sqrt.default(add_717);  add_717 = None
        reciprocal_278: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_278);  sqrt_278 = None
        mul_834: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_278, 1);  reciprocal_278 = None
        unsqueeze_2224: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg647_1, -1);  arg647_1 = None
        unsqueeze_2225: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2224, -1);  unsqueeze_2224 = None
        unsqueeze_2226: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_834, -1);  mul_834 = None
        unsqueeze_2227: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2226, -1);  unsqueeze_2226 = None
        sub_278: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_278, unsqueeze_2225);  convolution_278 = unsqueeze_2225 = None
        mul_835: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_2227);  sub_278 = unsqueeze_2227 = None
        unsqueeze_2228: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg649_1, -1);  arg649_1 = None
        unsqueeze_2229: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2228, -1);  unsqueeze_2228 = None
        mul_836: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_835, unsqueeze_2229);  mul_835 = unsqueeze_2229 = None
        unsqueeze_2230: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg650_1, -1);  arg650_1 = None
        unsqueeze_2231: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2230, -1);  unsqueeze_2230 = None
        add_718: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_836, unsqueeze_2231);  mul_836 = unsqueeze_2231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:111 in forward, code: shortcut = self.downsample(x)
        convolution_279: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_262, arg651_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_262 = arg651_1 = None
        add_719: "f32[2048]" = torch.ops.aten.add.Tensor(arg653_1, 1e-05);  arg653_1 = None
        sqrt_279: "f32[2048]" = torch.ops.aten.sqrt.default(add_719);  add_719 = None
        reciprocal_279: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_279);  sqrt_279 = None
        mul_837: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_279, 1);  reciprocal_279 = None
        unsqueeze_2232: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg652_1, -1);  arg652_1 = None
        unsqueeze_2233: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2232, -1);  unsqueeze_2232 = None
        unsqueeze_2234: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_837, -1);  mul_837 = None
        unsqueeze_2235: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2234, -1);  unsqueeze_2234 = None
        sub_279: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_279, unsqueeze_2233);  convolution_279 = unsqueeze_2233 = None
        mul_838: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_2235);  sub_279 = unsqueeze_2235 = None
        unsqueeze_2236: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg654_1, -1);  arg654_1 = None
        unsqueeze_2237: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2236, -1);  unsqueeze_2236 = None
        mul_839: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_838, unsqueeze_2237);  mul_838 = unsqueeze_2237 = None
        unsqueeze_2238: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg655_1, -1);  arg655_1 = None
        unsqueeze_2239: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2238, -1);  unsqueeze_2238 = None
        add_720: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_839, unsqueeze_2239);  mul_839 = unsqueeze_2239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_721: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_718, add_720);  add_718 = add_720 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_271: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_721);  add_721 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_280: "f32[8, 896, 7, 7]" = torch.ops.aten.convolution.default(relu_271, arg656_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg656_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_722: "f32[896]" = torch.ops.aten.add.Tensor(arg658_1, 1e-05);  arg658_1 = None
        sqrt_280: "f32[896]" = torch.ops.aten.sqrt.default(add_722);  add_722 = None
        reciprocal_280: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_280);  sqrt_280 = None
        mul_840: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_280, 1);  reciprocal_280 = None
        unsqueeze_2240: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg657_1, -1);  arg657_1 = None
        unsqueeze_2241: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2240, -1);  unsqueeze_2240 = None
        unsqueeze_2242: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_840, -1);  mul_840 = None
        unsqueeze_2243: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2242, -1);  unsqueeze_2242 = None
        sub_280: "f32[8, 896, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_280, unsqueeze_2241);  convolution_280 = unsqueeze_2241 = None
        mul_841: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_2243);  sub_280 = unsqueeze_2243 = None
        unsqueeze_2244: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg659_1, -1);  arg659_1 = None
        unsqueeze_2245: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2244, -1);  unsqueeze_2244 = None
        mul_842: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(mul_841, unsqueeze_2245);  mul_841 = unsqueeze_2245 = None
        unsqueeze_2246: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg660_1, -1);  arg660_1 = None
        unsqueeze_2247: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2246, -1);  unsqueeze_2246 = None
        add_723: "f32[8, 896, 7, 7]" = torch.ops.aten.add.Tensor(mul_842, unsqueeze_2247);  mul_842 = unsqueeze_2247 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_272: "f32[8, 896, 7, 7]" = torch.ops.aten.relu.default(add_723);  add_723 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_271 = torch.ops.aten.split.Tensor(relu_272, 112, 1)
        getitem_2172: "f32[8, 112, 7, 7]" = split_271[0];  split_271 = None
        convolution_281: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_2172, arg661_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2172 = arg661_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_724: "f32[112]" = torch.ops.aten.add.Tensor(arg663_1, 1e-05);  arg663_1 = None
        sqrt_281: "f32[112]" = torch.ops.aten.sqrt.default(add_724);  add_724 = None
        reciprocal_281: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_281);  sqrt_281 = None
        mul_843: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_281, 1);  reciprocal_281 = None
        unsqueeze_2248: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg662_1, -1);  arg662_1 = None
        unsqueeze_2249: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2248, -1);  unsqueeze_2248 = None
        unsqueeze_2250: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_843, -1);  mul_843 = None
        unsqueeze_2251: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2250, -1);  unsqueeze_2250 = None
        sub_281: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_281, unsqueeze_2249);  convolution_281 = unsqueeze_2249 = None
        mul_844: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_2251);  sub_281 = unsqueeze_2251 = None
        unsqueeze_2252: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg664_1, -1);  arg664_1 = None
        unsqueeze_2253: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2252, -1);  unsqueeze_2252 = None
        mul_845: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_844, unsqueeze_2253);  mul_844 = unsqueeze_2253 = None
        unsqueeze_2254: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg665_1, -1);  arg665_1 = None
        unsqueeze_2255: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2254, -1);  unsqueeze_2254 = None
        add_725: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_845, unsqueeze_2255);  mul_845 = unsqueeze_2255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_273: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_725);  add_725 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_272 = torch.ops.aten.split.Tensor(relu_272, 112, 1)
        getitem_2181: "f32[8, 112, 7, 7]" = split_272[1];  split_272 = None
        add_726: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_273, getitem_2181);  getitem_2181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_282: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_726, arg666_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_726 = arg666_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_727: "f32[112]" = torch.ops.aten.add.Tensor(arg668_1, 1e-05);  arg668_1 = None
        sqrt_282: "f32[112]" = torch.ops.aten.sqrt.default(add_727);  add_727 = None
        reciprocal_282: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_282);  sqrt_282 = None
        mul_846: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_282, 1);  reciprocal_282 = None
        unsqueeze_2256: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg667_1, -1);  arg667_1 = None
        unsqueeze_2257: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2256, -1);  unsqueeze_2256 = None
        unsqueeze_2258: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_846, -1);  mul_846 = None
        unsqueeze_2259: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2258, -1);  unsqueeze_2258 = None
        sub_282: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_282, unsqueeze_2257);  convolution_282 = unsqueeze_2257 = None
        mul_847: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_2259);  sub_282 = unsqueeze_2259 = None
        unsqueeze_2260: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg669_1, -1);  arg669_1 = None
        unsqueeze_2261: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2260, -1);  unsqueeze_2260 = None
        mul_848: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_847, unsqueeze_2261);  mul_847 = unsqueeze_2261 = None
        unsqueeze_2262: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg670_1, -1);  arg670_1 = None
        unsqueeze_2263: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2262, -1);  unsqueeze_2262 = None
        add_728: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_848, unsqueeze_2263);  mul_848 = unsqueeze_2263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_274: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_728);  add_728 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_273 = torch.ops.aten.split.Tensor(relu_272, 112, 1)
        getitem_2190: "f32[8, 112, 7, 7]" = split_273[2];  split_273 = None
        add_729: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_274, getitem_2190);  getitem_2190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_283: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_729, arg671_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_729 = arg671_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_730: "f32[112]" = torch.ops.aten.add.Tensor(arg673_1, 1e-05);  arg673_1 = None
        sqrt_283: "f32[112]" = torch.ops.aten.sqrt.default(add_730);  add_730 = None
        reciprocal_283: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_283);  sqrt_283 = None
        mul_849: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_283, 1);  reciprocal_283 = None
        unsqueeze_2264: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg672_1, -1);  arg672_1 = None
        unsqueeze_2265: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2264, -1);  unsqueeze_2264 = None
        unsqueeze_2266: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_849, -1);  mul_849 = None
        unsqueeze_2267: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2266, -1);  unsqueeze_2266 = None
        sub_283: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_283, unsqueeze_2265);  convolution_283 = unsqueeze_2265 = None
        mul_850: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_2267);  sub_283 = unsqueeze_2267 = None
        unsqueeze_2268: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg674_1, -1);  arg674_1 = None
        unsqueeze_2269: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2268, -1);  unsqueeze_2268 = None
        mul_851: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_850, unsqueeze_2269);  mul_850 = unsqueeze_2269 = None
        unsqueeze_2270: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg675_1, -1);  arg675_1 = None
        unsqueeze_2271: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2270, -1);  unsqueeze_2270 = None
        add_731: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_851, unsqueeze_2271);  mul_851 = unsqueeze_2271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_275: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_731);  add_731 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_274 = torch.ops.aten.split.Tensor(relu_272, 112, 1)
        getitem_2199: "f32[8, 112, 7, 7]" = split_274[3];  split_274 = None
        add_732: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_275, getitem_2199);  getitem_2199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_284: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_732, arg676_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_732 = arg676_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_733: "f32[112]" = torch.ops.aten.add.Tensor(arg678_1, 1e-05);  arg678_1 = None
        sqrt_284: "f32[112]" = torch.ops.aten.sqrt.default(add_733);  add_733 = None
        reciprocal_284: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_284);  sqrt_284 = None
        mul_852: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_284, 1);  reciprocal_284 = None
        unsqueeze_2272: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg677_1, -1);  arg677_1 = None
        unsqueeze_2273: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2272, -1);  unsqueeze_2272 = None
        unsqueeze_2274: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_852, -1);  mul_852 = None
        unsqueeze_2275: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2274, -1);  unsqueeze_2274 = None
        sub_284: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_284, unsqueeze_2273);  convolution_284 = unsqueeze_2273 = None
        mul_853: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_2275);  sub_284 = unsqueeze_2275 = None
        unsqueeze_2276: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg679_1, -1);  arg679_1 = None
        unsqueeze_2277: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2276, -1);  unsqueeze_2276 = None
        mul_854: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_853, unsqueeze_2277);  mul_853 = unsqueeze_2277 = None
        unsqueeze_2278: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg680_1, -1);  arg680_1 = None
        unsqueeze_2279: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2278, -1);  unsqueeze_2278 = None
        add_734: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_854, unsqueeze_2279);  mul_854 = unsqueeze_2279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_276: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_734);  add_734 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_275 = torch.ops.aten.split.Tensor(relu_272, 112, 1)
        getitem_2208: "f32[8, 112, 7, 7]" = split_275[4];  split_275 = None
        add_735: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_276, getitem_2208);  getitem_2208 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_285: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_735, arg681_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_735 = arg681_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_736: "f32[112]" = torch.ops.aten.add.Tensor(arg683_1, 1e-05);  arg683_1 = None
        sqrt_285: "f32[112]" = torch.ops.aten.sqrt.default(add_736);  add_736 = None
        reciprocal_285: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_285);  sqrt_285 = None
        mul_855: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_285, 1);  reciprocal_285 = None
        unsqueeze_2280: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg682_1, -1);  arg682_1 = None
        unsqueeze_2281: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2280, -1);  unsqueeze_2280 = None
        unsqueeze_2282: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_855, -1);  mul_855 = None
        unsqueeze_2283: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2282, -1);  unsqueeze_2282 = None
        sub_285: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_285, unsqueeze_2281);  convolution_285 = unsqueeze_2281 = None
        mul_856: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_2283);  sub_285 = unsqueeze_2283 = None
        unsqueeze_2284: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg684_1, -1);  arg684_1 = None
        unsqueeze_2285: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2284, -1);  unsqueeze_2284 = None
        mul_857: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_856, unsqueeze_2285);  mul_856 = unsqueeze_2285 = None
        unsqueeze_2286: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg685_1, -1);  arg685_1 = None
        unsqueeze_2287: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2286, -1);  unsqueeze_2286 = None
        add_737: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_857, unsqueeze_2287);  mul_857 = unsqueeze_2287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_277: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_737);  add_737 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_276 = torch.ops.aten.split.Tensor(relu_272, 112, 1)
        getitem_2217: "f32[8, 112, 7, 7]" = split_276[5];  split_276 = None
        add_738: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_277, getitem_2217);  getitem_2217 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_286: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_738, arg686_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_738 = arg686_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_739: "f32[112]" = torch.ops.aten.add.Tensor(arg688_1, 1e-05);  arg688_1 = None
        sqrt_286: "f32[112]" = torch.ops.aten.sqrt.default(add_739);  add_739 = None
        reciprocal_286: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_286);  sqrt_286 = None
        mul_858: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_286, 1);  reciprocal_286 = None
        unsqueeze_2288: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg687_1, -1);  arg687_1 = None
        unsqueeze_2289: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2288, -1);  unsqueeze_2288 = None
        unsqueeze_2290: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_858, -1);  mul_858 = None
        unsqueeze_2291: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2290, -1);  unsqueeze_2290 = None
        sub_286: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_286, unsqueeze_2289);  convolution_286 = unsqueeze_2289 = None
        mul_859: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_2291);  sub_286 = unsqueeze_2291 = None
        unsqueeze_2292: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg689_1, -1);  arg689_1 = None
        unsqueeze_2293: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2292, -1);  unsqueeze_2292 = None
        mul_860: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_859, unsqueeze_2293);  mul_859 = unsqueeze_2293 = None
        unsqueeze_2294: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg690_1, -1);  arg690_1 = None
        unsqueeze_2295: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2294, -1);  unsqueeze_2294 = None
        add_740: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_860, unsqueeze_2295);  mul_860 = unsqueeze_2295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_278: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_740);  add_740 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_277 = torch.ops.aten.split.Tensor(relu_272, 112, 1)
        getitem_2226: "f32[8, 112, 7, 7]" = split_277[6];  split_277 = None
        add_741: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_278, getitem_2226);  getitem_2226 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_287: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_741, arg691_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_741 = arg691_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_742: "f32[112]" = torch.ops.aten.add.Tensor(arg693_1, 1e-05);  arg693_1 = None
        sqrt_287: "f32[112]" = torch.ops.aten.sqrt.default(add_742);  add_742 = None
        reciprocal_287: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_287);  sqrt_287 = None
        mul_861: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_287, 1);  reciprocal_287 = None
        unsqueeze_2296: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg692_1, -1);  arg692_1 = None
        unsqueeze_2297: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2296, -1);  unsqueeze_2296 = None
        unsqueeze_2298: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_861, -1);  mul_861 = None
        unsqueeze_2299: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2298, -1);  unsqueeze_2298 = None
        sub_287: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_287, unsqueeze_2297);  convolution_287 = unsqueeze_2297 = None
        mul_862: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_2299);  sub_287 = unsqueeze_2299 = None
        unsqueeze_2300: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg694_1, -1);  arg694_1 = None
        unsqueeze_2301: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2300, -1);  unsqueeze_2300 = None
        mul_863: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_862, unsqueeze_2301);  mul_862 = unsqueeze_2301 = None
        unsqueeze_2302: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg695_1, -1);  arg695_1 = None
        unsqueeze_2303: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2302, -1);  unsqueeze_2302 = None
        add_743: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_863, unsqueeze_2303);  mul_863 = unsqueeze_2303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_279: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_743);  add_743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_278 = torch.ops.aten.split.Tensor(relu_272, 112, 1);  relu_272 = None
        getitem_2235: "f32[8, 112, 7, 7]" = split_278[7];  split_278 = None
        cat_30: "f32[8, 896, 7, 7]" = torch.ops.aten.cat.default([relu_273, relu_274, relu_275, relu_276, relu_277, relu_278, relu_279, getitem_2235], 1);  relu_273 = relu_274 = relu_275 = relu_276 = relu_277 = relu_278 = relu_279 = getitem_2235 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_288: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_30, arg696_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_30 = arg696_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_744: "f32[2048]" = torch.ops.aten.add.Tensor(arg698_1, 1e-05);  arg698_1 = None
        sqrt_288: "f32[2048]" = torch.ops.aten.sqrt.default(add_744);  add_744 = None
        reciprocal_288: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_288);  sqrt_288 = None
        mul_864: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_288, 1);  reciprocal_288 = None
        unsqueeze_2304: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg697_1, -1);  arg697_1 = None
        unsqueeze_2305: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2304, -1);  unsqueeze_2304 = None
        unsqueeze_2306: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_864, -1);  mul_864 = None
        unsqueeze_2307: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2306, -1);  unsqueeze_2306 = None
        sub_288: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_288, unsqueeze_2305);  convolution_288 = unsqueeze_2305 = None
        mul_865: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_2307);  sub_288 = unsqueeze_2307 = None
        unsqueeze_2308: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg699_1, -1);  arg699_1 = None
        unsqueeze_2309: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2308, -1);  unsqueeze_2308 = None
        mul_866: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_865, unsqueeze_2309);  mul_865 = unsqueeze_2309 = None
        unsqueeze_2310: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg700_1, -1);  arg700_1 = None
        unsqueeze_2311: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2310, -1);  unsqueeze_2310 = None
        add_745: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_866, unsqueeze_2311);  mul_866 = unsqueeze_2311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_746: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_745, relu_271);  add_745 = relu_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_280: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_746);  add_746 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:81 in forward, code: out = self.conv1(x)
        convolution_289: "f32[8, 896, 7, 7]" = torch.ops.aten.convolution.default(relu_280, arg701_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg701_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:82 in forward, code: out = self.bn1(out)
        add_747: "f32[896]" = torch.ops.aten.add.Tensor(arg703_1, 1e-05);  arg703_1 = None
        sqrt_289: "f32[896]" = torch.ops.aten.sqrt.default(add_747);  add_747 = None
        reciprocal_289: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_289);  sqrt_289 = None
        mul_867: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_289, 1);  reciprocal_289 = None
        unsqueeze_2312: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg702_1, -1);  arg702_1 = None
        unsqueeze_2313: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2312, -1);  unsqueeze_2312 = None
        unsqueeze_2314: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_867, -1);  mul_867 = None
        unsqueeze_2315: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2314, -1);  unsqueeze_2314 = None
        sub_289: "f32[8, 896, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_289, unsqueeze_2313);  convolution_289 = unsqueeze_2313 = None
        mul_868: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_2315);  sub_289 = unsqueeze_2315 = None
        unsqueeze_2316: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg704_1, -1);  arg704_1 = None
        unsqueeze_2317: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2316, -1);  unsqueeze_2316 = None
        mul_869: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(mul_868, unsqueeze_2317);  mul_868 = unsqueeze_2317 = None
        unsqueeze_2318: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg705_1, -1);  arg705_1 = None
        unsqueeze_2319: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2318, -1);  unsqueeze_2318 = None
        add_748: "f32[8, 896, 7, 7]" = torch.ops.aten.add.Tensor(mul_869, unsqueeze_2319);  mul_869 = unsqueeze_2319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:83 in forward, code: out = self.relu(out)
        relu_281: "f32[8, 896, 7, 7]" = torch.ops.aten.relu.default(add_748);  add_748 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        split_280 = torch.ops.aten.split.Tensor(relu_281, 112, 1)
        getitem_2244: "f32[8, 112, 7, 7]" = split_280[0];  split_280 = None
        convolution_290: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(getitem_2244, arg706_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2244 = arg706_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_749: "f32[112]" = torch.ops.aten.add.Tensor(arg708_1, 1e-05);  arg708_1 = None
        sqrt_290: "f32[112]" = torch.ops.aten.sqrt.default(add_749);  add_749 = None
        reciprocal_290: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_290);  sqrt_290 = None
        mul_870: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_290, 1);  reciprocal_290 = None
        unsqueeze_2320: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg707_1, -1);  arg707_1 = None
        unsqueeze_2321: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2320, -1);  unsqueeze_2320 = None
        unsqueeze_2322: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_870, -1);  mul_870 = None
        unsqueeze_2323: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2322, -1);  unsqueeze_2322 = None
        sub_290: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_290, unsqueeze_2321);  convolution_290 = unsqueeze_2321 = None
        mul_871: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_2323);  sub_290 = unsqueeze_2323 = None
        unsqueeze_2324: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg709_1, -1);  arg709_1 = None
        unsqueeze_2325: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2324, -1);  unsqueeze_2324 = None
        mul_872: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_871, unsqueeze_2325);  mul_871 = unsqueeze_2325 = None
        unsqueeze_2326: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg710_1, -1);  arg710_1 = None
        unsqueeze_2327: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2326, -1);  unsqueeze_2326 = None
        add_750: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_872, unsqueeze_2327);  mul_872 = unsqueeze_2327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_282: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_750);  add_750 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_281 = torch.ops.aten.split.Tensor(relu_281, 112, 1)
        getitem_2253: "f32[8, 112, 7, 7]" = split_281[1];  split_281 = None
        add_751: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_282, getitem_2253);  getitem_2253 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_291: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_751, arg711_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_751 = arg711_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_752: "f32[112]" = torch.ops.aten.add.Tensor(arg713_1, 1e-05);  arg713_1 = None
        sqrt_291: "f32[112]" = torch.ops.aten.sqrt.default(add_752);  add_752 = None
        reciprocal_291: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_291);  sqrt_291 = None
        mul_873: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_291, 1);  reciprocal_291 = None
        unsqueeze_2328: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg712_1, -1);  arg712_1 = None
        unsqueeze_2329: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2328, -1);  unsqueeze_2328 = None
        unsqueeze_2330: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_873, -1);  mul_873 = None
        unsqueeze_2331: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2330, -1);  unsqueeze_2330 = None
        sub_291: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_291, unsqueeze_2329);  convolution_291 = unsqueeze_2329 = None
        mul_874: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_2331);  sub_291 = unsqueeze_2331 = None
        unsqueeze_2332: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg714_1, -1);  arg714_1 = None
        unsqueeze_2333: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2332, -1);  unsqueeze_2332 = None
        mul_875: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_874, unsqueeze_2333);  mul_874 = unsqueeze_2333 = None
        unsqueeze_2334: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg715_1, -1);  arg715_1 = None
        unsqueeze_2335: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2334, -1);  unsqueeze_2334 = None
        add_753: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_875, unsqueeze_2335);  mul_875 = unsqueeze_2335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_283: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_753);  add_753 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_282 = torch.ops.aten.split.Tensor(relu_281, 112, 1)
        getitem_2262: "f32[8, 112, 7, 7]" = split_282[2];  split_282 = None
        add_754: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_283, getitem_2262);  getitem_2262 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_292: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_754, arg716_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_754 = arg716_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_755: "f32[112]" = torch.ops.aten.add.Tensor(arg718_1, 1e-05);  arg718_1 = None
        sqrt_292: "f32[112]" = torch.ops.aten.sqrt.default(add_755);  add_755 = None
        reciprocal_292: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_292);  sqrt_292 = None
        mul_876: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_292, 1);  reciprocal_292 = None
        unsqueeze_2336: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg717_1, -1);  arg717_1 = None
        unsqueeze_2337: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2336, -1);  unsqueeze_2336 = None
        unsqueeze_2338: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_876, -1);  mul_876 = None
        unsqueeze_2339: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2338, -1);  unsqueeze_2338 = None
        sub_292: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_292, unsqueeze_2337);  convolution_292 = unsqueeze_2337 = None
        mul_877: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_2339);  sub_292 = unsqueeze_2339 = None
        unsqueeze_2340: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg719_1, -1);  arg719_1 = None
        unsqueeze_2341: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2340, -1);  unsqueeze_2340 = None
        mul_878: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_877, unsqueeze_2341);  mul_877 = unsqueeze_2341 = None
        unsqueeze_2342: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg720_1, -1);  arg720_1 = None
        unsqueeze_2343: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2342, -1);  unsqueeze_2342 = None
        add_756: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_878, unsqueeze_2343);  mul_878 = unsqueeze_2343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_284: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_756);  add_756 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_283 = torch.ops.aten.split.Tensor(relu_281, 112, 1)
        getitem_2271: "f32[8, 112, 7, 7]" = split_283[3];  split_283 = None
        add_757: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_284, getitem_2271);  getitem_2271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_293: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_757, arg721_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_757 = arg721_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_758: "f32[112]" = torch.ops.aten.add.Tensor(arg723_1, 1e-05);  arg723_1 = None
        sqrt_293: "f32[112]" = torch.ops.aten.sqrt.default(add_758);  add_758 = None
        reciprocal_293: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_293);  sqrt_293 = None
        mul_879: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_293, 1);  reciprocal_293 = None
        unsqueeze_2344: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg722_1, -1);  arg722_1 = None
        unsqueeze_2345: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2344, -1);  unsqueeze_2344 = None
        unsqueeze_2346: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_879, -1);  mul_879 = None
        unsqueeze_2347: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2346, -1);  unsqueeze_2346 = None
        sub_293: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_293, unsqueeze_2345);  convolution_293 = unsqueeze_2345 = None
        mul_880: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_2347);  sub_293 = unsqueeze_2347 = None
        unsqueeze_2348: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg724_1, -1);  arg724_1 = None
        unsqueeze_2349: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2348, -1);  unsqueeze_2348 = None
        mul_881: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_880, unsqueeze_2349);  mul_880 = unsqueeze_2349 = None
        unsqueeze_2350: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg725_1, -1);  arg725_1 = None
        unsqueeze_2351: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2350, -1);  unsqueeze_2350 = None
        add_759: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_881, unsqueeze_2351);  mul_881 = unsqueeze_2351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_285: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_759);  add_759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_284 = torch.ops.aten.split.Tensor(relu_281, 112, 1)
        getitem_2280: "f32[8, 112, 7, 7]" = split_284[4];  split_284 = None
        add_760: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_285, getitem_2280);  getitem_2280 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_294: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_760, arg726_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_760 = arg726_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_761: "f32[112]" = torch.ops.aten.add.Tensor(arg728_1, 1e-05);  arg728_1 = None
        sqrt_294: "f32[112]" = torch.ops.aten.sqrt.default(add_761);  add_761 = None
        reciprocal_294: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_294);  sqrt_294 = None
        mul_882: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_294, 1);  reciprocal_294 = None
        unsqueeze_2352: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg727_1, -1);  arg727_1 = None
        unsqueeze_2353: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2352, -1);  unsqueeze_2352 = None
        unsqueeze_2354: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_882, -1);  mul_882 = None
        unsqueeze_2355: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2354, -1);  unsqueeze_2354 = None
        sub_294: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_294, unsqueeze_2353);  convolution_294 = unsqueeze_2353 = None
        mul_883: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_2355);  sub_294 = unsqueeze_2355 = None
        unsqueeze_2356: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg729_1, -1);  arg729_1 = None
        unsqueeze_2357: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2356, -1);  unsqueeze_2356 = None
        mul_884: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_883, unsqueeze_2357);  mul_883 = unsqueeze_2357 = None
        unsqueeze_2358: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg730_1, -1);  arg730_1 = None
        unsqueeze_2359: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2358, -1);  unsqueeze_2358 = None
        add_762: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_884, unsqueeze_2359);  mul_884 = unsqueeze_2359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_286: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_762);  add_762 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_285 = torch.ops.aten.split.Tensor(relu_281, 112, 1)
        getitem_2289: "f32[8, 112, 7, 7]" = split_285[5];  split_285 = None
        add_763: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_286, getitem_2289);  getitem_2289 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_295: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_763, arg731_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_763 = arg731_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_764: "f32[112]" = torch.ops.aten.add.Tensor(arg733_1, 1e-05);  arg733_1 = None
        sqrt_295: "f32[112]" = torch.ops.aten.sqrt.default(add_764);  add_764 = None
        reciprocal_295: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_295);  sqrt_295 = None
        mul_885: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_295, 1);  reciprocal_295 = None
        unsqueeze_2360: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg732_1, -1);  arg732_1 = None
        unsqueeze_2361: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2360, -1);  unsqueeze_2360 = None
        unsqueeze_2362: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_885, -1);  mul_885 = None
        unsqueeze_2363: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2362, -1);  unsqueeze_2362 = None
        sub_295: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_295, unsqueeze_2361);  convolution_295 = unsqueeze_2361 = None
        mul_886: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_2363);  sub_295 = unsqueeze_2363 = None
        unsqueeze_2364: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg734_1, -1);  arg734_1 = None
        unsqueeze_2365: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2364, -1);  unsqueeze_2364 = None
        mul_887: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_886, unsqueeze_2365);  mul_886 = unsqueeze_2365 = None
        unsqueeze_2366: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg735_1, -1);  arg735_1 = None
        unsqueeze_2367: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2366, -1);  unsqueeze_2366 = None
        add_765: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_887, unsqueeze_2367);  mul_887 = unsqueeze_2367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_287: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_765);  add_765 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:92 in forward, code: sp = sp + spx[i]
        split_286 = torch.ops.aten.split.Tensor(relu_281, 112, 1)
        getitem_2298: "f32[8, 112, 7, 7]" = split_286[6];  split_286 = None
        add_766: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(relu_287, getitem_2298);  getitem_2298 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:93 in forward, code: sp = conv(sp)
        convolution_296: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_766, arg736_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  add_766 = arg736_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:94 in forward, code: sp = bn(sp)
        add_767: "f32[112]" = torch.ops.aten.add.Tensor(arg738_1, 1e-05);  arg738_1 = None
        sqrt_296: "f32[112]" = torch.ops.aten.sqrt.default(add_767);  add_767 = None
        reciprocal_296: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_296);  sqrt_296 = None
        mul_888: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_296, 1);  reciprocal_296 = None
        unsqueeze_2368: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg737_1, -1);  arg737_1 = None
        unsqueeze_2369: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2368, -1);  unsqueeze_2368 = None
        unsqueeze_2370: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_888, -1);  mul_888 = None
        unsqueeze_2371: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2370, -1);  unsqueeze_2370 = None
        sub_296: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_296, unsqueeze_2369);  convolution_296 = unsqueeze_2369 = None
        mul_889: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_2371);  sub_296 = unsqueeze_2371 = None
        unsqueeze_2372: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg739_1, -1);  arg739_1 = None
        unsqueeze_2373: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2372, -1);  unsqueeze_2372 = None
        mul_890: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_889, unsqueeze_2373);  mul_889 = unsqueeze_2373 = None
        unsqueeze_2374: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg740_1, -1);  arg740_1 = None
        unsqueeze_2375: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2374, -1);  unsqueeze_2374 = None
        add_768: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_890, unsqueeze_2375);  mul_890 = unsqueeze_2375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:95 in forward, code: sp = self.relu(sp)
        relu_288: "f32[8, 112, 7, 7]" = torch.ops.aten.relu.default(add_768);  add_768 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:102 in forward, code: out = torch.cat(spo, 1)
        split_287 = torch.ops.aten.split.Tensor(relu_281, 112, 1);  relu_281 = None
        getitem_2307: "f32[8, 112, 7, 7]" = split_287[7];  split_287 = None
        cat_31: "f32[8, 896, 7, 7]" = torch.ops.aten.cat.default([relu_282, relu_283, relu_284, relu_285, relu_286, relu_287, relu_288, getitem_2307], 1);  relu_282 = relu_283 = relu_284 = relu_285 = relu_286 = relu_287 = relu_288 = getitem_2307 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:104 in forward, code: out = self.conv3(out)
        convolution_297: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_31, arg741_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_31 = arg741_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:105 in forward, code: out = self.bn3(out)
        add_769: "f32[2048]" = torch.ops.aten.add.Tensor(arg743_1, 1e-05);  arg743_1 = None
        sqrt_297: "f32[2048]" = torch.ops.aten.sqrt.default(add_769);  add_769 = None
        reciprocal_297: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_297);  sqrt_297 = None
        mul_891: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_297, 1);  reciprocal_297 = None
        unsqueeze_2376: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg742_1, -1);  arg742_1 = None
        unsqueeze_2377: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2376, -1);  unsqueeze_2376 = None
        unsqueeze_2378: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_891, -1);  mul_891 = None
        unsqueeze_2379: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2378, -1);  unsqueeze_2378 = None
        sub_297: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_297, unsqueeze_2377);  convolution_297 = unsqueeze_2377 = None
        mul_892: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_2379);  sub_297 = unsqueeze_2379 = None
        unsqueeze_2380: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg744_1, -1);  arg744_1 = None
        unsqueeze_2381: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2380, -1);  unsqueeze_2380 = None
        mul_893: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_892, unsqueeze_2381);  mul_892 = unsqueeze_2381 = None
        unsqueeze_2382: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg745_1, -1);  arg745_1 = None
        unsqueeze_2383: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2382, -1);  unsqueeze_2382 = None
        add_770: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_893, unsqueeze_2383);  mul_893 = unsqueeze_2383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:113 in forward, code: out += shortcut
        add_771: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_770, relu_280);  add_770 = relu_280 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/res2net.py:114 in forward, code: out = self.relu(out)
        relu_289: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_771);  add_771 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_1: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_289, [-1, -2], True);  relu_289 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_1: "f32[8, 2048]" = torch.ops.aten.view.default(mean_1, [8, 2048]);  mean_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:633 in forward_head, code: return x if pre_logits else self.fc(x)
        permute_1: "f32[2048, 1000]" = torch.ops.aten.permute.default(arg746_1, [1, 0]);  arg746_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg747_1, view_1, permute_1);  arg747_1 = view_1 = permute_1 = None
        return (addmm_1,)
        