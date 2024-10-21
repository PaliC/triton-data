class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 3, 3, 3]", arg1_1: "f32[8, 3, 256, 256]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64, 64, 3, 3]", arg7_1: "f32[64]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[128, 64, 3, 3]", arg12_1: "f32[128]", arg13_1: "f32[128]", arg14_1: "f32[128]", arg15_1: "f32[128]", arg16_1: "f32[64, 128, 1, 1]", arg17_1: "f32[64]", arg18_1: "f32[64]", arg19_1: "f32[64]", arg20_1: "f32[64]", arg21_1: "f32[128, 32, 3, 3]", arg22_1: "f32[128]", arg23_1: "f32[128]", arg24_1: "f32[128]", arg25_1: "f32[128]", arg26_1: "f32[32, 64, 1, 1]", arg27_1: "f32[32]", arg28_1: "f32[32]", arg29_1: "f32[32]", arg30_1: "f32[32]", arg31_1: "f32[32]", arg32_1: "f32[128, 32, 1, 1]", arg33_1: "f32[128]", arg34_1: "f32[256, 64, 1, 1]", arg35_1: "f32[256]", arg36_1: "f32[256]", arg37_1: "f32[256]", arg38_1: "f32[256]", arg39_1: "f32[256, 128, 1, 1]", arg40_1: "f32[256]", arg41_1: "f32[256]", arg42_1: "f32[256]", arg43_1: "f32[256]", arg44_1: "f32[64, 256, 1, 1]", arg45_1: "f32[64]", arg46_1: "f32[64]", arg47_1: "f32[64]", arg48_1: "f32[64]", arg49_1: "f32[128, 32, 3, 3]", arg50_1: "f32[128]", arg51_1: "f32[128]", arg52_1: "f32[128]", arg53_1: "f32[128]", arg54_1: "f32[32, 64, 1, 1]", arg55_1: "f32[32]", arg56_1: "f32[32]", arg57_1: "f32[32]", arg58_1: "f32[32]", arg59_1: "f32[32]", arg60_1: "f32[128, 32, 1, 1]", arg61_1: "f32[128]", arg62_1: "f32[256, 64, 1, 1]", arg63_1: "f32[256]", arg64_1: "f32[256]", arg65_1: "f32[256]", arg66_1: "f32[256]", arg67_1: "f32[64, 256, 1, 1]", arg68_1: "f32[64]", arg69_1: "f32[64]", arg70_1: "f32[64]", arg71_1: "f32[64]", arg72_1: "f32[128, 32, 3, 3]", arg73_1: "f32[128]", arg74_1: "f32[128]", arg75_1: "f32[128]", arg76_1: "f32[128]", arg77_1: "f32[32, 64, 1, 1]", arg78_1: "f32[32]", arg79_1: "f32[32]", arg80_1: "f32[32]", arg81_1: "f32[32]", arg82_1: "f32[32]", arg83_1: "f32[128, 32, 1, 1]", arg84_1: "f32[128]", arg85_1: "f32[256, 64, 1, 1]", arg86_1: "f32[256]", arg87_1: "f32[256]", arg88_1: "f32[256]", arg89_1: "f32[256]", arg90_1: "f32[128, 256, 1, 1]", arg91_1: "f32[128]", arg92_1: "f32[128]", arg93_1: "f32[128]", arg94_1: "f32[128]", arg95_1: "f32[256, 64, 3, 3]", arg96_1: "f32[256]", arg97_1: "f32[256]", arg98_1: "f32[256]", arg99_1: "f32[256]", arg100_1: "f32[64, 128, 1, 1]", arg101_1: "f32[64]", arg102_1: "f32[64]", arg103_1: "f32[64]", arg104_1: "f32[64]", arg105_1: "f32[64]", arg106_1: "f32[256, 64, 1, 1]", arg107_1: "f32[256]", arg108_1: "f32[512, 128, 1, 1]", arg109_1: "f32[512]", arg110_1: "f32[512]", arg111_1: "f32[512]", arg112_1: "f32[512]", arg113_1: "f32[512, 256, 1, 1]", arg114_1: "f32[512]", arg115_1: "f32[512]", arg116_1: "f32[512]", arg117_1: "f32[512]", arg118_1: "f32[128, 512, 1, 1]", arg119_1: "f32[128]", arg120_1: "f32[128]", arg121_1: "f32[128]", arg122_1: "f32[128]", arg123_1: "f32[256, 64, 3, 3]", arg124_1: "f32[256]", arg125_1: "f32[256]", arg126_1: "f32[256]", arg127_1: "f32[256]", arg128_1: "f32[64, 128, 1, 1]", arg129_1: "f32[64]", arg130_1: "f32[64]", arg131_1: "f32[64]", arg132_1: "f32[64]", arg133_1: "f32[64]", arg134_1: "f32[256, 64, 1, 1]", arg135_1: "f32[256]", arg136_1: "f32[512, 128, 1, 1]", arg137_1: "f32[512]", arg138_1: "f32[512]", arg139_1: "f32[512]", arg140_1: "f32[512]", arg141_1: "f32[128, 512, 1, 1]", arg142_1: "f32[128]", arg143_1: "f32[128]", arg144_1: "f32[128]", arg145_1: "f32[128]", arg146_1: "f32[256, 64, 3, 3]", arg147_1: "f32[256]", arg148_1: "f32[256]", arg149_1: "f32[256]", arg150_1: "f32[256]", arg151_1: "f32[64, 128, 1, 1]", arg152_1: "f32[64]", arg153_1: "f32[64]", arg154_1: "f32[64]", arg155_1: "f32[64]", arg156_1: "f32[64]", arg157_1: "f32[256, 64, 1, 1]", arg158_1: "f32[256]", arg159_1: "f32[512, 128, 1, 1]", arg160_1: "f32[512]", arg161_1: "f32[512]", arg162_1: "f32[512]", arg163_1: "f32[512]", arg164_1: "f32[128, 512, 1, 1]", arg165_1: "f32[128]", arg166_1: "f32[128]", arg167_1: "f32[128]", arg168_1: "f32[128]", arg169_1: "f32[256, 64, 3, 3]", arg170_1: "f32[256]", arg171_1: "f32[256]", arg172_1: "f32[256]", arg173_1: "f32[256]", arg174_1: "f32[64, 128, 1, 1]", arg175_1: "f32[64]", arg176_1: "f32[64]", arg177_1: "f32[64]", arg178_1: "f32[64]", arg179_1: "f32[64]", arg180_1: "f32[256, 64, 1, 1]", arg181_1: "f32[256]", arg182_1: "f32[512, 128, 1, 1]", arg183_1: "f32[512]", arg184_1: "f32[512]", arg185_1: "f32[512]", arg186_1: "f32[512]", arg187_1: "f32[256, 512, 1, 1]", arg188_1: "f32[256]", arg189_1: "f32[256]", arg190_1: "f32[256]", arg191_1: "f32[256]", arg192_1: "f32[512, 128, 3, 3]", arg193_1: "f32[512]", arg194_1: "f32[512]", arg195_1: "f32[512]", arg196_1: "f32[512]", arg197_1: "f32[128, 256, 1, 1]", arg198_1: "f32[128]", arg199_1: "f32[128]", arg200_1: "f32[128]", arg201_1: "f32[128]", arg202_1: "f32[128]", arg203_1: "f32[512, 128, 1, 1]", arg204_1: "f32[512]", arg205_1: "f32[1024, 256, 1, 1]", arg206_1: "f32[1024]", arg207_1: "f32[1024]", arg208_1: "f32[1024]", arg209_1: "f32[1024]", arg210_1: "f32[1024, 512, 1, 1]", arg211_1: "f32[1024]", arg212_1: "f32[1024]", arg213_1: "f32[1024]", arg214_1: "f32[1024]", arg215_1: "f32[256, 1024, 1, 1]", arg216_1: "f32[256]", arg217_1: "f32[256]", arg218_1: "f32[256]", arg219_1: "f32[256]", arg220_1: "f32[512, 128, 3, 3]", arg221_1: "f32[512]", arg222_1: "f32[512]", arg223_1: "f32[512]", arg224_1: "f32[512]", arg225_1: "f32[128, 256, 1, 1]", arg226_1: "f32[128]", arg227_1: "f32[128]", arg228_1: "f32[128]", arg229_1: "f32[128]", arg230_1: "f32[128]", arg231_1: "f32[512, 128, 1, 1]", arg232_1: "f32[512]", arg233_1: "f32[1024, 256, 1, 1]", arg234_1: "f32[1024]", arg235_1: "f32[1024]", arg236_1: "f32[1024]", arg237_1: "f32[1024]", arg238_1: "f32[256, 1024, 1, 1]", arg239_1: "f32[256]", arg240_1: "f32[256]", arg241_1: "f32[256]", arg242_1: "f32[256]", arg243_1: "f32[512, 128, 3, 3]", arg244_1: "f32[512]", arg245_1: "f32[512]", arg246_1: "f32[512]", arg247_1: "f32[512]", arg248_1: "f32[128, 256, 1, 1]", arg249_1: "f32[128]", arg250_1: "f32[128]", arg251_1: "f32[128]", arg252_1: "f32[128]", arg253_1: "f32[128]", arg254_1: "f32[512, 128, 1, 1]", arg255_1: "f32[512]", arg256_1: "f32[1024, 256, 1, 1]", arg257_1: "f32[1024]", arg258_1: "f32[1024]", arg259_1: "f32[1024]", arg260_1: "f32[1024]", arg261_1: "f32[256, 1024, 1, 1]", arg262_1: "f32[256]", arg263_1: "f32[256]", arg264_1: "f32[256]", arg265_1: "f32[256]", arg266_1: "f32[512, 128, 3, 3]", arg267_1: "f32[512]", arg268_1: "f32[512]", arg269_1: "f32[512]", arg270_1: "f32[512]", arg271_1: "f32[128, 256, 1, 1]", arg272_1: "f32[128]", arg273_1: "f32[128]", arg274_1: "f32[128]", arg275_1: "f32[128]", arg276_1: "f32[128]", arg277_1: "f32[512, 128, 1, 1]", arg278_1: "f32[512]", arg279_1: "f32[1024, 256, 1, 1]", arg280_1: "f32[1024]", arg281_1: "f32[1024]", arg282_1: "f32[1024]", arg283_1: "f32[1024]", arg284_1: "f32[256, 1024, 1, 1]", arg285_1: "f32[256]", arg286_1: "f32[256]", arg287_1: "f32[256]", arg288_1: "f32[256]", arg289_1: "f32[512, 128, 3, 3]", arg290_1: "f32[512]", arg291_1: "f32[512]", arg292_1: "f32[512]", arg293_1: "f32[512]", arg294_1: "f32[128, 256, 1, 1]", arg295_1: "f32[128]", arg296_1: "f32[128]", arg297_1: "f32[128]", arg298_1: "f32[128]", arg299_1: "f32[128]", arg300_1: "f32[512, 128, 1, 1]", arg301_1: "f32[512]", arg302_1: "f32[1024, 256, 1, 1]", arg303_1: "f32[1024]", arg304_1: "f32[1024]", arg305_1: "f32[1024]", arg306_1: "f32[1024]", arg307_1: "f32[256, 1024, 1, 1]", arg308_1: "f32[256]", arg309_1: "f32[256]", arg310_1: "f32[256]", arg311_1: "f32[256]", arg312_1: "f32[512, 128, 3, 3]", arg313_1: "f32[512]", arg314_1: "f32[512]", arg315_1: "f32[512]", arg316_1: "f32[512]", arg317_1: "f32[128, 256, 1, 1]", arg318_1: "f32[128]", arg319_1: "f32[128]", arg320_1: "f32[128]", arg321_1: "f32[128]", arg322_1: "f32[128]", arg323_1: "f32[512, 128, 1, 1]", arg324_1: "f32[512]", arg325_1: "f32[1024, 256, 1, 1]", arg326_1: "f32[1024]", arg327_1: "f32[1024]", arg328_1: "f32[1024]", arg329_1: "f32[1024]", arg330_1: "f32[256, 1024, 1, 1]", arg331_1: "f32[256]", arg332_1: "f32[256]", arg333_1: "f32[256]", arg334_1: "f32[256]", arg335_1: "f32[512, 128, 3, 3]", arg336_1: "f32[512]", arg337_1: "f32[512]", arg338_1: "f32[512]", arg339_1: "f32[512]", arg340_1: "f32[128, 256, 1, 1]", arg341_1: "f32[128]", arg342_1: "f32[128]", arg343_1: "f32[128]", arg344_1: "f32[128]", arg345_1: "f32[128]", arg346_1: "f32[512, 128, 1, 1]", arg347_1: "f32[512]", arg348_1: "f32[1024, 256, 1, 1]", arg349_1: "f32[1024]", arg350_1: "f32[1024]", arg351_1: "f32[1024]", arg352_1: "f32[1024]", arg353_1: "f32[256, 1024, 1, 1]", arg354_1: "f32[256]", arg355_1: "f32[256]", arg356_1: "f32[256]", arg357_1: "f32[256]", arg358_1: "f32[512, 128, 3, 3]", arg359_1: "f32[512]", arg360_1: "f32[512]", arg361_1: "f32[512]", arg362_1: "f32[512]", arg363_1: "f32[128, 256, 1, 1]", arg364_1: "f32[128]", arg365_1: "f32[128]", arg366_1: "f32[128]", arg367_1: "f32[128]", arg368_1: "f32[128]", arg369_1: "f32[512, 128, 1, 1]", arg370_1: "f32[512]", arg371_1: "f32[1024, 256, 1, 1]", arg372_1: "f32[1024]", arg373_1: "f32[1024]", arg374_1: "f32[1024]", arg375_1: "f32[1024]", arg376_1: "f32[256, 1024, 1, 1]", arg377_1: "f32[256]", arg378_1: "f32[256]", arg379_1: "f32[256]", arg380_1: "f32[256]", arg381_1: "f32[512, 128, 3, 3]", arg382_1: "f32[512]", arg383_1: "f32[512]", arg384_1: "f32[512]", arg385_1: "f32[512]", arg386_1: "f32[128, 256, 1, 1]", arg387_1: "f32[128]", arg388_1: "f32[128]", arg389_1: "f32[128]", arg390_1: "f32[128]", arg391_1: "f32[128]", arg392_1: "f32[512, 128, 1, 1]", arg393_1: "f32[512]", arg394_1: "f32[1024, 256, 1, 1]", arg395_1: "f32[1024]", arg396_1: "f32[1024]", arg397_1: "f32[1024]", arg398_1: "f32[1024]", arg399_1: "f32[256, 1024, 1, 1]", arg400_1: "f32[256]", arg401_1: "f32[256]", arg402_1: "f32[256]", arg403_1: "f32[256]", arg404_1: "f32[512, 128, 3, 3]", arg405_1: "f32[512]", arg406_1: "f32[512]", arg407_1: "f32[512]", arg408_1: "f32[512]", arg409_1: "f32[128, 256, 1, 1]", arg410_1: "f32[128]", arg411_1: "f32[128]", arg412_1: "f32[128]", arg413_1: "f32[128]", arg414_1: "f32[128]", arg415_1: "f32[512, 128, 1, 1]", arg416_1: "f32[512]", arg417_1: "f32[1024, 256, 1, 1]", arg418_1: "f32[1024]", arg419_1: "f32[1024]", arg420_1: "f32[1024]", arg421_1: "f32[1024]", arg422_1: "f32[256, 1024, 1, 1]", arg423_1: "f32[256]", arg424_1: "f32[256]", arg425_1: "f32[256]", arg426_1: "f32[256]", arg427_1: "f32[512, 128, 3, 3]", arg428_1: "f32[512]", arg429_1: "f32[512]", arg430_1: "f32[512]", arg431_1: "f32[512]", arg432_1: "f32[128, 256, 1, 1]", arg433_1: "f32[128]", arg434_1: "f32[128]", arg435_1: "f32[128]", arg436_1: "f32[128]", arg437_1: "f32[128]", arg438_1: "f32[512, 128, 1, 1]", arg439_1: "f32[512]", arg440_1: "f32[1024, 256, 1, 1]", arg441_1: "f32[1024]", arg442_1: "f32[1024]", arg443_1: "f32[1024]", arg444_1: "f32[1024]", arg445_1: "f32[256, 1024, 1, 1]", arg446_1: "f32[256]", arg447_1: "f32[256]", arg448_1: "f32[256]", arg449_1: "f32[256]", arg450_1: "f32[512, 128, 3, 3]", arg451_1: "f32[512]", arg452_1: "f32[512]", arg453_1: "f32[512]", arg454_1: "f32[512]", arg455_1: "f32[128, 256, 1, 1]", arg456_1: "f32[128]", arg457_1: "f32[128]", arg458_1: "f32[128]", arg459_1: "f32[128]", arg460_1: "f32[128]", arg461_1: "f32[512, 128, 1, 1]", arg462_1: "f32[512]", arg463_1: "f32[1024, 256, 1, 1]", arg464_1: "f32[1024]", arg465_1: "f32[1024]", arg466_1: "f32[1024]", arg467_1: "f32[1024]", arg468_1: "f32[256, 1024, 1, 1]", arg469_1: "f32[256]", arg470_1: "f32[256]", arg471_1: "f32[256]", arg472_1: "f32[256]", arg473_1: "f32[512, 128, 3, 3]", arg474_1: "f32[512]", arg475_1: "f32[512]", arg476_1: "f32[512]", arg477_1: "f32[512]", arg478_1: "f32[128, 256, 1, 1]", arg479_1: "f32[128]", arg480_1: "f32[128]", arg481_1: "f32[128]", arg482_1: "f32[128]", arg483_1: "f32[128]", arg484_1: "f32[512, 128, 1, 1]", arg485_1: "f32[512]", arg486_1: "f32[1024, 256, 1, 1]", arg487_1: "f32[1024]", arg488_1: "f32[1024]", arg489_1: "f32[1024]", arg490_1: "f32[1024]", arg491_1: "f32[256, 1024, 1, 1]", arg492_1: "f32[256]", arg493_1: "f32[256]", arg494_1: "f32[256]", arg495_1: "f32[256]", arg496_1: "f32[512, 128, 3, 3]", arg497_1: "f32[512]", arg498_1: "f32[512]", arg499_1: "f32[512]", arg500_1: "f32[512]", arg501_1: "f32[128, 256, 1, 1]", arg502_1: "f32[128]", arg503_1: "f32[128]", arg504_1: "f32[128]", arg505_1: "f32[128]", arg506_1: "f32[128]", arg507_1: "f32[512, 128, 1, 1]", arg508_1: "f32[512]", arg509_1: "f32[1024, 256, 1, 1]", arg510_1: "f32[1024]", arg511_1: "f32[1024]", arg512_1: "f32[1024]", arg513_1: "f32[1024]", arg514_1: "f32[256, 1024, 1, 1]", arg515_1: "f32[256]", arg516_1: "f32[256]", arg517_1: "f32[256]", arg518_1: "f32[256]", arg519_1: "f32[512, 128, 3, 3]", arg520_1: "f32[512]", arg521_1: "f32[512]", arg522_1: "f32[512]", arg523_1: "f32[512]", arg524_1: "f32[128, 256, 1, 1]", arg525_1: "f32[128]", arg526_1: "f32[128]", arg527_1: "f32[128]", arg528_1: "f32[128]", arg529_1: "f32[128]", arg530_1: "f32[512, 128, 1, 1]", arg531_1: "f32[512]", arg532_1: "f32[1024, 256, 1, 1]", arg533_1: "f32[1024]", arg534_1: "f32[1024]", arg535_1: "f32[1024]", arg536_1: "f32[1024]", arg537_1: "f32[256, 1024, 1, 1]", arg538_1: "f32[256]", arg539_1: "f32[256]", arg540_1: "f32[256]", arg541_1: "f32[256]", arg542_1: "f32[512, 128, 3, 3]", arg543_1: "f32[512]", arg544_1: "f32[512]", arg545_1: "f32[512]", arg546_1: "f32[512]", arg547_1: "f32[128, 256, 1, 1]", arg548_1: "f32[128]", arg549_1: "f32[128]", arg550_1: "f32[128]", arg551_1: "f32[128]", arg552_1: "f32[128]", arg553_1: "f32[512, 128, 1, 1]", arg554_1: "f32[512]", arg555_1: "f32[1024, 256, 1, 1]", arg556_1: "f32[1024]", arg557_1: "f32[1024]", arg558_1: "f32[1024]", arg559_1: "f32[1024]", arg560_1: "f32[256, 1024, 1, 1]", arg561_1: "f32[256]", arg562_1: "f32[256]", arg563_1: "f32[256]", arg564_1: "f32[256]", arg565_1: "f32[512, 128, 3, 3]", arg566_1: "f32[512]", arg567_1: "f32[512]", arg568_1: "f32[512]", arg569_1: "f32[512]", arg570_1: "f32[128, 256, 1, 1]", arg571_1: "f32[128]", arg572_1: "f32[128]", arg573_1: "f32[128]", arg574_1: "f32[128]", arg575_1: "f32[128]", arg576_1: "f32[512, 128, 1, 1]", arg577_1: "f32[512]", arg578_1: "f32[1024, 256, 1, 1]", arg579_1: "f32[1024]", arg580_1: "f32[1024]", arg581_1: "f32[1024]", arg582_1: "f32[1024]", arg583_1: "f32[256, 1024, 1, 1]", arg584_1: "f32[256]", arg585_1: "f32[256]", arg586_1: "f32[256]", arg587_1: "f32[256]", arg588_1: "f32[512, 128, 3, 3]", arg589_1: "f32[512]", arg590_1: "f32[512]", arg591_1: "f32[512]", arg592_1: "f32[512]", arg593_1: "f32[128, 256, 1, 1]", arg594_1: "f32[128]", arg595_1: "f32[128]", arg596_1: "f32[128]", arg597_1: "f32[128]", arg598_1: "f32[128]", arg599_1: "f32[512, 128, 1, 1]", arg600_1: "f32[512]", arg601_1: "f32[1024, 256, 1, 1]", arg602_1: "f32[1024]", arg603_1: "f32[1024]", arg604_1: "f32[1024]", arg605_1: "f32[1024]", arg606_1: "f32[256, 1024, 1, 1]", arg607_1: "f32[256]", arg608_1: "f32[256]", arg609_1: "f32[256]", arg610_1: "f32[256]", arg611_1: "f32[512, 128, 3, 3]", arg612_1: "f32[512]", arg613_1: "f32[512]", arg614_1: "f32[512]", arg615_1: "f32[512]", arg616_1: "f32[128, 256, 1, 1]", arg617_1: "f32[128]", arg618_1: "f32[128]", arg619_1: "f32[128]", arg620_1: "f32[128]", arg621_1: "f32[128]", arg622_1: "f32[512, 128, 1, 1]", arg623_1: "f32[512]", arg624_1: "f32[1024, 256, 1, 1]", arg625_1: "f32[1024]", arg626_1: "f32[1024]", arg627_1: "f32[1024]", arg628_1: "f32[1024]", arg629_1: "f32[256, 1024, 1, 1]", arg630_1: "f32[256]", arg631_1: "f32[256]", arg632_1: "f32[256]", arg633_1: "f32[256]", arg634_1: "f32[512, 128, 3, 3]", arg635_1: "f32[512]", arg636_1: "f32[512]", arg637_1: "f32[512]", arg638_1: "f32[512]", arg639_1: "f32[128, 256, 1, 1]", arg640_1: "f32[128]", arg641_1: "f32[128]", arg642_1: "f32[128]", arg643_1: "f32[128]", arg644_1: "f32[128]", arg645_1: "f32[512, 128, 1, 1]", arg646_1: "f32[512]", arg647_1: "f32[1024, 256, 1, 1]", arg648_1: "f32[1024]", arg649_1: "f32[1024]", arg650_1: "f32[1024]", arg651_1: "f32[1024]", arg652_1: "f32[256, 1024, 1, 1]", arg653_1: "f32[256]", arg654_1: "f32[256]", arg655_1: "f32[256]", arg656_1: "f32[256]", arg657_1: "f32[512, 128, 3, 3]", arg658_1: "f32[512]", arg659_1: "f32[512]", arg660_1: "f32[512]", arg661_1: "f32[512]", arg662_1: "f32[128, 256, 1, 1]", arg663_1: "f32[128]", arg664_1: "f32[128]", arg665_1: "f32[128]", arg666_1: "f32[128]", arg667_1: "f32[128]", arg668_1: "f32[512, 128, 1, 1]", arg669_1: "f32[512]", arg670_1: "f32[1024, 256, 1, 1]", arg671_1: "f32[1024]", arg672_1: "f32[1024]", arg673_1: "f32[1024]", arg674_1: "f32[1024]", arg675_1: "f32[256, 1024, 1, 1]", arg676_1: "f32[256]", arg677_1: "f32[256]", arg678_1: "f32[256]", arg679_1: "f32[256]", arg680_1: "f32[512, 128, 3, 3]", arg681_1: "f32[512]", arg682_1: "f32[512]", arg683_1: "f32[512]", arg684_1: "f32[512]", arg685_1: "f32[128, 256, 1, 1]", arg686_1: "f32[128]", arg687_1: "f32[128]", arg688_1: "f32[128]", arg689_1: "f32[128]", arg690_1: "f32[128]", arg691_1: "f32[512, 128, 1, 1]", arg692_1: "f32[512]", arg693_1: "f32[1024, 256, 1, 1]", arg694_1: "f32[1024]", arg695_1: "f32[1024]", arg696_1: "f32[1024]", arg697_1: "f32[1024]", arg698_1: "f32[256, 1024, 1, 1]", arg699_1: "f32[256]", arg700_1: "f32[256]", arg701_1: "f32[256]", arg702_1: "f32[256]", arg703_1: "f32[512, 128, 3, 3]", arg704_1: "f32[512]", arg705_1: "f32[512]", arg706_1: "f32[512]", arg707_1: "f32[512]", arg708_1: "f32[128, 256, 1, 1]", arg709_1: "f32[128]", arg710_1: "f32[128]", arg711_1: "f32[128]", arg712_1: "f32[128]", arg713_1: "f32[128]", arg714_1: "f32[512, 128, 1, 1]", arg715_1: "f32[512]", arg716_1: "f32[1024, 256, 1, 1]", arg717_1: "f32[1024]", arg718_1: "f32[1024]", arg719_1: "f32[1024]", arg720_1: "f32[1024]", arg721_1: "f32[512, 1024, 1, 1]", arg722_1: "f32[512]", arg723_1: "f32[512]", arg724_1: "f32[512]", arg725_1: "f32[512]", arg726_1: "f32[1024, 256, 3, 3]", arg727_1: "f32[1024]", arg728_1: "f32[1024]", arg729_1: "f32[1024]", arg730_1: "f32[1024]", arg731_1: "f32[256, 512, 1, 1]", arg732_1: "f32[256]", arg733_1: "f32[256]", arg734_1: "f32[256]", arg735_1: "f32[256]", arg736_1: "f32[256]", arg737_1: "f32[1024, 256, 1, 1]", arg738_1: "f32[1024]", arg739_1: "f32[2048, 512, 1, 1]", arg740_1: "f32[2048]", arg741_1: "f32[2048]", arg742_1: "f32[2048]", arg743_1: "f32[2048]", arg744_1: "f32[2048, 1024, 1, 1]", arg745_1: "f32[2048]", arg746_1: "f32[2048]", arg747_1: "f32[2048]", arg748_1: "f32[2048]", arg749_1: "f32[512, 2048, 1, 1]", arg750_1: "f32[512]", arg751_1: "f32[512]", arg752_1: "f32[512]", arg753_1: "f32[512]", arg754_1: "f32[1024, 256, 3, 3]", arg755_1: "f32[1024]", arg756_1: "f32[1024]", arg757_1: "f32[1024]", arg758_1: "f32[1024]", arg759_1: "f32[256, 512, 1, 1]", arg760_1: "f32[256]", arg761_1: "f32[256]", arg762_1: "f32[256]", arg763_1: "f32[256]", arg764_1: "f32[256]", arg765_1: "f32[1024, 256, 1, 1]", arg766_1: "f32[1024]", arg767_1: "f32[2048, 512, 1, 1]", arg768_1: "f32[2048]", arg769_1: "f32[2048]", arg770_1: "f32[2048]", arg771_1: "f32[2048]", arg772_1: "f32[512, 2048, 1, 1]", arg773_1: "f32[512]", arg774_1: "f32[512]", arg775_1: "f32[512]", arg776_1: "f32[512]", arg777_1: "f32[1024, 256, 3, 3]", arg778_1: "f32[1024]", arg779_1: "f32[1024]", arg780_1: "f32[1024]", arg781_1: "f32[1024]", arg782_1: "f32[256, 512, 1, 1]", arg783_1: "f32[256]", arg784_1: "f32[256]", arg785_1: "f32[256]", arg786_1: "f32[256]", arg787_1: "f32[256]", arg788_1: "f32[1024, 256, 1, 1]", arg789_1: "f32[1024]", arg790_1: "f32[2048, 512, 1, 1]", arg791_1: "f32[2048]", arg792_1: "f32[2048]", arg793_1: "f32[2048]", arg794_1: "f32[2048]", arg795_1: "f32[1000, 2048]", arg796_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:615 in forward_features, code: x = self.conv1(x)
        convolution_172: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_311: "f32[64]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_139: "f32[64]" = torch.ops.aten.sqrt.default(add_311);  add_311 = None
        reciprocal_139: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_139);  sqrt_139 = None
        mul_450: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_139, 1);  reciprocal_139 = None
        unsqueeze_1112: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_1113: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, -1);  unsqueeze_1112 = None
        unsqueeze_1114: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_450, -1);  mul_450 = None
        unsqueeze_1115: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, -1);  unsqueeze_1114 = None
        sub_172: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_172, unsqueeze_1113);  convolution_172 = unsqueeze_1113 = None
        mul_451: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_1115);  sub_172 = unsqueeze_1115 = None
        unsqueeze_1116: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_1117: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, -1);  unsqueeze_1116 = None
        mul_452: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_451, unsqueeze_1117);  mul_451 = unsqueeze_1117 = None
        unsqueeze_1118: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_1119: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, -1);  unsqueeze_1118 = None
        add_312: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_452, unsqueeze_1119);  mul_452 = unsqueeze_1119 = None
        relu_135: "f32[8, 64, 128, 128]" = torch.ops.aten.relu.default(add_312);  add_312 = None
        convolution_173: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(relu_135, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_135 = arg6_1 = None
        add_313: "f32[64]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_140: "f32[64]" = torch.ops.aten.sqrt.default(add_313);  add_313 = None
        reciprocal_140: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_140);  sqrt_140 = None
        mul_453: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_140, 1);  reciprocal_140 = None
        unsqueeze_1120: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_1121: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, -1);  unsqueeze_1120 = None
        unsqueeze_1122: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_453, -1);  mul_453 = None
        unsqueeze_1123: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, -1);  unsqueeze_1122 = None
        sub_173: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_173, unsqueeze_1121);  convolution_173 = unsqueeze_1121 = None
        mul_454: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_1123);  sub_173 = unsqueeze_1123 = None
        unsqueeze_1124: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_1125: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, -1);  unsqueeze_1124 = None
        mul_455: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_454, unsqueeze_1125);  mul_454 = unsqueeze_1125 = None
        unsqueeze_1126: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_1127: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, -1);  unsqueeze_1126 = None
        add_314: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_455, unsqueeze_1127);  mul_455 = unsqueeze_1127 = None
        relu_136: "f32[8, 64, 128, 128]" = torch.ops.aten.relu.default(add_314);  add_314 = None
        convolution_174: "f32[8, 128, 128, 128]" = torch.ops.aten.convolution.default(relu_136, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_136 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:616 in forward_features, code: x = self.bn1(x)
        add_315: "f32[128]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_141: "f32[128]" = torch.ops.aten.sqrt.default(add_315);  add_315 = None
        reciprocal_141: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_141);  sqrt_141 = None
        mul_456: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_141, 1);  reciprocal_141 = None
        unsqueeze_1128: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_1129: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, -1);  unsqueeze_1128 = None
        unsqueeze_1130: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_456, -1);  mul_456 = None
        unsqueeze_1131: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, -1);  unsqueeze_1130 = None
        sub_174: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_174, unsqueeze_1129);  convolution_174 = unsqueeze_1129 = None
        mul_457: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_1131);  sub_174 = unsqueeze_1131 = None
        unsqueeze_1132: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_1133: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, -1);  unsqueeze_1132 = None
        mul_458: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_457, unsqueeze_1133);  mul_457 = unsqueeze_1133 = None
        unsqueeze_1134: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_1135: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, -1);  unsqueeze_1134 = None
        add_316: "f32[8, 128, 128, 128]" = torch.ops.aten.add.Tensor(mul_458, unsqueeze_1135);  mul_458 = unsqueeze_1135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:617 in forward_features, code: x = self.act1(x)
        relu_137: "f32[8, 128, 128, 128]" = torch.ops.aten.relu.default(add_316);  add_316 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:618 in forward_features, code: x = self.maxpool(x)
        _low_memory_max_pool2d_with_offsets_1 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_137, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu_137 = None
        getitem_2: "f32[8, 128, 64, 64]" = _low_memory_max_pool2d_with_offsets_1[0];  _low_memory_max_pool2d_with_offsets_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_175: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(getitem_2, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_317: "f32[64]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_142: "f32[64]" = torch.ops.aten.sqrt.default(add_317);  add_317 = None
        reciprocal_142: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_142);  sqrt_142 = None
        mul_459: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_142, 1);  reciprocal_142 = None
        unsqueeze_1136: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_1137: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, -1);  unsqueeze_1136 = None
        unsqueeze_1138: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_459, -1);  mul_459 = None
        unsqueeze_1139: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, -1);  unsqueeze_1138 = None
        sub_175: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_175, unsqueeze_1137);  convolution_175 = unsqueeze_1137 = None
        mul_460: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_1139);  sub_175 = unsqueeze_1139 = None
        unsqueeze_1140: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_1141: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, -1);  unsqueeze_1140 = None
        mul_461: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_460, unsqueeze_1141);  mul_460 = unsqueeze_1141 = None
        unsqueeze_1142: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_1143: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, -1);  unsqueeze_1142 = None
        add_318: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_1143);  mul_461 = unsqueeze_1143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_138: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_318);  add_318 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_176: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(relu_138, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_138 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_319: "f32[128]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_143: "f32[128]" = torch.ops.aten.sqrt.default(add_319);  add_319 = None
        reciprocal_143: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_143);  sqrt_143 = None
        mul_462: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_143, 1);  reciprocal_143 = None
        unsqueeze_1144: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_1145: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, -1);  unsqueeze_1144 = None
        unsqueeze_1146: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_462, -1);  mul_462 = None
        unsqueeze_1147: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, -1);  unsqueeze_1146 = None
        sub_176: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_176, unsqueeze_1145);  convolution_176 = unsqueeze_1145 = None
        mul_463: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_1147);  sub_176 = unsqueeze_1147 = None
        unsqueeze_1148: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_1149: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, -1);  unsqueeze_1148 = None
        mul_464: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_463, unsqueeze_1149);  mul_463 = unsqueeze_1149 = None
        unsqueeze_1150: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_1151: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, -1);  unsqueeze_1150 = None
        add_320: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_464, unsqueeze_1151);  mul_464 = unsqueeze_1151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_139: "f32[8, 128, 64, 64]" = torch.ops.aten.relu.default(add_320);  add_320 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_200: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.view.default(relu_139, [8, 2, 64, 64, 64]);  relu_139 = None
        sum_100: "f32[8, 64, 64, 64]" = torch.ops.aten.sum.dim_IntList(view_200, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_34: "f32[8, 64, 1, 1]" = torch.ops.aten.mean.dim(sum_100, [2, 3], True);  sum_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_177: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_34, arg26_1, arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_34 = arg26_1 = arg27_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_321: "f32[32]" = torch.ops.aten.add.Tensor(arg29_1, 1e-05);  arg29_1 = None
        sqrt_144: "f32[32]" = torch.ops.aten.sqrt.default(add_321);  add_321 = None
        reciprocal_144: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_144);  sqrt_144 = None
        mul_465: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_144, 1);  reciprocal_144 = None
        unsqueeze_1152: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_1153: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, -1);  unsqueeze_1152 = None
        unsqueeze_1154: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_465, -1);  mul_465 = None
        unsqueeze_1155: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, -1);  unsqueeze_1154 = None
        sub_177: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_177, unsqueeze_1153);  convolution_177 = unsqueeze_1153 = None
        mul_466: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_1155);  sub_177 = unsqueeze_1155 = None
        unsqueeze_1156: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_1157: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, -1);  unsqueeze_1156 = None
        mul_467: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(mul_466, unsqueeze_1157);  mul_466 = unsqueeze_1157 = None
        unsqueeze_1158: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_1159: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, -1);  unsqueeze_1158 = None
        add_322: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(mul_467, unsqueeze_1159);  mul_467 = unsqueeze_1159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_140: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(add_322);  add_322 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_178: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_140, arg32_1, arg33_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_140 = arg32_1 = arg33_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_201: "f32[8, 1, 2, 64]" = torch.ops.aten.view.default(convolution_178, [8, 1, 2, -1]);  convolution_178 = None
        permute_34: "f32[8, 2, 1, 64]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_33: "f32[8, 1, 1, 64]" = torch.ops.aten.amax.default(permute_34, [1], True)
        sub_178: "f32[8, 2, 1, 64]" = torch.ops.aten.sub.Tensor(permute_34, amax_33);  permute_34 = amax_33 = None
        exp_33: "f32[8, 2, 1, 64]" = torch.ops.aten.exp.default(sub_178);  sub_178 = None
        sum_101: "f32[8, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp_33, [1], True)
        div_33: "f32[8, 2, 1, 64]" = torch.ops.aten.div.Tensor(exp_33, sum_101);  exp_33 = sum_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_202: "f32[8, 128]" = torch.ops.aten.view.default(div_33, [8, -1]);  div_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_203: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(view_202, [8, -1, 1, 1]);  view_202 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_204: "f32[8, 2, 64, 1, 1]" = torch.ops.aten.view.default(view_203, [8, 2, 64, 1, 1]);  view_203 = None
        mul_468: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.mul.Tensor(view_200, view_204);  view_200 = view_204 = None
        sum_102: "f32[8, 64, 64, 64]" = torch.ops.aten.sum.dim_IntList(mul_468, [1]);  mul_468 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_179: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(sum_102, arg34_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_102 = arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_323: "f32[256]" = torch.ops.aten.add.Tensor(arg36_1, 1e-05);  arg36_1 = None
        sqrt_145: "f32[256]" = torch.ops.aten.sqrt.default(add_323);  add_323 = None
        reciprocal_145: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_145);  sqrt_145 = None
        mul_469: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_145, 1);  reciprocal_145 = None
        unsqueeze_1160: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_1161: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, -1);  unsqueeze_1160 = None
        unsqueeze_1162: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_469, -1);  mul_469 = None
        unsqueeze_1163: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, -1);  unsqueeze_1162 = None
        sub_179: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_179, unsqueeze_1161);  convolution_179 = unsqueeze_1161 = None
        mul_470: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_1163);  sub_179 = unsqueeze_1163 = None
        unsqueeze_1164: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_1165: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, -1);  unsqueeze_1164 = None
        mul_471: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_470, unsqueeze_1165);  mul_470 = unsqueeze_1165 = None
        unsqueeze_1166: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_1167: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, -1);  unsqueeze_1166 = None
        add_324: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_471, unsqueeze_1167);  mul_471 = unsqueeze_1167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:113 in forward, code: shortcut = self.downsample(x)
        convolution_180: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(getitem_2, arg39_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_2 = arg39_1 = None
        add_325: "f32[256]" = torch.ops.aten.add.Tensor(arg41_1, 1e-05);  arg41_1 = None
        sqrt_146: "f32[256]" = torch.ops.aten.sqrt.default(add_325);  add_325 = None
        reciprocal_146: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_146);  sqrt_146 = None
        mul_472: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_146, 1);  reciprocal_146 = None
        unsqueeze_1168: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_1169: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, -1);  unsqueeze_1168 = None
        unsqueeze_1170: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_472, -1);  mul_472 = None
        unsqueeze_1171: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, -1);  unsqueeze_1170 = None
        sub_180: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_180, unsqueeze_1169);  convolution_180 = unsqueeze_1169 = None
        mul_473: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_1171);  sub_180 = unsqueeze_1171 = None
        unsqueeze_1172: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_1173: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, -1);  unsqueeze_1172 = None
        mul_474: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_473, unsqueeze_1173);  mul_473 = unsqueeze_1173 = None
        unsqueeze_1174: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_1175: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, -1);  unsqueeze_1174 = None
        add_326: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_474, unsqueeze_1175);  mul_474 = unsqueeze_1175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_327: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_324, add_326);  add_324 = add_326 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_141: "f32[8, 256, 64, 64]" = torch.ops.aten.relu.default(add_327);  add_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_181: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(relu_141, arg44_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_328: "f32[64]" = torch.ops.aten.add.Tensor(arg46_1, 1e-05);  arg46_1 = None
        sqrt_147: "f32[64]" = torch.ops.aten.sqrt.default(add_328);  add_328 = None
        reciprocal_147: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_147);  sqrt_147 = None
        mul_475: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_147, 1);  reciprocal_147 = None
        unsqueeze_1176: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_1177: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, -1);  unsqueeze_1176 = None
        unsqueeze_1178: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_475, -1);  mul_475 = None
        unsqueeze_1179: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, -1);  unsqueeze_1178 = None
        sub_181: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_181, unsqueeze_1177);  convolution_181 = unsqueeze_1177 = None
        mul_476: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_1179);  sub_181 = unsqueeze_1179 = None
        unsqueeze_1180: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_1181: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, -1);  unsqueeze_1180 = None
        mul_477: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_1181);  mul_476 = unsqueeze_1181 = None
        unsqueeze_1182: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_1183: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, -1);  unsqueeze_1182 = None
        add_329: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_477, unsqueeze_1183);  mul_477 = unsqueeze_1183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_142: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_329);  add_329 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_182: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(relu_142, arg49_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_142 = arg49_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_330: "f32[128]" = torch.ops.aten.add.Tensor(arg51_1, 1e-05);  arg51_1 = None
        sqrt_148: "f32[128]" = torch.ops.aten.sqrt.default(add_330);  add_330 = None
        reciprocal_148: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_148);  sqrt_148 = None
        mul_478: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_148, 1);  reciprocal_148 = None
        unsqueeze_1184: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_1185: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, -1);  unsqueeze_1184 = None
        unsqueeze_1186: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_478, -1);  mul_478 = None
        unsqueeze_1187: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, -1);  unsqueeze_1186 = None
        sub_182: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_182, unsqueeze_1185);  convolution_182 = unsqueeze_1185 = None
        mul_479: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_1187);  sub_182 = unsqueeze_1187 = None
        unsqueeze_1188: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_1189: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, -1);  unsqueeze_1188 = None
        mul_480: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_479, unsqueeze_1189);  mul_479 = unsqueeze_1189 = None
        unsqueeze_1190: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_1191: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, -1);  unsqueeze_1190 = None
        add_331: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_480, unsqueeze_1191);  mul_480 = unsqueeze_1191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_143: "f32[8, 128, 64, 64]" = torch.ops.aten.relu.default(add_331);  add_331 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_206: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.view.default(relu_143, [8, 2, 64, 64, 64]);  relu_143 = None
        sum_103: "f32[8, 64, 64, 64]" = torch.ops.aten.sum.dim_IntList(view_206, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_35: "f32[8, 64, 1, 1]" = torch.ops.aten.mean.dim(sum_103, [2, 3], True);  sum_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_183: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_35, arg54_1, arg55_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_35 = arg54_1 = arg55_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_332: "f32[32]" = torch.ops.aten.add.Tensor(arg57_1, 1e-05);  arg57_1 = None
        sqrt_149: "f32[32]" = torch.ops.aten.sqrt.default(add_332);  add_332 = None
        reciprocal_149: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_149);  sqrt_149 = None
        mul_481: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_149, 1);  reciprocal_149 = None
        unsqueeze_1192: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_1193: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, -1);  unsqueeze_1192 = None
        unsqueeze_1194: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_481, -1);  mul_481 = None
        unsqueeze_1195: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, -1);  unsqueeze_1194 = None
        sub_183: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_183, unsqueeze_1193);  convolution_183 = unsqueeze_1193 = None
        mul_482: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_1195);  sub_183 = unsqueeze_1195 = None
        unsqueeze_1196: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_1197: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, -1);  unsqueeze_1196 = None
        mul_483: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(mul_482, unsqueeze_1197);  mul_482 = unsqueeze_1197 = None
        unsqueeze_1198: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_1199: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, -1);  unsqueeze_1198 = None
        add_333: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(mul_483, unsqueeze_1199);  mul_483 = unsqueeze_1199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_144: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(add_333);  add_333 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_184: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_144, arg60_1, arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_144 = arg60_1 = arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_207: "f32[8, 1, 2, 64]" = torch.ops.aten.view.default(convolution_184, [8, 1, 2, -1]);  convolution_184 = None
        permute_35: "f32[8, 2, 1, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_34: "f32[8, 1, 1, 64]" = torch.ops.aten.amax.default(permute_35, [1], True)
        sub_184: "f32[8, 2, 1, 64]" = torch.ops.aten.sub.Tensor(permute_35, amax_34);  permute_35 = amax_34 = None
        exp_34: "f32[8, 2, 1, 64]" = torch.ops.aten.exp.default(sub_184);  sub_184 = None
        sum_104: "f32[8, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp_34, [1], True)
        div_34: "f32[8, 2, 1, 64]" = torch.ops.aten.div.Tensor(exp_34, sum_104);  exp_34 = sum_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_208: "f32[8, 128]" = torch.ops.aten.view.default(div_34, [8, -1]);  div_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_209: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(view_208, [8, -1, 1, 1]);  view_208 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_210: "f32[8, 2, 64, 1, 1]" = torch.ops.aten.view.default(view_209, [8, 2, 64, 1, 1]);  view_209 = None
        mul_484: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.mul.Tensor(view_206, view_210);  view_206 = view_210 = None
        sum_105: "f32[8, 64, 64, 64]" = torch.ops.aten.sum.dim_IntList(mul_484, [1]);  mul_484 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_185: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(sum_105, arg62_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_105 = arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_334: "f32[256]" = torch.ops.aten.add.Tensor(arg64_1, 1e-05);  arg64_1 = None
        sqrt_150: "f32[256]" = torch.ops.aten.sqrt.default(add_334);  add_334 = None
        reciprocal_150: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_150);  sqrt_150 = None
        mul_485: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_150, 1);  reciprocal_150 = None
        unsqueeze_1200: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_1201: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, -1);  unsqueeze_1200 = None
        unsqueeze_1202: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_485, -1);  mul_485 = None
        unsqueeze_1203: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, -1);  unsqueeze_1202 = None
        sub_185: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_185, unsqueeze_1201);  convolution_185 = unsqueeze_1201 = None
        mul_486: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_1203);  sub_185 = unsqueeze_1203 = None
        unsqueeze_1204: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_1205: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, -1);  unsqueeze_1204 = None
        mul_487: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_486, unsqueeze_1205);  mul_486 = unsqueeze_1205 = None
        unsqueeze_1206: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_1207: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, -1);  unsqueeze_1206 = None
        add_335: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_487, unsqueeze_1207);  mul_487 = unsqueeze_1207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_336: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_335, relu_141);  add_335 = relu_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_145: "f32[8, 256, 64, 64]" = torch.ops.aten.relu.default(add_336);  add_336 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_186: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(relu_145, arg67_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg67_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_337: "f32[64]" = torch.ops.aten.add.Tensor(arg69_1, 1e-05);  arg69_1 = None
        sqrt_151: "f32[64]" = torch.ops.aten.sqrt.default(add_337);  add_337 = None
        reciprocal_151: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_151);  sqrt_151 = None
        mul_488: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_151, 1);  reciprocal_151 = None
        unsqueeze_1208: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_1209: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, -1);  unsqueeze_1208 = None
        unsqueeze_1210: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_488, -1);  mul_488 = None
        unsqueeze_1211: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, -1);  unsqueeze_1210 = None
        sub_186: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_186, unsqueeze_1209);  convolution_186 = unsqueeze_1209 = None
        mul_489: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_1211);  sub_186 = unsqueeze_1211 = None
        unsqueeze_1212: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_1213: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, -1);  unsqueeze_1212 = None
        mul_490: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_489, unsqueeze_1213);  mul_489 = unsqueeze_1213 = None
        unsqueeze_1214: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_1215: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, -1);  unsqueeze_1214 = None
        add_338: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_490, unsqueeze_1215);  mul_490 = unsqueeze_1215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_146: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_338);  add_338 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_187: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(relu_146, arg72_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_146 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_339: "f32[128]" = torch.ops.aten.add.Tensor(arg74_1, 1e-05);  arg74_1 = None
        sqrt_152: "f32[128]" = torch.ops.aten.sqrt.default(add_339);  add_339 = None
        reciprocal_152: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_152);  sqrt_152 = None
        mul_491: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_152, 1);  reciprocal_152 = None
        unsqueeze_1216: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_1217: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, -1);  unsqueeze_1216 = None
        unsqueeze_1218: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_491, -1);  mul_491 = None
        unsqueeze_1219: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, -1);  unsqueeze_1218 = None
        sub_187: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_187, unsqueeze_1217);  convolution_187 = unsqueeze_1217 = None
        mul_492: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_1219);  sub_187 = unsqueeze_1219 = None
        unsqueeze_1220: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_1221: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, -1);  unsqueeze_1220 = None
        mul_493: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_492, unsqueeze_1221);  mul_492 = unsqueeze_1221 = None
        unsqueeze_1222: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_1223: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, -1);  unsqueeze_1222 = None
        add_340: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_493, unsqueeze_1223);  mul_493 = unsqueeze_1223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_147: "f32[8, 128, 64, 64]" = torch.ops.aten.relu.default(add_340);  add_340 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_212: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.view.default(relu_147, [8, 2, 64, 64, 64]);  relu_147 = None
        sum_106: "f32[8, 64, 64, 64]" = torch.ops.aten.sum.dim_IntList(view_212, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_36: "f32[8, 64, 1, 1]" = torch.ops.aten.mean.dim(sum_106, [2, 3], True);  sum_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_188: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_36, arg77_1, arg78_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_36 = arg77_1 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_341: "f32[32]" = torch.ops.aten.add.Tensor(arg80_1, 1e-05);  arg80_1 = None
        sqrt_153: "f32[32]" = torch.ops.aten.sqrt.default(add_341);  add_341 = None
        reciprocal_153: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_153);  sqrt_153 = None
        mul_494: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_153, 1);  reciprocal_153 = None
        unsqueeze_1224: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_1225: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, -1);  unsqueeze_1224 = None
        unsqueeze_1226: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_494, -1);  mul_494 = None
        unsqueeze_1227: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, -1);  unsqueeze_1226 = None
        sub_188: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_188, unsqueeze_1225);  convolution_188 = unsqueeze_1225 = None
        mul_495: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_1227);  sub_188 = unsqueeze_1227 = None
        unsqueeze_1228: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_1229: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, -1);  unsqueeze_1228 = None
        mul_496: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(mul_495, unsqueeze_1229);  mul_495 = unsqueeze_1229 = None
        unsqueeze_1230: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_1231: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, -1);  unsqueeze_1230 = None
        add_342: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(mul_496, unsqueeze_1231);  mul_496 = unsqueeze_1231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_148: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(add_342);  add_342 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_189: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_148, arg83_1, arg84_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_148 = arg83_1 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_213: "f32[8, 1, 2, 64]" = torch.ops.aten.view.default(convolution_189, [8, 1, 2, -1]);  convolution_189 = None
        permute_36: "f32[8, 2, 1, 64]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_35: "f32[8, 1, 1, 64]" = torch.ops.aten.amax.default(permute_36, [1], True)
        sub_189: "f32[8, 2, 1, 64]" = torch.ops.aten.sub.Tensor(permute_36, amax_35);  permute_36 = amax_35 = None
        exp_35: "f32[8, 2, 1, 64]" = torch.ops.aten.exp.default(sub_189);  sub_189 = None
        sum_107: "f32[8, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp_35, [1], True)
        div_35: "f32[8, 2, 1, 64]" = torch.ops.aten.div.Tensor(exp_35, sum_107);  exp_35 = sum_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_214: "f32[8, 128]" = torch.ops.aten.view.default(div_35, [8, -1]);  div_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_215: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(view_214, [8, -1, 1, 1]);  view_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_216: "f32[8, 2, 64, 1, 1]" = torch.ops.aten.view.default(view_215, [8, 2, 64, 1, 1]);  view_215 = None
        mul_497: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.mul.Tensor(view_212, view_216);  view_212 = view_216 = None
        sum_108: "f32[8, 64, 64, 64]" = torch.ops.aten.sum.dim_IntList(mul_497, [1]);  mul_497 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_190: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(sum_108, arg85_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_108 = arg85_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_343: "f32[256]" = torch.ops.aten.add.Tensor(arg87_1, 1e-05);  arg87_1 = None
        sqrt_154: "f32[256]" = torch.ops.aten.sqrt.default(add_343);  add_343 = None
        reciprocal_154: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_154);  sqrt_154 = None
        mul_498: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_154, 1);  reciprocal_154 = None
        unsqueeze_1232: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_1233: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, -1);  unsqueeze_1232 = None
        unsqueeze_1234: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_498, -1);  mul_498 = None
        unsqueeze_1235: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, -1);  unsqueeze_1234 = None
        sub_190: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_190, unsqueeze_1233);  convolution_190 = unsqueeze_1233 = None
        mul_499: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_1235);  sub_190 = unsqueeze_1235 = None
        unsqueeze_1236: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_1237: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, -1);  unsqueeze_1236 = None
        mul_500: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_1237);  mul_499 = unsqueeze_1237 = None
        unsqueeze_1238: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_1239: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, -1);  unsqueeze_1238 = None
        add_344: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_500, unsqueeze_1239);  mul_500 = unsqueeze_1239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_345: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_344, relu_145);  add_344 = relu_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_149: "f32[8, 256, 64, 64]" = torch.ops.aten.relu.default(add_345);  add_345 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_191: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(relu_149, arg90_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_346: "f32[128]" = torch.ops.aten.add.Tensor(arg92_1, 1e-05);  arg92_1 = None
        sqrt_155: "f32[128]" = torch.ops.aten.sqrt.default(add_346);  add_346 = None
        reciprocal_155: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_155);  sqrt_155 = None
        mul_501: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_155, 1);  reciprocal_155 = None
        unsqueeze_1240: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_1241: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, -1);  unsqueeze_1240 = None
        unsqueeze_1242: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_501, -1);  mul_501 = None
        unsqueeze_1243: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, -1);  unsqueeze_1242 = None
        sub_191: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_191, unsqueeze_1241);  convolution_191 = unsqueeze_1241 = None
        mul_502: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_1243);  sub_191 = unsqueeze_1243 = None
        unsqueeze_1244: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_1245: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, -1);  unsqueeze_1244 = None
        mul_503: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_502, unsqueeze_1245);  mul_502 = unsqueeze_1245 = None
        unsqueeze_1246: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_1247: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, -1);  unsqueeze_1246 = None
        add_347: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_1247);  mul_503 = unsqueeze_1247 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_150: "f32[8, 128, 64, 64]" = torch.ops.aten.relu.default(add_347);  add_347 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_192: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(relu_150, arg95_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_150 = arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_348: "f32[256]" = torch.ops.aten.add.Tensor(arg97_1, 1e-05);  arg97_1 = None
        sqrt_156: "f32[256]" = torch.ops.aten.sqrt.default(add_348);  add_348 = None
        reciprocal_156: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_156);  sqrt_156 = None
        mul_504: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_156, 1);  reciprocal_156 = None
        unsqueeze_1248: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_1249: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, -1);  unsqueeze_1248 = None
        unsqueeze_1250: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_504, -1);  mul_504 = None
        unsqueeze_1251: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, -1);  unsqueeze_1250 = None
        sub_192: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_192, unsqueeze_1249);  convolution_192 = unsqueeze_1249 = None
        mul_505: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_1251);  sub_192 = unsqueeze_1251 = None
        unsqueeze_1252: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_1253: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, -1);  unsqueeze_1252 = None
        mul_506: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_505, unsqueeze_1253);  mul_505 = unsqueeze_1253 = None
        unsqueeze_1254: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_1255: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, -1);  unsqueeze_1254 = None
        add_349: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_506, unsqueeze_1255);  mul_506 = unsqueeze_1255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_151: "f32[8, 256, 64, 64]" = torch.ops.aten.relu.default(add_349);  add_349 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_218: "f32[8, 2, 128, 64, 64]" = torch.ops.aten.view.default(relu_151, [8, 2, 128, 64, 64]);  relu_151 = None
        sum_109: "f32[8, 128, 64, 64]" = torch.ops.aten.sum.dim_IntList(view_218, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_37: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(sum_109, [2, 3], True);  sum_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_193: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_37, arg100_1, arg101_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_37 = arg100_1 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_350: "f32[64]" = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_157: "f32[64]" = torch.ops.aten.sqrt.default(add_350);  add_350 = None
        reciprocal_157: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_157);  sqrt_157 = None
        mul_507: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_157, 1);  reciprocal_157 = None
        unsqueeze_1256: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_1257: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, -1);  unsqueeze_1256 = None
        unsqueeze_1258: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_507, -1);  mul_507 = None
        unsqueeze_1259: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, -1);  unsqueeze_1258 = None
        sub_193: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_193, unsqueeze_1257);  convolution_193 = unsqueeze_1257 = None
        mul_508: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_1259);  sub_193 = unsqueeze_1259 = None
        unsqueeze_1260: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_1261: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, -1);  unsqueeze_1260 = None
        mul_509: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_508, unsqueeze_1261);  mul_508 = unsqueeze_1261 = None
        unsqueeze_1262: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_1263: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, -1);  unsqueeze_1262 = None
        add_351: "f32[8, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_509, unsqueeze_1263);  mul_509 = unsqueeze_1263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_152: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(add_351);  add_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_194: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_152, arg106_1, arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_152 = arg106_1 = arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_219: "f32[8, 1, 2, 128]" = torch.ops.aten.view.default(convolution_194, [8, 1, 2, -1]);  convolution_194 = None
        permute_37: "f32[8, 2, 1, 128]" = torch.ops.aten.permute.default(view_219, [0, 2, 1, 3]);  view_219 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_36: "f32[8, 1, 1, 128]" = torch.ops.aten.amax.default(permute_37, [1], True)
        sub_194: "f32[8, 2, 1, 128]" = torch.ops.aten.sub.Tensor(permute_37, amax_36);  permute_37 = amax_36 = None
        exp_36: "f32[8, 2, 1, 128]" = torch.ops.aten.exp.default(sub_194);  sub_194 = None
        sum_110: "f32[8, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp_36, [1], True)
        div_36: "f32[8, 2, 1, 128]" = torch.ops.aten.div.Tensor(exp_36, sum_110);  exp_36 = sum_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_220: "f32[8, 256]" = torch.ops.aten.view.default(div_36, [8, -1]);  div_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_221: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(view_220, [8, -1, 1, 1]);  view_220 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_222: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.view.default(view_221, [8, 2, 128, 1, 1]);  view_221 = None
        mul_510: "f32[8, 2, 128, 64, 64]" = torch.ops.aten.mul.Tensor(view_218, view_222);  view_218 = view_222 = None
        sum_111: "f32[8, 128, 64, 64]" = torch.ops.aten.sum.dim_IntList(mul_510, [1]);  mul_510 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:107 in forward, code: out = self.avd_last(out)
        avg_pool2d_6: "f32[8, 128, 32, 32]" = torch.ops.aten.avg_pool2d.default(sum_111, [3, 3], [2, 2], [1, 1]);  sum_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_195: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(avg_pool2d_6, arg108_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_6 = arg108_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_352: "f32[512]" = torch.ops.aten.add.Tensor(arg110_1, 1e-05);  arg110_1 = None
        sqrt_158: "f32[512]" = torch.ops.aten.sqrt.default(add_352);  add_352 = None
        reciprocal_158: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_158);  sqrt_158 = None
        mul_511: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_158, 1);  reciprocal_158 = None
        unsqueeze_1264: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_1265: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, -1);  unsqueeze_1264 = None
        unsqueeze_1266: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_511, -1);  mul_511 = None
        unsqueeze_1267: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, -1);  unsqueeze_1266 = None
        sub_195: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_195, unsqueeze_1265);  convolution_195 = unsqueeze_1265 = None
        mul_512: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_1267);  sub_195 = unsqueeze_1267 = None
        unsqueeze_1268: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_1269: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, -1);  unsqueeze_1268 = None
        mul_513: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_512, unsqueeze_1269);  mul_512 = unsqueeze_1269 = None
        unsqueeze_1270: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_1271: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, -1);  unsqueeze_1270 = None
        add_353: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_513, unsqueeze_1271);  mul_513 = unsqueeze_1271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:113 in forward, code: shortcut = self.downsample(x)
        avg_pool2d_7: "f32[8, 256, 32, 32]" = torch.ops.aten.avg_pool2d.default(relu_149, [2, 2], [2, 2], [0, 0], True, False);  relu_149 = None
        convolution_196: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(avg_pool2d_7, arg113_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_7 = arg113_1 = None
        add_354: "f32[512]" = torch.ops.aten.add.Tensor(arg115_1, 1e-05);  arg115_1 = None
        sqrt_159: "f32[512]" = torch.ops.aten.sqrt.default(add_354);  add_354 = None
        reciprocal_159: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_159);  sqrt_159 = None
        mul_514: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_159, 1);  reciprocal_159 = None
        unsqueeze_1272: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_1273: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, -1);  unsqueeze_1272 = None
        unsqueeze_1274: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_514, -1);  mul_514 = None
        unsqueeze_1275: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, -1);  unsqueeze_1274 = None
        sub_196: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_196, unsqueeze_1273);  convolution_196 = unsqueeze_1273 = None
        mul_515: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_1275);  sub_196 = unsqueeze_1275 = None
        unsqueeze_1276: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_1277: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, -1);  unsqueeze_1276 = None
        mul_516: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_515, unsqueeze_1277);  mul_515 = unsqueeze_1277 = None
        unsqueeze_1278: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_1279: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, -1);  unsqueeze_1278 = None
        add_355: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_516, unsqueeze_1279);  mul_516 = unsqueeze_1279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_356: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_353, add_355);  add_353 = add_355 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_153: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_356);  add_356 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_197: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(relu_153, arg118_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_357: "f32[128]" = torch.ops.aten.add.Tensor(arg120_1, 1e-05);  arg120_1 = None
        sqrt_160: "f32[128]" = torch.ops.aten.sqrt.default(add_357);  add_357 = None
        reciprocal_160: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_160);  sqrt_160 = None
        mul_517: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_160, 1);  reciprocal_160 = None
        unsqueeze_1280: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_1281: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, -1);  unsqueeze_1280 = None
        unsqueeze_1282: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_517, -1);  mul_517 = None
        unsqueeze_1283: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, -1);  unsqueeze_1282 = None
        sub_197: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_197, unsqueeze_1281);  convolution_197 = unsqueeze_1281 = None
        mul_518: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_1283);  sub_197 = unsqueeze_1283 = None
        unsqueeze_1284: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_1285: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, -1);  unsqueeze_1284 = None
        mul_519: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_518, unsqueeze_1285);  mul_518 = unsqueeze_1285 = None
        unsqueeze_1286: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_1287: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, -1);  unsqueeze_1286 = None
        add_358: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_519, unsqueeze_1287);  mul_519 = unsqueeze_1287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_154: "f32[8, 128, 32, 32]" = torch.ops.aten.relu.default(add_358);  add_358 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_198: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(relu_154, arg123_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_154 = arg123_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_359: "f32[256]" = torch.ops.aten.add.Tensor(arg125_1, 1e-05);  arg125_1 = None
        sqrt_161: "f32[256]" = torch.ops.aten.sqrt.default(add_359);  add_359 = None
        reciprocal_161: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_161);  sqrt_161 = None
        mul_520: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_161, 1);  reciprocal_161 = None
        unsqueeze_1288: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_1289: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, -1);  unsqueeze_1288 = None
        unsqueeze_1290: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_520, -1);  mul_520 = None
        unsqueeze_1291: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, -1);  unsqueeze_1290 = None
        sub_198: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_198, unsqueeze_1289);  convolution_198 = unsqueeze_1289 = None
        mul_521: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_1291);  sub_198 = unsqueeze_1291 = None
        unsqueeze_1292: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_1293: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, -1);  unsqueeze_1292 = None
        mul_522: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_521, unsqueeze_1293);  mul_521 = unsqueeze_1293 = None
        unsqueeze_1294: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_1295: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, -1);  unsqueeze_1294 = None
        add_360: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_522, unsqueeze_1295);  mul_522 = unsqueeze_1295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_155: "f32[8, 256, 32, 32]" = torch.ops.aten.relu.default(add_360);  add_360 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_224: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.view.default(relu_155, [8, 2, 128, 32, 32]);  relu_155 = None
        sum_112: "f32[8, 128, 32, 32]" = torch.ops.aten.sum.dim_IntList(view_224, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_38: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(sum_112, [2, 3], True);  sum_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_199: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_38, arg128_1, arg129_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_38 = arg128_1 = arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_361: "f32[64]" = torch.ops.aten.add.Tensor(arg131_1, 1e-05);  arg131_1 = None
        sqrt_162: "f32[64]" = torch.ops.aten.sqrt.default(add_361);  add_361 = None
        reciprocal_162: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_162);  sqrt_162 = None
        mul_523: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_162, 1);  reciprocal_162 = None
        unsqueeze_1296: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_1297: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, -1);  unsqueeze_1296 = None
        unsqueeze_1298: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_523, -1);  mul_523 = None
        unsqueeze_1299: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, -1);  unsqueeze_1298 = None
        sub_199: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_199, unsqueeze_1297);  convolution_199 = unsqueeze_1297 = None
        mul_524: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_1299);  sub_199 = unsqueeze_1299 = None
        unsqueeze_1300: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_1301: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, -1);  unsqueeze_1300 = None
        mul_525: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_524, unsqueeze_1301);  mul_524 = unsqueeze_1301 = None
        unsqueeze_1302: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_1303: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, -1);  unsqueeze_1302 = None
        add_362: "f32[8, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_525, unsqueeze_1303);  mul_525 = unsqueeze_1303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_156: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(add_362);  add_362 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_200: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_156, arg134_1, arg135_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_156 = arg134_1 = arg135_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_225: "f32[8, 1, 2, 128]" = torch.ops.aten.view.default(convolution_200, [8, 1, 2, -1]);  convolution_200 = None
        permute_38: "f32[8, 2, 1, 128]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_37: "f32[8, 1, 1, 128]" = torch.ops.aten.amax.default(permute_38, [1], True)
        sub_200: "f32[8, 2, 1, 128]" = torch.ops.aten.sub.Tensor(permute_38, amax_37);  permute_38 = amax_37 = None
        exp_37: "f32[8, 2, 1, 128]" = torch.ops.aten.exp.default(sub_200);  sub_200 = None
        sum_113: "f32[8, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp_37, [1], True)
        div_37: "f32[8, 2, 1, 128]" = torch.ops.aten.div.Tensor(exp_37, sum_113);  exp_37 = sum_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_226: "f32[8, 256]" = torch.ops.aten.view.default(div_37, [8, -1]);  div_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_227: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(view_226, [8, -1, 1, 1]);  view_226 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_228: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.view.default(view_227, [8, 2, 128, 1, 1]);  view_227 = None
        mul_526: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.mul.Tensor(view_224, view_228);  view_224 = view_228 = None
        sum_114: "f32[8, 128, 32, 32]" = torch.ops.aten.sum.dim_IntList(mul_526, [1]);  mul_526 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_201: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(sum_114, arg136_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_114 = arg136_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_363: "f32[512]" = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
        sqrt_163: "f32[512]" = torch.ops.aten.sqrt.default(add_363);  add_363 = None
        reciprocal_163: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_163);  sqrt_163 = None
        mul_527: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_163, 1);  reciprocal_163 = None
        unsqueeze_1304: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_1305: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, -1);  unsqueeze_1304 = None
        unsqueeze_1306: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_527, -1);  mul_527 = None
        unsqueeze_1307: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, -1);  unsqueeze_1306 = None
        sub_201: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_201, unsqueeze_1305);  convolution_201 = unsqueeze_1305 = None
        mul_528: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_1307);  sub_201 = unsqueeze_1307 = None
        unsqueeze_1308: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_1309: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, -1);  unsqueeze_1308 = None
        mul_529: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_528, unsqueeze_1309);  mul_528 = unsqueeze_1309 = None
        unsqueeze_1310: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_1311: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, -1);  unsqueeze_1310 = None
        add_364: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_529, unsqueeze_1311);  mul_529 = unsqueeze_1311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_365: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_364, relu_153);  add_364 = relu_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_157: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_365);  add_365 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_202: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(relu_157, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg141_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_366: "f32[128]" = torch.ops.aten.add.Tensor(arg143_1, 1e-05);  arg143_1 = None
        sqrt_164: "f32[128]" = torch.ops.aten.sqrt.default(add_366);  add_366 = None
        reciprocal_164: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_164);  sqrt_164 = None
        mul_530: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_164, 1);  reciprocal_164 = None
        unsqueeze_1312: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_1313: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, -1);  unsqueeze_1312 = None
        unsqueeze_1314: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_530, -1);  mul_530 = None
        unsqueeze_1315: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, -1);  unsqueeze_1314 = None
        sub_202: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_202, unsqueeze_1313);  convolution_202 = unsqueeze_1313 = None
        mul_531: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_1315);  sub_202 = unsqueeze_1315 = None
        unsqueeze_1316: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_1317: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, -1);  unsqueeze_1316 = None
        mul_532: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_531, unsqueeze_1317);  mul_531 = unsqueeze_1317 = None
        unsqueeze_1318: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_1319: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, -1);  unsqueeze_1318 = None
        add_367: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_532, unsqueeze_1319);  mul_532 = unsqueeze_1319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_158: "f32[8, 128, 32, 32]" = torch.ops.aten.relu.default(add_367);  add_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_203: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(relu_158, arg146_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_158 = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_368: "f32[256]" = torch.ops.aten.add.Tensor(arg148_1, 1e-05);  arg148_1 = None
        sqrt_165: "f32[256]" = torch.ops.aten.sqrt.default(add_368);  add_368 = None
        reciprocal_165: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_165);  sqrt_165 = None
        mul_533: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_165, 1);  reciprocal_165 = None
        unsqueeze_1320: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_1321: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, -1);  unsqueeze_1320 = None
        unsqueeze_1322: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_533, -1);  mul_533 = None
        unsqueeze_1323: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, -1);  unsqueeze_1322 = None
        sub_203: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_203, unsqueeze_1321);  convolution_203 = unsqueeze_1321 = None
        mul_534: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_1323);  sub_203 = unsqueeze_1323 = None
        unsqueeze_1324: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_1325: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, -1);  unsqueeze_1324 = None
        mul_535: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_534, unsqueeze_1325);  mul_534 = unsqueeze_1325 = None
        unsqueeze_1326: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_1327: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, -1);  unsqueeze_1326 = None
        add_369: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_535, unsqueeze_1327);  mul_535 = unsqueeze_1327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_159: "f32[8, 256, 32, 32]" = torch.ops.aten.relu.default(add_369);  add_369 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_230: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.view.default(relu_159, [8, 2, 128, 32, 32]);  relu_159 = None
        sum_115: "f32[8, 128, 32, 32]" = torch.ops.aten.sum.dim_IntList(view_230, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_39: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(sum_115, [2, 3], True);  sum_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_204: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_39, arg151_1, arg152_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_39 = arg151_1 = arg152_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_370: "f32[64]" = torch.ops.aten.add.Tensor(arg154_1, 1e-05);  arg154_1 = None
        sqrt_166: "f32[64]" = torch.ops.aten.sqrt.default(add_370);  add_370 = None
        reciprocal_166: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_166);  sqrt_166 = None
        mul_536: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_166, 1);  reciprocal_166 = None
        unsqueeze_1328: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_1329: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, -1);  unsqueeze_1328 = None
        unsqueeze_1330: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_536, -1);  mul_536 = None
        unsqueeze_1331: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, -1);  unsqueeze_1330 = None
        sub_204: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_204, unsqueeze_1329);  convolution_204 = unsqueeze_1329 = None
        mul_537: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_1331);  sub_204 = unsqueeze_1331 = None
        unsqueeze_1332: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_1333: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, -1);  unsqueeze_1332 = None
        mul_538: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_537, unsqueeze_1333);  mul_537 = unsqueeze_1333 = None
        unsqueeze_1334: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_1335: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, -1);  unsqueeze_1334 = None
        add_371: "f32[8, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_538, unsqueeze_1335);  mul_538 = unsqueeze_1335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_160: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(add_371);  add_371 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_205: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_160, arg157_1, arg158_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_160 = arg157_1 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_231: "f32[8, 1, 2, 128]" = torch.ops.aten.view.default(convolution_205, [8, 1, 2, -1]);  convolution_205 = None
        permute_39: "f32[8, 2, 1, 128]" = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_38: "f32[8, 1, 1, 128]" = torch.ops.aten.amax.default(permute_39, [1], True)
        sub_205: "f32[8, 2, 1, 128]" = torch.ops.aten.sub.Tensor(permute_39, amax_38);  permute_39 = amax_38 = None
        exp_38: "f32[8, 2, 1, 128]" = torch.ops.aten.exp.default(sub_205);  sub_205 = None
        sum_116: "f32[8, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp_38, [1], True)
        div_38: "f32[8, 2, 1, 128]" = torch.ops.aten.div.Tensor(exp_38, sum_116);  exp_38 = sum_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_232: "f32[8, 256]" = torch.ops.aten.view.default(div_38, [8, -1]);  div_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_233: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(view_232, [8, -1, 1, 1]);  view_232 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_234: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.view.default(view_233, [8, 2, 128, 1, 1]);  view_233 = None
        mul_539: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.mul.Tensor(view_230, view_234);  view_230 = view_234 = None
        sum_117: "f32[8, 128, 32, 32]" = torch.ops.aten.sum.dim_IntList(mul_539, [1]);  mul_539 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_206: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(sum_117, arg159_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_117 = arg159_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_372: "f32[512]" = torch.ops.aten.add.Tensor(arg161_1, 1e-05);  arg161_1 = None
        sqrt_167: "f32[512]" = torch.ops.aten.sqrt.default(add_372);  add_372 = None
        reciprocal_167: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_167);  sqrt_167 = None
        mul_540: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_167, 1);  reciprocal_167 = None
        unsqueeze_1336: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_1337: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, -1);  unsqueeze_1336 = None
        unsqueeze_1338: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_540, -1);  mul_540 = None
        unsqueeze_1339: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, -1);  unsqueeze_1338 = None
        sub_206: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_206, unsqueeze_1337);  convolution_206 = unsqueeze_1337 = None
        mul_541: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_1339);  sub_206 = unsqueeze_1339 = None
        unsqueeze_1340: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_1341: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, -1);  unsqueeze_1340 = None
        mul_542: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_541, unsqueeze_1341);  mul_541 = unsqueeze_1341 = None
        unsqueeze_1342: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_1343: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, -1);  unsqueeze_1342 = None
        add_373: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_542, unsqueeze_1343);  mul_542 = unsqueeze_1343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_374: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_373, relu_157);  add_373 = relu_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_161: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_374);  add_374 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_207: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(relu_161, arg164_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg164_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_375: "f32[128]" = torch.ops.aten.add.Tensor(arg166_1, 1e-05);  arg166_1 = None
        sqrt_168: "f32[128]" = torch.ops.aten.sqrt.default(add_375);  add_375 = None
        reciprocal_168: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_168);  sqrt_168 = None
        mul_543: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_168, 1);  reciprocal_168 = None
        unsqueeze_1344: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_1345: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, -1);  unsqueeze_1344 = None
        unsqueeze_1346: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_543, -1);  mul_543 = None
        unsqueeze_1347: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, -1);  unsqueeze_1346 = None
        sub_207: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_207, unsqueeze_1345);  convolution_207 = unsqueeze_1345 = None
        mul_544: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_1347);  sub_207 = unsqueeze_1347 = None
        unsqueeze_1348: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_1349: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, -1);  unsqueeze_1348 = None
        mul_545: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_544, unsqueeze_1349);  mul_544 = unsqueeze_1349 = None
        unsqueeze_1350: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_1351: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, -1);  unsqueeze_1350 = None
        add_376: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_1351);  mul_545 = unsqueeze_1351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_162: "f32[8, 128, 32, 32]" = torch.ops.aten.relu.default(add_376);  add_376 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_208: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(relu_162, arg169_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_162 = arg169_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_377: "f32[256]" = torch.ops.aten.add.Tensor(arg171_1, 1e-05);  arg171_1 = None
        sqrt_169: "f32[256]" = torch.ops.aten.sqrt.default(add_377);  add_377 = None
        reciprocal_169: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_169);  sqrt_169 = None
        mul_546: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_169, 1);  reciprocal_169 = None
        unsqueeze_1352: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_1353: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, -1);  unsqueeze_1352 = None
        unsqueeze_1354: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_546, -1);  mul_546 = None
        unsqueeze_1355: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, -1);  unsqueeze_1354 = None
        sub_208: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_208, unsqueeze_1353);  convolution_208 = unsqueeze_1353 = None
        mul_547: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_1355);  sub_208 = unsqueeze_1355 = None
        unsqueeze_1356: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_1357: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, -1);  unsqueeze_1356 = None
        mul_548: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_547, unsqueeze_1357);  mul_547 = unsqueeze_1357 = None
        unsqueeze_1358: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_1359: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, -1);  unsqueeze_1358 = None
        add_378: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_548, unsqueeze_1359);  mul_548 = unsqueeze_1359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_163: "f32[8, 256, 32, 32]" = torch.ops.aten.relu.default(add_378);  add_378 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_236: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.view.default(relu_163, [8, 2, 128, 32, 32]);  relu_163 = None
        sum_118: "f32[8, 128, 32, 32]" = torch.ops.aten.sum.dim_IntList(view_236, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_40: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(sum_118, [2, 3], True);  sum_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_209: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_40, arg174_1, arg175_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_40 = arg174_1 = arg175_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_379: "f32[64]" = torch.ops.aten.add.Tensor(arg177_1, 1e-05);  arg177_1 = None
        sqrt_170: "f32[64]" = torch.ops.aten.sqrt.default(add_379);  add_379 = None
        reciprocal_170: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_170);  sqrt_170 = None
        mul_549: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_170, 1);  reciprocal_170 = None
        unsqueeze_1360: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
        unsqueeze_1361: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, -1);  unsqueeze_1360 = None
        unsqueeze_1362: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_549, -1);  mul_549 = None
        unsqueeze_1363: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, -1);  unsqueeze_1362 = None
        sub_209: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_209, unsqueeze_1361);  convolution_209 = unsqueeze_1361 = None
        mul_550: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_1363);  sub_209 = unsqueeze_1363 = None
        unsqueeze_1364: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_1365: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, -1);  unsqueeze_1364 = None
        mul_551: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_550, unsqueeze_1365);  mul_550 = unsqueeze_1365 = None
        unsqueeze_1366: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_1367: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, -1);  unsqueeze_1366 = None
        add_380: "f32[8, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_551, unsqueeze_1367);  mul_551 = unsqueeze_1367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_164: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(add_380);  add_380 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_210: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_164, arg180_1, arg181_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_164 = arg180_1 = arg181_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_237: "f32[8, 1, 2, 128]" = torch.ops.aten.view.default(convolution_210, [8, 1, 2, -1]);  convolution_210 = None
        permute_40: "f32[8, 2, 1, 128]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_39: "f32[8, 1, 1, 128]" = torch.ops.aten.amax.default(permute_40, [1], True)
        sub_210: "f32[8, 2, 1, 128]" = torch.ops.aten.sub.Tensor(permute_40, amax_39);  permute_40 = amax_39 = None
        exp_39: "f32[8, 2, 1, 128]" = torch.ops.aten.exp.default(sub_210);  sub_210 = None
        sum_119: "f32[8, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp_39, [1], True)
        div_39: "f32[8, 2, 1, 128]" = torch.ops.aten.div.Tensor(exp_39, sum_119);  exp_39 = sum_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_238: "f32[8, 256]" = torch.ops.aten.view.default(div_39, [8, -1]);  div_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_239: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(view_238, [8, -1, 1, 1]);  view_238 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_240: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.view.default(view_239, [8, 2, 128, 1, 1]);  view_239 = None
        mul_552: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.mul.Tensor(view_236, view_240);  view_236 = view_240 = None
        sum_120: "f32[8, 128, 32, 32]" = torch.ops.aten.sum.dim_IntList(mul_552, [1]);  mul_552 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_211: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(sum_120, arg182_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_120 = arg182_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_381: "f32[512]" = torch.ops.aten.add.Tensor(arg184_1, 1e-05);  arg184_1 = None
        sqrt_171: "f32[512]" = torch.ops.aten.sqrt.default(add_381);  add_381 = None
        reciprocal_171: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_171);  sqrt_171 = None
        mul_553: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_171, 1);  reciprocal_171 = None
        unsqueeze_1368: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_1369: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, -1);  unsqueeze_1368 = None
        unsqueeze_1370: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_553, -1);  mul_553 = None
        unsqueeze_1371: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, -1);  unsqueeze_1370 = None
        sub_211: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_211, unsqueeze_1369);  convolution_211 = unsqueeze_1369 = None
        mul_554: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_1371);  sub_211 = unsqueeze_1371 = None
        unsqueeze_1372: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_1373: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, -1);  unsqueeze_1372 = None
        mul_555: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_554, unsqueeze_1373);  mul_554 = unsqueeze_1373 = None
        unsqueeze_1374: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
        unsqueeze_1375: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, -1);  unsqueeze_1374 = None
        add_382: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_555, unsqueeze_1375);  mul_555 = unsqueeze_1375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_383: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_382, relu_161);  add_382 = relu_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_165: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_383);  add_383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_212: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(relu_165, arg187_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg187_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_384: "f32[256]" = torch.ops.aten.add.Tensor(arg189_1, 1e-05);  arg189_1 = None
        sqrt_172: "f32[256]" = torch.ops.aten.sqrt.default(add_384);  add_384 = None
        reciprocal_172: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_172);  sqrt_172 = None
        mul_556: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_172, 1);  reciprocal_172 = None
        unsqueeze_1376: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_1377: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, -1);  unsqueeze_1376 = None
        unsqueeze_1378: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_556, -1);  mul_556 = None
        unsqueeze_1379: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, -1);  unsqueeze_1378 = None
        sub_212: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_212, unsqueeze_1377);  convolution_212 = unsqueeze_1377 = None
        mul_557: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_1379);  sub_212 = unsqueeze_1379 = None
        unsqueeze_1380: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_1381: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, -1);  unsqueeze_1380 = None
        mul_558: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_557, unsqueeze_1381);  mul_557 = unsqueeze_1381 = None
        unsqueeze_1382: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
        unsqueeze_1383: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, -1);  unsqueeze_1382 = None
        add_385: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_558, unsqueeze_1383);  mul_558 = unsqueeze_1383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_166: "f32[8, 256, 32, 32]" = torch.ops.aten.relu.default(add_385);  add_385 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_213: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(relu_166, arg192_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_166 = arg192_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_386: "f32[512]" = torch.ops.aten.add.Tensor(arg194_1, 1e-05);  arg194_1 = None
        sqrt_173: "f32[512]" = torch.ops.aten.sqrt.default(add_386);  add_386 = None
        reciprocal_173: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_173);  sqrt_173 = None
        mul_559: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_173, 1);  reciprocal_173 = None
        unsqueeze_1384: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
        unsqueeze_1385: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, -1);  unsqueeze_1384 = None
        unsqueeze_1386: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_559, -1);  mul_559 = None
        unsqueeze_1387: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, -1);  unsqueeze_1386 = None
        sub_213: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_213, unsqueeze_1385);  convolution_213 = unsqueeze_1385 = None
        mul_560: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_1387);  sub_213 = unsqueeze_1387 = None
        unsqueeze_1388: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_1389: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, -1);  unsqueeze_1388 = None
        mul_561: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_1389);  mul_560 = unsqueeze_1389 = None
        unsqueeze_1390: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
        unsqueeze_1391: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, -1);  unsqueeze_1390 = None
        add_387: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_561, unsqueeze_1391);  mul_561 = unsqueeze_1391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_167: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_387);  add_387 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_242: "f32[8, 2, 256, 32, 32]" = torch.ops.aten.view.default(relu_167, [8, 2, 256, 32, 32]);  relu_167 = None
        sum_121: "f32[8, 256, 32, 32]" = torch.ops.aten.sum.dim_IntList(view_242, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_41: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_121, [2, 3], True);  sum_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_214: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_41, arg197_1, arg198_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_41 = arg197_1 = arg198_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_388: "f32[128]" = torch.ops.aten.add.Tensor(arg200_1, 1e-05);  arg200_1 = None
        sqrt_174: "f32[128]" = torch.ops.aten.sqrt.default(add_388);  add_388 = None
        reciprocal_174: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_174);  sqrt_174 = None
        mul_562: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_174, 1);  reciprocal_174 = None
        unsqueeze_1392: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_1393: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1392, -1);  unsqueeze_1392 = None
        unsqueeze_1394: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_562, -1);  mul_562 = None
        unsqueeze_1395: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, -1);  unsqueeze_1394 = None
        sub_214: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_214, unsqueeze_1393);  convolution_214 = unsqueeze_1393 = None
        mul_563: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_1395);  sub_214 = unsqueeze_1395 = None
        unsqueeze_1396: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg201_1, -1);  arg201_1 = None
        unsqueeze_1397: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1396, -1);  unsqueeze_1396 = None
        mul_564: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_563, unsqueeze_1397);  mul_563 = unsqueeze_1397 = None
        unsqueeze_1398: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_1399: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, -1);  unsqueeze_1398 = None
        add_389: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_564, unsqueeze_1399);  mul_564 = unsqueeze_1399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_168: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_389);  add_389 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_215: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_168, arg203_1, arg204_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_168 = arg203_1 = arg204_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_243: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_215, [8, 1, 2, -1]);  convolution_215 = None
        permute_41: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_40: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_41, [1], True)
        sub_215: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_41, amax_40);  permute_41 = amax_40 = None
        exp_40: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_215);  sub_215 = None
        sum_122: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_40, [1], True)
        div_40: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_40, sum_122);  exp_40 = sum_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_244: "f32[8, 512]" = torch.ops.aten.view.default(div_40, [8, -1]);  div_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_245: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_244, [8, -1, 1, 1]);  view_244 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_246: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_245, [8, 2, 256, 1, 1]);  view_245 = None
        mul_565: "f32[8, 2, 256, 32, 32]" = torch.ops.aten.mul.Tensor(view_242, view_246);  view_242 = view_246 = None
        sum_123: "f32[8, 256, 32, 32]" = torch.ops.aten.sum.dim_IntList(mul_565, [1]);  mul_565 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:107 in forward, code: out = self.avd_last(out)
        avg_pool2d_8: "f32[8, 256, 16, 16]" = torch.ops.aten.avg_pool2d.default(sum_123, [3, 3], [2, 2], [1, 1]);  sum_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_216: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(avg_pool2d_8, arg205_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_8 = arg205_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_390: "f32[1024]" = torch.ops.aten.add.Tensor(arg207_1, 1e-05);  arg207_1 = None
        sqrt_175: "f32[1024]" = torch.ops.aten.sqrt.default(add_390);  add_390 = None
        reciprocal_175: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_175);  sqrt_175 = None
        mul_566: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_175, 1);  reciprocal_175 = None
        unsqueeze_1400: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_1401: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, -1);  unsqueeze_1400 = None
        unsqueeze_1402: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_566, -1);  mul_566 = None
        unsqueeze_1403: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1402, -1);  unsqueeze_1402 = None
        sub_216: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_216, unsqueeze_1401);  convolution_216 = unsqueeze_1401 = None
        mul_567: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_1403);  sub_216 = unsqueeze_1403 = None
        unsqueeze_1404: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_1405: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1404, -1);  unsqueeze_1404 = None
        mul_568: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_567, unsqueeze_1405);  mul_567 = unsqueeze_1405 = None
        unsqueeze_1406: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_1407: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, -1);  unsqueeze_1406 = None
        add_391: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_568, unsqueeze_1407);  mul_568 = unsqueeze_1407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:113 in forward, code: shortcut = self.downsample(x)
        avg_pool2d_9: "f32[8, 512, 16, 16]" = torch.ops.aten.avg_pool2d.default(relu_165, [2, 2], [2, 2], [0, 0], True, False);  relu_165 = None
        convolution_217: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(avg_pool2d_9, arg210_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_9 = arg210_1 = None
        add_392: "f32[1024]" = torch.ops.aten.add.Tensor(arg212_1, 1e-05);  arg212_1 = None
        sqrt_176: "f32[1024]" = torch.ops.aten.sqrt.default(add_392);  add_392 = None
        reciprocal_176: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_176);  sqrt_176 = None
        mul_569: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_176, 1);  reciprocal_176 = None
        unsqueeze_1408: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_1409: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1408, -1);  unsqueeze_1408 = None
        unsqueeze_1410: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_569, -1);  mul_569 = None
        unsqueeze_1411: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, -1);  unsqueeze_1410 = None
        sub_217: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_217, unsqueeze_1409);  convolution_217 = unsqueeze_1409 = None
        mul_570: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_1411);  sub_217 = unsqueeze_1411 = None
        unsqueeze_1412: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_1413: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, -1);  unsqueeze_1412 = None
        mul_571: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_570, unsqueeze_1413);  mul_570 = unsqueeze_1413 = None
        unsqueeze_1414: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_1415: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1414, -1);  unsqueeze_1414 = None
        add_393: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_571, unsqueeze_1415);  mul_571 = unsqueeze_1415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_394: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_391, add_393);  add_391 = add_393 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_169: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_394);  add_394 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_218: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_169, arg215_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg215_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_395: "f32[256]" = torch.ops.aten.add.Tensor(arg217_1, 1e-05);  arg217_1 = None
        sqrt_177: "f32[256]" = torch.ops.aten.sqrt.default(add_395);  add_395 = None
        reciprocal_177: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_177);  sqrt_177 = None
        mul_572: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_177, 1);  reciprocal_177 = None
        unsqueeze_1416: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_1417: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1416, -1);  unsqueeze_1416 = None
        unsqueeze_1418: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_572, -1);  mul_572 = None
        unsqueeze_1419: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, -1);  unsqueeze_1418 = None
        sub_218: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_218, unsqueeze_1417);  convolution_218 = unsqueeze_1417 = None
        mul_573: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_1419);  sub_218 = unsqueeze_1419 = None
        unsqueeze_1420: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_1421: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1420, -1);  unsqueeze_1420 = None
        mul_574: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_573, unsqueeze_1421);  mul_573 = unsqueeze_1421 = None
        unsqueeze_1422: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_1423: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, -1);  unsqueeze_1422 = None
        add_396: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_574, unsqueeze_1423);  mul_574 = unsqueeze_1423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_170: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_396);  add_396 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_219: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_170, arg220_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_170 = arg220_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_397: "f32[512]" = torch.ops.aten.add.Tensor(arg222_1, 1e-05);  arg222_1 = None
        sqrt_178: "f32[512]" = torch.ops.aten.sqrt.default(add_397);  add_397 = None
        reciprocal_178: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_178);  sqrt_178 = None
        mul_575: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_178, 1);  reciprocal_178 = None
        unsqueeze_1424: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_1425: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, -1);  unsqueeze_1424 = None
        unsqueeze_1426: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_575, -1);  mul_575 = None
        unsqueeze_1427: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1426, -1);  unsqueeze_1426 = None
        sub_219: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_219, unsqueeze_1425);  convolution_219 = unsqueeze_1425 = None
        mul_576: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_1427);  sub_219 = unsqueeze_1427 = None
        unsqueeze_1428: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_1429: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1428, -1);  unsqueeze_1428 = None
        mul_577: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_576, unsqueeze_1429);  mul_576 = unsqueeze_1429 = None
        unsqueeze_1430: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_1431: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1430, -1);  unsqueeze_1430 = None
        add_398: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_577, unsqueeze_1431);  mul_577 = unsqueeze_1431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_171: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_398);  add_398 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_248: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_171, [8, 2, 256, 16, 16]);  relu_171 = None
        sum_124: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_248, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_42: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_124, [2, 3], True);  sum_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_220: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_42, arg225_1, arg226_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_42 = arg225_1 = arg226_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_399: "f32[128]" = torch.ops.aten.add.Tensor(arg228_1, 1e-05);  arg228_1 = None
        sqrt_179: "f32[128]" = torch.ops.aten.sqrt.default(add_399);  add_399 = None
        reciprocal_179: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_179);  sqrt_179 = None
        mul_578: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_179, 1);  reciprocal_179 = None
        unsqueeze_1432: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_1433: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, -1);  unsqueeze_1432 = None
        unsqueeze_1434: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_578, -1);  mul_578 = None
        unsqueeze_1435: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, -1);  unsqueeze_1434 = None
        sub_220: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_220, unsqueeze_1433);  convolution_220 = unsqueeze_1433 = None
        mul_579: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_1435);  sub_220 = unsqueeze_1435 = None
        unsqueeze_1436: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_1437: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1436, -1);  unsqueeze_1436 = None
        mul_580: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_579, unsqueeze_1437);  mul_579 = unsqueeze_1437 = None
        unsqueeze_1438: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_1439: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, -1);  unsqueeze_1438 = None
        add_400: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_580, unsqueeze_1439);  mul_580 = unsqueeze_1439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_172: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_400);  add_400 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_221: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_172, arg231_1, arg232_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_172 = arg231_1 = arg232_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_249: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_221, [8, 1, 2, -1]);  convolution_221 = None
        permute_42: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_41: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_42, [1], True)
        sub_221: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_42, amax_41);  permute_42 = amax_41 = None
        exp_41: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_221);  sub_221 = None
        sum_125: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_41, [1], True)
        div_41: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_41, sum_125);  exp_41 = sum_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_250: "f32[8, 512]" = torch.ops.aten.view.default(div_41, [8, -1]);  div_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_251: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_250, [8, -1, 1, 1]);  view_250 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_252: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_251, [8, 2, 256, 1, 1]);  view_251 = None
        mul_581: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_248, view_252);  view_248 = view_252 = None
        sum_126: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_581, [1]);  mul_581 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_222: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_126, arg233_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_126 = arg233_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_401: "f32[1024]" = torch.ops.aten.add.Tensor(arg235_1, 1e-05);  arg235_1 = None
        sqrt_180: "f32[1024]" = torch.ops.aten.sqrt.default(add_401);  add_401 = None
        reciprocal_180: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_180);  sqrt_180 = None
        mul_582: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_180, 1);  reciprocal_180 = None
        unsqueeze_1440: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_1441: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1440, -1);  unsqueeze_1440 = None
        unsqueeze_1442: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_582, -1);  mul_582 = None
        unsqueeze_1443: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1442, -1);  unsqueeze_1442 = None
        sub_222: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_222, unsqueeze_1441);  convolution_222 = unsqueeze_1441 = None
        mul_583: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_1443);  sub_222 = unsqueeze_1443 = None
        unsqueeze_1444: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_1445: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, -1);  unsqueeze_1444 = None
        mul_584: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_583, unsqueeze_1445);  mul_583 = unsqueeze_1445 = None
        unsqueeze_1446: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_1447: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, -1);  unsqueeze_1446 = None
        add_402: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_584, unsqueeze_1447);  mul_584 = unsqueeze_1447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_403: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_402, relu_169);  add_402 = relu_169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_173: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_403);  add_403 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_223: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_173, arg238_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg238_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_404: "f32[256]" = torch.ops.aten.add.Tensor(arg240_1, 1e-05);  arg240_1 = None
        sqrt_181: "f32[256]" = torch.ops.aten.sqrt.default(add_404);  add_404 = None
        reciprocal_181: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_181);  sqrt_181 = None
        mul_585: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_181, 1);  reciprocal_181 = None
        unsqueeze_1448: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_1449: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, -1);  unsqueeze_1448 = None
        unsqueeze_1450: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_585, -1);  mul_585 = None
        unsqueeze_1451: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, -1);  unsqueeze_1450 = None
        sub_223: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_223, unsqueeze_1449);  convolution_223 = unsqueeze_1449 = None
        mul_586: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_1451);  sub_223 = unsqueeze_1451 = None
        unsqueeze_1452: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg241_1, -1);  arg241_1 = None
        unsqueeze_1453: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1452, -1);  unsqueeze_1452 = None
        mul_587: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_586, unsqueeze_1453);  mul_586 = unsqueeze_1453 = None
        unsqueeze_1454: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_1455: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, -1);  unsqueeze_1454 = None
        add_405: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_1455);  mul_587 = unsqueeze_1455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_174: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_405);  add_405 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_224: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_174, arg243_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_174 = arg243_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_406: "f32[512]" = torch.ops.aten.add.Tensor(arg245_1, 1e-05);  arg245_1 = None
        sqrt_182: "f32[512]" = torch.ops.aten.sqrt.default(add_406);  add_406 = None
        reciprocal_182: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_182);  sqrt_182 = None
        mul_588: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_182, 1);  reciprocal_182 = None
        unsqueeze_1456: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_1457: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, -1);  unsqueeze_1456 = None
        unsqueeze_1458: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_588, -1);  mul_588 = None
        unsqueeze_1459: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, -1);  unsqueeze_1458 = None
        sub_224: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_224, unsqueeze_1457);  convolution_224 = unsqueeze_1457 = None
        mul_589: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_1459);  sub_224 = unsqueeze_1459 = None
        unsqueeze_1460: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg246_1, -1);  arg246_1 = None
        unsqueeze_1461: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, -1);  unsqueeze_1460 = None
        mul_590: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_589, unsqueeze_1461);  mul_589 = unsqueeze_1461 = None
        unsqueeze_1462: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_1463: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, -1);  unsqueeze_1462 = None
        add_407: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_590, unsqueeze_1463);  mul_590 = unsqueeze_1463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_175: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_407);  add_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_254: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_175, [8, 2, 256, 16, 16]);  relu_175 = None
        sum_127: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_254, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_43: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_127, [2, 3], True);  sum_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_225: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_43, arg248_1, arg249_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_43 = arg248_1 = arg249_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_408: "f32[128]" = torch.ops.aten.add.Tensor(arg251_1, 1e-05);  arg251_1 = None
        sqrt_183: "f32[128]" = torch.ops.aten.sqrt.default(add_408);  add_408 = None
        reciprocal_183: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_183);  sqrt_183 = None
        mul_591: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_183, 1);  reciprocal_183 = None
        unsqueeze_1464: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_1465: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1464, -1);  unsqueeze_1464 = None
        unsqueeze_1466: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_591, -1);  mul_591 = None
        unsqueeze_1467: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, -1);  unsqueeze_1466 = None
        sub_225: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_225, unsqueeze_1465);  convolution_225 = unsqueeze_1465 = None
        mul_592: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_1467);  sub_225 = unsqueeze_1467 = None
        unsqueeze_1468: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_1469: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, -1);  unsqueeze_1468 = None
        mul_593: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_592, unsqueeze_1469);  mul_592 = unsqueeze_1469 = None
        unsqueeze_1470: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg253_1, -1);  arg253_1 = None
        unsqueeze_1471: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, -1);  unsqueeze_1470 = None
        add_409: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_593, unsqueeze_1471);  mul_593 = unsqueeze_1471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_176: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_409);  add_409 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_226: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_176, arg254_1, arg255_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_176 = arg254_1 = arg255_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_255: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_226, [8, 1, 2, -1]);  convolution_226 = None
        permute_43: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_42: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_43, [1], True)
        sub_226: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_43, amax_42);  permute_43 = amax_42 = None
        exp_42: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_226);  sub_226 = None
        sum_128: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_42, [1], True)
        div_42: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_42, sum_128);  exp_42 = sum_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_256: "f32[8, 512]" = torch.ops.aten.view.default(div_42, [8, -1]);  div_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_257: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_256, [8, -1, 1, 1]);  view_256 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_258: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_257, [8, 2, 256, 1, 1]);  view_257 = None
        mul_594: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_254, view_258);  view_254 = view_258 = None
        sum_129: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_594, [1]);  mul_594 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_227: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_129, arg256_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_129 = arg256_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_410: "f32[1024]" = torch.ops.aten.add.Tensor(arg258_1, 1e-05);  arg258_1 = None
        sqrt_184: "f32[1024]" = torch.ops.aten.sqrt.default(add_410);  add_410 = None
        reciprocal_184: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_184);  sqrt_184 = None
        mul_595: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_184, 1);  reciprocal_184 = None
        unsqueeze_1472: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_1473: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, -1);  unsqueeze_1472 = None
        unsqueeze_1474: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_595, -1);  mul_595 = None
        unsqueeze_1475: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, -1);  unsqueeze_1474 = None
        sub_227: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_227, unsqueeze_1473);  convolution_227 = unsqueeze_1473 = None
        mul_596: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_1475);  sub_227 = unsqueeze_1475 = None
        unsqueeze_1476: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_1477: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1476, -1);  unsqueeze_1476 = None
        mul_597: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_596, unsqueeze_1477);  mul_596 = unsqueeze_1477 = None
        unsqueeze_1478: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_1479: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, -1);  unsqueeze_1478 = None
        add_411: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_597, unsqueeze_1479);  mul_597 = unsqueeze_1479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_412: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_411, relu_173);  add_411 = relu_173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_177: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_412);  add_412 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_228: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_177, arg261_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg261_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_413: "f32[256]" = torch.ops.aten.add.Tensor(arg263_1, 1e-05);  arg263_1 = None
        sqrt_185: "f32[256]" = torch.ops.aten.sqrt.default(add_413);  add_413 = None
        reciprocal_185: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_185);  sqrt_185 = None
        mul_598: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_185, 1);  reciprocal_185 = None
        unsqueeze_1480: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_1481: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1480, -1);  unsqueeze_1480 = None
        unsqueeze_1482: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_598, -1);  mul_598 = None
        unsqueeze_1483: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, -1);  unsqueeze_1482 = None
        sub_228: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_228, unsqueeze_1481);  convolution_228 = unsqueeze_1481 = None
        mul_599: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_1483);  sub_228 = unsqueeze_1483 = None
        unsqueeze_1484: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_1485: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, -1);  unsqueeze_1484 = None
        mul_600: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_599, unsqueeze_1485);  mul_599 = unsqueeze_1485 = None
        unsqueeze_1486: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_1487: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1486, -1);  unsqueeze_1486 = None
        add_414: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_600, unsqueeze_1487);  mul_600 = unsqueeze_1487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_178: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_414);  add_414 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_229: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_178, arg266_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_178 = arg266_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_415: "f32[512]" = torch.ops.aten.add.Tensor(arg268_1, 1e-05);  arg268_1 = None
        sqrt_186: "f32[512]" = torch.ops.aten.sqrt.default(add_415);  add_415 = None
        reciprocal_186: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_186);  sqrt_186 = None
        mul_601: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_186, 1);  reciprocal_186 = None
        unsqueeze_1488: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_1489: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1488, -1);  unsqueeze_1488 = None
        unsqueeze_1490: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_601, -1);  mul_601 = None
        unsqueeze_1491: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, -1);  unsqueeze_1490 = None
        sub_229: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_229, unsqueeze_1489);  convolution_229 = unsqueeze_1489 = None
        mul_602: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_1491);  sub_229 = unsqueeze_1491 = None
        unsqueeze_1492: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_1493: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1492, -1);  unsqueeze_1492 = None
        mul_603: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_602, unsqueeze_1493);  mul_602 = unsqueeze_1493 = None
        unsqueeze_1494: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_1495: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, -1);  unsqueeze_1494 = None
        add_416: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_603, unsqueeze_1495);  mul_603 = unsqueeze_1495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_179: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_416);  add_416 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_260: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_179, [8, 2, 256, 16, 16]);  relu_179 = None
        sum_130: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_260, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_44: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_130, [2, 3], True);  sum_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_230: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_44, arg271_1, arg272_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_44 = arg271_1 = arg272_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_417: "f32[128]" = torch.ops.aten.add.Tensor(arg274_1, 1e-05);  arg274_1 = None
        sqrt_187: "f32[128]" = torch.ops.aten.sqrt.default(add_417);  add_417 = None
        reciprocal_187: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_187);  sqrt_187 = None
        mul_604: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_187, 1);  reciprocal_187 = None
        unsqueeze_1496: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg273_1, -1);  arg273_1 = None
        unsqueeze_1497: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1496, -1);  unsqueeze_1496 = None
        unsqueeze_1498: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_604, -1);  mul_604 = None
        unsqueeze_1499: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, -1);  unsqueeze_1498 = None
        sub_230: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_230, unsqueeze_1497);  convolution_230 = unsqueeze_1497 = None
        mul_605: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_1499);  sub_230 = unsqueeze_1499 = None
        unsqueeze_1500: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_1501: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1500, -1);  unsqueeze_1500 = None
        mul_606: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_605, unsqueeze_1501);  mul_605 = unsqueeze_1501 = None
        unsqueeze_1502: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_1503: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1502, -1);  unsqueeze_1502 = None
        add_418: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_606, unsqueeze_1503);  mul_606 = unsqueeze_1503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_180: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_418);  add_418 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_231: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_180, arg277_1, arg278_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_180 = arg277_1 = arg278_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_261: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_231, [8, 1, 2, -1]);  convolution_231 = None
        permute_44: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_43: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_44, [1], True)
        sub_231: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_44, amax_43);  permute_44 = amax_43 = None
        exp_43: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_231);  sub_231 = None
        sum_131: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_43, [1], True)
        div_43: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_43, sum_131);  exp_43 = sum_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_262: "f32[8, 512]" = torch.ops.aten.view.default(div_43, [8, -1]);  div_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_263: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_262, [8, -1, 1, 1]);  view_262 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_264: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_263, [8, 2, 256, 1, 1]);  view_263 = None
        mul_607: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_260, view_264);  view_260 = view_264 = None
        sum_132: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_607, [1]);  mul_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_232: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_132, arg279_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_132 = arg279_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_419: "f32[1024]" = torch.ops.aten.add.Tensor(arg281_1, 1e-05);  arg281_1 = None
        sqrt_188: "f32[1024]" = torch.ops.aten.sqrt.default(add_419);  add_419 = None
        reciprocal_188: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_188);  sqrt_188 = None
        mul_608: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_188, 1);  reciprocal_188 = None
        unsqueeze_1504: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_1505: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1504, -1);  unsqueeze_1504 = None
        unsqueeze_1506: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_608, -1);  mul_608 = None
        unsqueeze_1507: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, -1);  unsqueeze_1506 = None
        sub_232: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_232, unsqueeze_1505);  convolution_232 = unsqueeze_1505 = None
        mul_609: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_1507);  sub_232 = unsqueeze_1507 = None
        unsqueeze_1508: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_1509: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1508, -1);  unsqueeze_1508 = None
        mul_610: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_609, unsqueeze_1509);  mul_609 = unsqueeze_1509 = None
        unsqueeze_1510: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg283_1, -1);  arg283_1 = None
        unsqueeze_1511: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1510, -1);  unsqueeze_1510 = None
        add_420: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_610, unsqueeze_1511);  mul_610 = unsqueeze_1511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_421: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_420, relu_177);  add_420 = relu_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_181: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_421);  add_421 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_233: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_181, arg284_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg284_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_422: "f32[256]" = torch.ops.aten.add.Tensor(arg286_1, 1e-05);  arg286_1 = None
        sqrt_189: "f32[256]" = torch.ops.aten.sqrt.default(add_422);  add_422 = None
        reciprocal_189: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_189);  sqrt_189 = None
        mul_611: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_189, 1);  reciprocal_189 = None
        unsqueeze_1512: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_1513: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1512, -1);  unsqueeze_1512 = None
        unsqueeze_1514: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_611, -1);  mul_611 = None
        unsqueeze_1515: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1514, -1);  unsqueeze_1514 = None
        sub_233: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_233, unsqueeze_1513);  convolution_233 = unsqueeze_1513 = None
        mul_612: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_1515);  sub_233 = unsqueeze_1515 = None
        unsqueeze_1516: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_1517: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1516, -1);  unsqueeze_1516 = None
        mul_613: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_612, unsqueeze_1517);  mul_612 = unsqueeze_1517 = None
        unsqueeze_1518: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_1519: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, -1);  unsqueeze_1518 = None
        add_423: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_613, unsqueeze_1519);  mul_613 = unsqueeze_1519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_182: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_423);  add_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_234: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_182, arg289_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_182 = arg289_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_424: "f32[512]" = torch.ops.aten.add.Tensor(arg291_1, 1e-05);  arg291_1 = None
        sqrt_190: "f32[512]" = torch.ops.aten.sqrt.default(add_424);  add_424 = None
        reciprocal_190: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_190);  sqrt_190 = None
        mul_614: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_190, 1);  reciprocal_190 = None
        unsqueeze_1520: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_1521: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1520, -1);  unsqueeze_1520 = None
        unsqueeze_1522: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_614, -1);  mul_614 = None
        unsqueeze_1523: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1522, -1);  unsqueeze_1522 = None
        sub_234: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_234, unsqueeze_1521);  convolution_234 = unsqueeze_1521 = None
        mul_615: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_1523);  sub_234 = unsqueeze_1523 = None
        unsqueeze_1524: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_1525: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1524, -1);  unsqueeze_1524 = None
        mul_616: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_615, unsqueeze_1525);  mul_615 = unsqueeze_1525 = None
        unsqueeze_1526: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg293_1, -1);  arg293_1 = None
        unsqueeze_1527: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1526, -1);  unsqueeze_1526 = None
        add_425: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_616, unsqueeze_1527);  mul_616 = unsqueeze_1527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_183: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_425);  add_425 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_266: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_183, [8, 2, 256, 16, 16]);  relu_183 = None
        sum_133: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_266, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_45: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_133, [2, 3], True);  sum_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_235: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_45, arg294_1, arg295_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_45 = arg294_1 = arg295_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_426: "f32[128]" = torch.ops.aten.add.Tensor(arg297_1, 1e-05);  arg297_1 = None
        sqrt_191: "f32[128]" = torch.ops.aten.sqrt.default(add_426);  add_426 = None
        reciprocal_191: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_191);  sqrt_191 = None
        mul_617: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_191, 1);  reciprocal_191 = None
        unsqueeze_1528: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg296_1, -1);  arg296_1 = None
        unsqueeze_1529: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1528, -1);  unsqueeze_1528 = None
        unsqueeze_1530: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_617, -1);  mul_617 = None
        unsqueeze_1531: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, -1);  unsqueeze_1530 = None
        sub_235: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_235, unsqueeze_1529);  convolution_235 = unsqueeze_1529 = None
        mul_618: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_1531);  sub_235 = unsqueeze_1531 = None
        unsqueeze_1532: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
        unsqueeze_1533: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1532, -1);  unsqueeze_1532 = None
        mul_619: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_618, unsqueeze_1533);  mul_618 = unsqueeze_1533 = None
        unsqueeze_1534: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_1535: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1534, -1);  unsqueeze_1534 = None
        add_427: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_619, unsqueeze_1535);  mul_619 = unsqueeze_1535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_184: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_427);  add_427 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_236: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_184, arg300_1, arg301_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_184 = arg300_1 = arg301_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_267: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_236, [8, 1, 2, -1]);  convolution_236 = None
        permute_45: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_44: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_45, [1], True)
        sub_236: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_45, amax_44);  permute_45 = amax_44 = None
        exp_44: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_236);  sub_236 = None
        sum_134: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_44, [1], True)
        div_44: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_44, sum_134);  exp_44 = sum_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_268: "f32[8, 512]" = torch.ops.aten.view.default(div_44, [8, -1]);  div_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_269: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_268, [8, -1, 1, 1]);  view_268 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_270: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_269, [8, 2, 256, 1, 1]);  view_269 = None
        mul_620: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_266, view_270);  view_266 = view_270 = None
        sum_135: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_620, [1]);  mul_620 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_237: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_135, arg302_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_135 = arg302_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_428: "f32[1024]" = torch.ops.aten.add.Tensor(arg304_1, 1e-05);  arg304_1 = None
        sqrt_192: "f32[1024]" = torch.ops.aten.sqrt.default(add_428);  add_428 = None
        reciprocal_192: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_192);  sqrt_192 = None
        mul_621: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_192, 1);  reciprocal_192 = None
        unsqueeze_1536: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_1537: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1536, -1);  unsqueeze_1536 = None
        unsqueeze_1538: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_621, -1);  mul_621 = None
        unsqueeze_1539: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1538, -1);  unsqueeze_1538 = None
        sub_237: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_237, unsqueeze_1537);  convolution_237 = unsqueeze_1537 = None
        mul_622: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_1539);  sub_237 = unsqueeze_1539 = None
        unsqueeze_1540: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_1541: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1540, -1);  unsqueeze_1540 = None
        mul_623: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_622, unsqueeze_1541);  mul_622 = unsqueeze_1541 = None
        unsqueeze_1542: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg306_1, -1);  arg306_1 = None
        unsqueeze_1543: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, -1);  unsqueeze_1542 = None
        add_429: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_623, unsqueeze_1543);  mul_623 = unsqueeze_1543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_430: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_429, relu_181);  add_429 = relu_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_185: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_430);  add_430 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_238: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_185, arg307_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg307_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_431: "f32[256]" = torch.ops.aten.add.Tensor(arg309_1, 1e-05);  arg309_1 = None
        sqrt_193: "f32[256]" = torch.ops.aten.sqrt.default(add_431);  add_431 = None
        reciprocal_193: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_193);  sqrt_193 = None
        mul_624: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_193, 1);  reciprocal_193 = None
        unsqueeze_1544: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_1545: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1544, -1);  unsqueeze_1544 = None
        unsqueeze_1546: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_624, -1);  mul_624 = None
        unsqueeze_1547: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1546, -1);  unsqueeze_1546 = None
        sub_238: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_238, unsqueeze_1545);  convolution_238 = unsqueeze_1545 = None
        mul_625: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_1547);  sub_238 = unsqueeze_1547 = None
        unsqueeze_1548: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
        unsqueeze_1549: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1548, -1);  unsqueeze_1548 = None
        mul_626: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_625, unsqueeze_1549);  mul_625 = unsqueeze_1549 = None
        unsqueeze_1550: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_1551: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1550, -1);  unsqueeze_1550 = None
        add_432: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_626, unsqueeze_1551);  mul_626 = unsqueeze_1551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_186: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_432);  add_432 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_239: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_186, arg312_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_186 = arg312_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_433: "f32[512]" = torch.ops.aten.add.Tensor(arg314_1, 1e-05);  arg314_1 = None
        sqrt_194: "f32[512]" = torch.ops.aten.sqrt.default(add_433);  add_433 = None
        reciprocal_194: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_194);  sqrt_194 = None
        mul_627: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_194, 1);  reciprocal_194 = None
        unsqueeze_1552: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
        unsqueeze_1553: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1552, -1);  unsqueeze_1552 = None
        unsqueeze_1554: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_627, -1);  mul_627 = None
        unsqueeze_1555: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, -1);  unsqueeze_1554 = None
        sub_239: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_239, unsqueeze_1553);  convolution_239 = unsqueeze_1553 = None
        mul_628: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_1555);  sub_239 = unsqueeze_1555 = None
        unsqueeze_1556: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg315_1, -1);  arg315_1 = None
        unsqueeze_1557: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1556, -1);  unsqueeze_1556 = None
        mul_629: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_628, unsqueeze_1557);  mul_628 = unsqueeze_1557 = None
        unsqueeze_1558: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg316_1, -1);  arg316_1 = None
        unsqueeze_1559: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1558, -1);  unsqueeze_1558 = None
        add_434: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_629, unsqueeze_1559);  mul_629 = unsqueeze_1559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_187: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_434);  add_434 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_272: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_187, [8, 2, 256, 16, 16]);  relu_187 = None
        sum_136: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_272, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_46: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_136, [2, 3], True);  sum_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_240: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_46, arg317_1, arg318_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_46 = arg317_1 = arg318_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_435: "f32[128]" = torch.ops.aten.add.Tensor(arg320_1, 1e-05);  arg320_1 = None
        sqrt_195: "f32[128]" = torch.ops.aten.sqrt.default(add_435);  add_435 = None
        reciprocal_195: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_195);  sqrt_195 = None
        mul_630: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_195, 1);  reciprocal_195 = None
        unsqueeze_1560: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_1561: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1560, -1);  unsqueeze_1560 = None
        unsqueeze_1562: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_630, -1);  mul_630 = None
        unsqueeze_1563: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1562, -1);  unsqueeze_1562 = None
        sub_240: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_240, unsqueeze_1561);  convolution_240 = unsqueeze_1561 = None
        mul_631: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_1563);  sub_240 = unsqueeze_1563 = None
        unsqueeze_1564: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg321_1, -1);  arg321_1 = None
        unsqueeze_1565: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1564, -1);  unsqueeze_1564 = None
        mul_632: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_631, unsqueeze_1565);  mul_631 = unsqueeze_1565 = None
        unsqueeze_1566: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_1567: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, -1);  unsqueeze_1566 = None
        add_436: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_632, unsqueeze_1567);  mul_632 = unsqueeze_1567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_188: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_436);  add_436 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_241: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_188, arg323_1, arg324_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_188 = arg323_1 = arg324_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_273: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_241, [8, 1, 2, -1]);  convolution_241 = None
        permute_46: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_45: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_46, [1], True)
        sub_241: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_46, amax_45);  permute_46 = amax_45 = None
        exp_45: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_241);  sub_241 = None
        sum_137: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_45, [1], True)
        div_45: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_45, sum_137);  exp_45 = sum_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_274: "f32[8, 512]" = torch.ops.aten.view.default(div_45, [8, -1]);  div_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_275: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_274, [8, -1, 1, 1]);  view_274 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_276: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_275, [8, 2, 256, 1, 1]);  view_275 = None
        mul_633: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_272, view_276);  view_272 = view_276 = None
        sum_138: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_633, [1]);  mul_633 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_242: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_138, arg325_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_138 = arg325_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_437: "f32[1024]" = torch.ops.aten.add.Tensor(arg327_1, 1e-05);  arg327_1 = None
        sqrt_196: "f32[1024]" = torch.ops.aten.sqrt.default(add_437);  add_437 = None
        reciprocal_196: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_196);  sqrt_196 = None
        mul_634: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_196, 1);  reciprocal_196 = None
        unsqueeze_1568: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg326_1, -1);  arg326_1 = None
        unsqueeze_1569: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1568, -1);  unsqueeze_1568 = None
        unsqueeze_1570: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_634, -1);  mul_634 = None
        unsqueeze_1571: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1570, -1);  unsqueeze_1570 = None
        sub_242: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_242, unsqueeze_1569);  convolution_242 = unsqueeze_1569 = None
        mul_635: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_1571);  sub_242 = unsqueeze_1571 = None
        unsqueeze_1572: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
        unsqueeze_1573: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1572, -1);  unsqueeze_1572 = None
        mul_636: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_635, unsqueeze_1573);  mul_635 = unsqueeze_1573 = None
        unsqueeze_1574: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_1575: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1574, -1);  unsqueeze_1574 = None
        add_438: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_636, unsqueeze_1575);  mul_636 = unsqueeze_1575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_439: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_438, relu_185);  add_438 = relu_185 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_189: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_439);  add_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_243: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_189, arg330_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg330_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_440: "f32[256]" = torch.ops.aten.add.Tensor(arg332_1, 1e-05);  arg332_1 = None
        sqrt_197: "f32[256]" = torch.ops.aten.sqrt.default(add_440);  add_440 = None
        reciprocal_197: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_197);  sqrt_197 = None
        mul_637: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_197, 1);  reciprocal_197 = None
        unsqueeze_1576: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg331_1, -1);  arg331_1 = None
        unsqueeze_1577: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1576, -1);  unsqueeze_1576 = None
        unsqueeze_1578: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_637, -1);  mul_637 = None
        unsqueeze_1579: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, -1);  unsqueeze_1578 = None
        sub_243: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_243, unsqueeze_1577);  convolution_243 = unsqueeze_1577 = None
        mul_638: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_1579);  sub_243 = unsqueeze_1579 = None
        unsqueeze_1580: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg333_1, -1);  arg333_1 = None
        unsqueeze_1581: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1580, -1);  unsqueeze_1580 = None
        mul_639: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_638, unsqueeze_1581);  mul_638 = unsqueeze_1581 = None
        unsqueeze_1582: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_1583: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1582, -1);  unsqueeze_1582 = None
        add_441: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_639, unsqueeze_1583);  mul_639 = unsqueeze_1583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_190: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_441);  add_441 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_244: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_190, arg335_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_190 = arg335_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_442: "f32[512]" = torch.ops.aten.add.Tensor(arg337_1, 1e-05);  arg337_1 = None
        sqrt_198: "f32[512]" = torch.ops.aten.sqrt.default(add_442);  add_442 = None
        reciprocal_198: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_198);  sqrt_198 = None
        mul_640: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_198, 1);  reciprocal_198 = None
        unsqueeze_1584: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
        unsqueeze_1585: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1584, -1);  unsqueeze_1584 = None
        unsqueeze_1586: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_640, -1);  mul_640 = None
        unsqueeze_1587: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1586, -1);  unsqueeze_1586 = None
        sub_244: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_244, unsqueeze_1585);  convolution_244 = unsqueeze_1585 = None
        mul_641: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_1587);  sub_244 = unsqueeze_1587 = None
        unsqueeze_1588: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg338_1, -1);  arg338_1 = None
        unsqueeze_1589: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1588, -1);  unsqueeze_1588 = None
        mul_642: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_641, unsqueeze_1589);  mul_641 = unsqueeze_1589 = None
        unsqueeze_1590: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_1591: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, -1);  unsqueeze_1590 = None
        add_443: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_642, unsqueeze_1591);  mul_642 = unsqueeze_1591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_191: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_443);  add_443 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_278: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_191, [8, 2, 256, 16, 16]);  relu_191 = None
        sum_139: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_278, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_47: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_139, [2, 3], True);  sum_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_245: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_47, arg340_1, arg341_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_47 = arg340_1 = arg341_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_444: "f32[128]" = torch.ops.aten.add.Tensor(arg343_1, 1e-05);  arg343_1 = None
        sqrt_199: "f32[128]" = torch.ops.aten.sqrt.default(add_444);  add_444 = None
        reciprocal_199: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_199);  sqrt_199 = None
        mul_643: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_199, 1);  reciprocal_199 = None
        unsqueeze_1592: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_1593: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1592, -1);  unsqueeze_1592 = None
        unsqueeze_1594: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_643, -1);  mul_643 = None
        unsqueeze_1595: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1594, -1);  unsqueeze_1594 = None
        sub_245: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_245, unsqueeze_1593);  convolution_245 = unsqueeze_1593 = None
        mul_644: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_1595);  sub_245 = unsqueeze_1595 = None
        unsqueeze_1596: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_1597: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1596, -1);  unsqueeze_1596 = None
        mul_645: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_644, unsqueeze_1597);  mul_644 = unsqueeze_1597 = None
        unsqueeze_1598: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_1599: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1598, -1);  unsqueeze_1598 = None
        add_445: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_645, unsqueeze_1599);  mul_645 = unsqueeze_1599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_192: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_445);  add_445 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_246: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_192, arg346_1, arg347_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_192 = arg346_1 = arg347_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_279: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_246, [8, 1, 2, -1]);  convolution_246 = None
        permute_47: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_46: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_47, [1], True)
        sub_246: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_47, amax_46);  permute_47 = amax_46 = None
        exp_46: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_246);  sub_246 = None
        sum_140: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_46, [1], True)
        div_46: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_46, sum_140);  exp_46 = sum_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_280: "f32[8, 512]" = torch.ops.aten.view.default(div_46, [8, -1]);  div_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_281: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_280, [8, -1, 1, 1]);  view_280 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_282: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_281, [8, 2, 256, 1, 1]);  view_281 = None
        mul_646: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_278, view_282);  view_278 = view_282 = None
        sum_141: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_646, [1]);  mul_646 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_247: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_141, arg348_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_141 = arg348_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_446: "f32[1024]" = torch.ops.aten.add.Tensor(arg350_1, 1e-05);  arg350_1 = None
        sqrt_200: "f32[1024]" = torch.ops.aten.sqrt.default(add_446);  add_446 = None
        reciprocal_200: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_200);  sqrt_200 = None
        mul_647: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_200, 1);  reciprocal_200 = None
        unsqueeze_1600: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_1601: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1600, -1);  unsqueeze_1600 = None
        unsqueeze_1602: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_647, -1);  mul_647 = None
        unsqueeze_1603: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, -1);  unsqueeze_1602 = None
        sub_247: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_247, unsqueeze_1601);  convolution_247 = unsqueeze_1601 = None
        mul_648: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_1603);  sub_247 = unsqueeze_1603 = None
        unsqueeze_1604: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg351_1, -1);  arg351_1 = None
        unsqueeze_1605: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1604, -1);  unsqueeze_1604 = None
        mul_649: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_648, unsqueeze_1605);  mul_648 = unsqueeze_1605 = None
        unsqueeze_1606: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
        unsqueeze_1607: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1606, -1);  unsqueeze_1606 = None
        add_447: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_649, unsqueeze_1607);  mul_649 = unsqueeze_1607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_448: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_447, relu_189);  add_447 = relu_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_193: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_448);  add_448 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_248: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_193, arg353_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg353_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_449: "f32[256]" = torch.ops.aten.add.Tensor(arg355_1, 1e-05);  arg355_1 = None
        sqrt_201: "f32[256]" = torch.ops.aten.sqrt.default(add_449);  add_449 = None
        reciprocal_201: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_201);  sqrt_201 = None
        mul_650: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_201, 1);  reciprocal_201 = None
        unsqueeze_1608: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_1609: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1608, -1);  unsqueeze_1608 = None
        unsqueeze_1610: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_650, -1);  mul_650 = None
        unsqueeze_1611: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1610, -1);  unsqueeze_1610 = None
        sub_248: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_248, unsqueeze_1609);  convolution_248 = unsqueeze_1609 = None
        mul_651: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_1611);  sub_248 = unsqueeze_1611 = None
        unsqueeze_1612: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg356_1, -1);  arg356_1 = None
        unsqueeze_1613: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1612, -1);  unsqueeze_1612 = None
        mul_652: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_651, unsqueeze_1613);  mul_651 = unsqueeze_1613 = None
        unsqueeze_1614: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg357_1, -1);  arg357_1 = None
        unsqueeze_1615: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, -1);  unsqueeze_1614 = None
        add_450: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_652, unsqueeze_1615);  mul_652 = unsqueeze_1615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_194: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_450);  add_450 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_249: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_194, arg358_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_194 = arg358_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_451: "f32[512]" = torch.ops.aten.add.Tensor(arg360_1, 1e-05);  arg360_1 = None
        sqrt_202: "f32[512]" = torch.ops.aten.sqrt.default(add_451);  add_451 = None
        reciprocal_202: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_202);  sqrt_202 = None
        mul_653: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_202, 1);  reciprocal_202 = None
        unsqueeze_1616: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_1617: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1616, -1);  unsqueeze_1616 = None
        unsqueeze_1618: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_653, -1);  mul_653 = None
        unsqueeze_1619: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1618, -1);  unsqueeze_1618 = None
        sub_249: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_249, unsqueeze_1617);  convolution_249 = unsqueeze_1617 = None
        mul_654: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_1619);  sub_249 = unsqueeze_1619 = None
        unsqueeze_1620: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg361_1, -1);  arg361_1 = None
        unsqueeze_1621: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1620, -1);  unsqueeze_1620 = None
        mul_655: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_654, unsqueeze_1621);  mul_654 = unsqueeze_1621 = None
        unsqueeze_1622: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg362_1, -1);  arg362_1 = None
        unsqueeze_1623: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1622, -1);  unsqueeze_1622 = None
        add_452: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_655, unsqueeze_1623);  mul_655 = unsqueeze_1623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_195: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_452);  add_452 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_284: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_195, [8, 2, 256, 16, 16]);  relu_195 = None
        sum_142: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_284, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_48: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_142, [2, 3], True);  sum_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_250: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_48, arg363_1, arg364_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_48 = arg363_1 = arg364_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_453: "f32[128]" = torch.ops.aten.add.Tensor(arg366_1, 1e-05);  arg366_1 = None
        sqrt_203: "f32[128]" = torch.ops.aten.sqrt.default(add_453);  add_453 = None
        reciprocal_203: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_203);  sqrt_203 = None
        mul_656: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_203, 1);  reciprocal_203 = None
        unsqueeze_1624: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
        unsqueeze_1625: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1624, -1);  unsqueeze_1624 = None
        unsqueeze_1626: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_656, -1);  mul_656 = None
        unsqueeze_1627: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, -1);  unsqueeze_1626 = None
        sub_250: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_250, unsqueeze_1625);  convolution_250 = unsqueeze_1625 = None
        mul_657: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_1627);  sub_250 = unsqueeze_1627 = None
        unsqueeze_1628: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg367_1, -1);  arg367_1 = None
        unsqueeze_1629: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1628, -1);  unsqueeze_1628 = None
        mul_658: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_657, unsqueeze_1629);  mul_657 = unsqueeze_1629 = None
        unsqueeze_1630: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg368_1, -1);  arg368_1 = None
        unsqueeze_1631: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1630, -1);  unsqueeze_1630 = None
        add_454: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_658, unsqueeze_1631);  mul_658 = unsqueeze_1631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_196: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_454);  add_454 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_251: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_196, arg369_1, arg370_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_196 = arg369_1 = arg370_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_285: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_251, [8, 1, 2, -1]);  convolution_251 = None
        permute_48: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_47: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_48, [1], True)
        sub_251: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_48, amax_47);  permute_48 = amax_47 = None
        exp_47: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_251);  sub_251 = None
        sum_143: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_47, [1], True)
        div_47: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_47, sum_143);  exp_47 = sum_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_286: "f32[8, 512]" = torch.ops.aten.view.default(div_47, [8, -1]);  div_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_287: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_286, [8, -1, 1, 1]);  view_286 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_288: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_287, [8, 2, 256, 1, 1]);  view_287 = None
        mul_659: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_284, view_288);  view_284 = view_288 = None
        sum_144: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_659, [1]);  mul_659 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_252: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_144, arg371_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_144 = arg371_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_455: "f32[1024]" = torch.ops.aten.add.Tensor(arg373_1, 1e-05);  arg373_1 = None
        sqrt_204: "f32[1024]" = torch.ops.aten.sqrt.default(add_455);  add_455 = None
        reciprocal_204: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_204);  sqrt_204 = None
        mul_660: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_204, 1);  reciprocal_204 = None
        unsqueeze_1632: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg372_1, -1);  arg372_1 = None
        unsqueeze_1633: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1632, -1);  unsqueeze_1632 = None
        unsqueeze_1634: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_660, -1);  mul_660 = None
        unsqueeze_1635: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1634, -1);  unsqueeze_1634 = None
        sub_252: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_252, unsqueeze_1633);  convolution_252 = unsqueeze_1633 = None
        mul_661: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_1635);  sub_252 = unsqueeze_1635 = None
        unsqueeze_1636: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
        unsqueeze_1637: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1636, -1);  unsqueeze_1636 = None
        mul_662: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_661, unsqueeze_1637);  mul_661 = unsqueeze_1637 = None
        unsqueeze_1638: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg375_1, -1);  arg375_1 = None
        unsqueeze_1639: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, -1);  unsqueeze_1638 = None
        add_456: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_662, unsqueeze_1639);  mul_662 = unsqueeze_1639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_457: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_456, relu_193);  add_456 = relu_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_197: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_457);  add_457 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_253: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_197, arg376_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg376_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_458: "f32[256]" = torch.ops.aten.add.Tensor(arg378_1, 1e-05);  arg378_1 = None
        sqrt_205: "f32[256]" = torch.ops.aten.sqrt.default(add_458);  add_458 = None
        reciprocal_205: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_205);  sqrt_205 = None
        mul_663: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_205, 1);  reciprocal_205 = None
        unsqueeze_1640: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg377_1, -1);  arg377_1 = None
        unsqueeze_1641: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1640, -1);  unsqueeze_1640 = None
        unsqueeze_1642: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_663, -1);  mul_663 = None
        unsqueeze_1643: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1642, -1);  unsqueeze_1642 = None
        sub_253: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_253, unsqueeze_1641);  convolution_253 = unsqueeze_1641 = None
        mul_664: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_1643);  sub_253 = unsqueeze_1643 = None
        unsqueeze_1644: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_1645: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1644, -1);  unsqueeze_1644 = None
        mul_665: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_664, unsqueeze_1645);  mul_664 = unsqueeze_1645 = None
        unsqueeze_1646: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
        unsqueeze_1647: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1646, -1);  unsqueeze_1646 = None
        add_459: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_665, unsqueeze_1647);  mul_665 = unsqueeze_1647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_198: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_459);  add_459 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_254: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_198, arg381_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_198 = arg381_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_460: "f32[512]" = torch.ops.aten.add.Tensor(arg383_1, 1e-05);  arg383_1 = None
        sqrt_206: "f32[512]" = torch.ops.aten.sqrt.default(add_460);  add_460 = None
        reciprocal_206: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_206);  sqrt_206 = None
        mul_666: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_206, 1);  reciprocal_206 = None
        unsqueeze_1648: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg382_1, -1);  arg382_1 = None
        unsqueeze_1649: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1648, -1);  unsqueeze_1648 = None
        unsqueeze_1650: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_666, -1);  mul_666 = None
        unsqueeze_1651: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, -1);  unsqueeze_1650 = None
        sub_254: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_254, unsqueeze_1649);  convolution_254 = unsqueeze_1649 = None
        mul_667: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_1651);  sub_254 = unsqueeze_1651 = None
        unsqueeze_1652: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg384_1, -1);  arg384_1 = None
        unsqueeze_1653: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1652, -1);  unsqueeze_1652 = None
        mul_668: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_667, unsqueeze_1653);  mul_667 = unsqueeze_1653 = None
        unsqueeze_1654: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
        unsqueeze_1655: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1654, -1);  unsqueeze_1654 = None
        add_461: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_668, unsqueeze_1655);  mul_668 = unsqueeze_1655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_199: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_461);  add_461 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_290: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_199, [8, 2, 256, 16, 16]);  relu_199 = None
        sum_145: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_290, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_49: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_145, [2, 3], True);  sum_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_255: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_49, arg386_1, arg387_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_49 = arg386_1 = arg387_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_462: "f32[128]" = torch.ops.aten.add.Tensor(arg389_1, 1e-05);  arg389_1 = None
        sqrt_207: "f32[128]" = torch.ops.aten.sqrt.default(add_462);  add_462 = None
        reciprocal_207: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_207);  sqrt_207 = None
        mul_669: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_207, 1);  reciprocal_207 = None
        unsqueeze_1656: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
        unsqueeze_1657: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1656, -1);  unsqueeze_1656 = None
        unsqueeze_1658: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_669, -1);  mul_669 = None
        unsqueeze_1659: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1658, -1);  unsqueeze_1658 = None
        sub_255: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_255, unsqueeze_1657);  convolution_255 = unsqueeze_1657 = None
        mul_670: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_1659);  sub_255 = unsqueeze_1659 = None
        unsqueeze_1660: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg390_1, -1);  arg390_1 = None
        unsqueeze_1661: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1660, -1);  unsqueeze_1660 = None
        mul_671: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_670, unsqueeze_1661);  mul_670 = unsqueeze_1661 = None
        unsqueeze_1662: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg391_1, -1);  arg391_1 = None
        unsqueeze_1663: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, -1);  unsqueeze_1662 = None
        add_463: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_671, unsqueeze_1663);  mul_671 = unsqueeze_1663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_200: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_463);  add_463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_256: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_200, arg392_1, arg393_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_200 = arg392_1 = arg393_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_291: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_256, [8, 1, 2, -1]);  convolution_256 = None
        permute_49: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_48: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_49, [1], True)
        sub_256: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_49, amax_48);  permute_49 = amax_48 = None
        exp_48: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_256);  sub_256 = None
        sum_146: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_48, [1], True)
        div_48: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_48, sum_146);  exp_48 = sum_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_292: "f32[8, 512]" = torch.ops.aten.view.default(div_48, [8, -1]);  div_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_293: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_292, [8, -1, 1, 1]);  view_292 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_294: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_293, [8, 2, 256, 1, 1]);  view_293 = None
        mul_672: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_290, view_294);  view_290 = view_294 = None
        sum_147: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_672, [1]);  mul_672 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_257: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_147, arg394_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_147 = arg394_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_464: "f32[1024]" = torch.ops.aten.add.Tensor(arg396_1, 1e-05);  arg396_1 = None
        sqrt_208: "f32[1024]" = torch.ops.aten.sqrt.default(add_464);  add_464 = None
        reciprocal_208: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_208);  sqrt_208 = None
        mul_673: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_208, 1);  reciprocal_208 = None
        unsqueeze_1664: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
        unsqueeze_1665: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1664, -1);  unsqueeze_1664 = None
        unsqueeze_1666: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_673, -1);  mul_673 = None
        unsqueeze_1667: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1666, -1);  unsqueeze_1666 = None
        sub_257: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_257, unsqueeze_1665);  convolution_257 = unsqueeze_1665 = None
        mul_674: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_1667);  sub_257 = unsqueeze_1667 = None
        unsqueeze_1668: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
        unsqueeze_1669: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1668, -1);  unsqueeze_1668 = None
        mul_675: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_674, unsqueeze_1669);  mul_674 = unsqueeze_1669 = None
        unsqueeze_1670: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg398_1, -1);  arg398_1 = None
        unsqueeze_1671: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1670, -1);  unsqueeze_1670 = None
        add_465: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_675, unsqueeze_1671);  mul_675 = unsqueeze_1671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_466: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_465, relu_197);  add_465 = relu_197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_201: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_466);  add_466 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_258: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_201, arg399_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg399_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_467: "f32[256]" = torch.ops.aten.add.Tensor(arg401_1, 1e-05);  arg401_1 = None
        sqrt_209: "f32[256]" = torch.ops.aten.sqrt.default(add_467);  add_467 = None
        reciprocal_209: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_209);  sqrt_209 = None
        mul_676: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_209, 1);  reciprocal_209 = None
        unsqueeze_1672: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg400_1, -1);  arg400_1 = None
        unsqueeze_1673: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1672, -1);  unsqueeze_1672 = None
        unsqueeze_1674: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_676, -1);  mul_676 = None
        unsqueeze_1675: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1674, -1);  unsqueeze_1674 = None
        sub_258: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_258, unsqueeze_1673);  convolution_258 = unsqueeze_1673 = None
        mul_677: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_1675);  sub_258 = unsqueeze_1675 = None
        unsqueeze_1676: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg402_1, -1);  arg402_1 = None
        unsqueeze_1677: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1676, -1);  unsqueeze_1676 = None
        mul_678: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_677, unsqueeze_1677);  mul_677 = unsqueeze_1677 = None
        unsqueeze_1678: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg403_1, -1);  arg403_1 = None
        unsqueeze_1679: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1678, -1);  unsqueeze_1678 = None
        add_468: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_678, unsqueeze_1679);  mul_678 = unsqueeze_1679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_202: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_468);  add_468 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_259: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_202, arg404_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_202 = arg404_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_469: "f32[512]" = torch.ops.aten.add.Tensor(arg406_1, 1e-05);  arg406_1 = None
        sqrt_210: "f32[512]" = torch.ops.aten.sqrt.default(add_469);  add_469 = None
        reciprocal_210: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_210);  sqrt_210 = None
        mul_679: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_210, 1);  reciprocal_210 = None
        unsqueeze_1680: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg405_1, -1);  arg405_1 = None
        unsqueeze_1681: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1680, -1);  unsqueeze_1680 = None
        unsqueeze_1682: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_679, -1);  mul_679 = None
        unsqueeze_1683: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1682, -1);  unsqueeze_1682 = None
        sub_259: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_259, unsqueeze_1681);  convolution_259 = unsqueeze_1681 = None
        mul_680: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_1683);  sub_259 = unsqueeze_1683 = None
        unsqueeze_1684: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg407_1, -1);  arg407_1 = None
        unsqueeze_1685: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1684, -1);  unsqueeze_1684 = None
        mul_681: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_680, unsqueeze_1685);  mul_680 = unsqueeze_1685 = None
        unsqueeze_1686: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg408_1, -1);  arg408_1 = None
        unsqueeze_1687: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1686, -1);  unsqueeze_1686 = None
        add_470: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_681, unsqueeze_1687);  mul_681 = unsqueeze_1687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_203: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_470);  add_470 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_296: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_203, [8, 2, 256, 16, 16]);  relu_203 = None
        sum_148: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_296, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_50: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_148, [2, 3], True);  sum_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_260: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_50, arg409_1, arg410_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_50 = arg409_1 = arg410_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_471: "f32[128]" = torch.ops.aten.add.Tensor(arg412_1, 1e-05);  arg412_1 = None
        sqrt_211: "f32[128]" = torch.ops.aten.sqrt.default(add_471);  add_471 = None
        reciprocal_211: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_211);  sqrt_211 = None
        mul_682: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_211, 1);  reciprocal_211 = None
        unsqueeze_1688: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg411_1, -1);  arg411_1 = None
        unsqueeze_1689: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1688, -1);  unsqueeze_1688 = None
        unsqueeze_1690: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_682, -1);  mul_682 = None
        unsqueeze_1691: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1690, -1);  unsqueeze_1690 = None
        sub_260: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_260, unsqueeze_1689);  convolution_260 = unsqueeze_1689 = None
        mul_683: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_1691);  sub_260 = unsqueeze_1691 = None
        unsqueeze_1692: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg413_1, -1);  arg413_1 = None
        unsqueeze_1693: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1692, -1);  unsqueeze_1692 = None
        mul_684: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_683, unsqueeze_1693);  mul_683 = unsqueeze_1693 = None
        unsqueeze_1694: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_1695: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1694, -1);  unsqueeze_1694 = None
        add_472: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_684, unsqueeze_1695);  mul_684 = unsqueeze_1695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_204: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_472);  add_472 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_261: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_204, arg415_1, arg416_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_204 = arg415_1 = arg416_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_297: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_261, [8, 1, 2, -1]);  convolution_261 = None
        permute_50: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_49: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_50, [1], True)
        sub_261: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_50, amax_49);  permute_50 = amax_49 = None
        exp_49: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_261);  sub_261 = None
        sum_149: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_49, [1], True)
        div_49: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_49, sum_149);  exp_49 = sum_149 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_298: "f32[8, 512]" = torch.ops.aten.view.default(div_49, [8, -1]);  div_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_299: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_298, [8, -1, 1, 1]);  view_298 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_300: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_299, [8, 2, 256, 1, 1]);  view_299 = None
        mul_685: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_296, view_300);  view_296 = view_300 = None
        sum_150: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_685, [1]);  mul_685 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_262: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_150, arg417_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_150 = arg417_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_473: "f32[1024]" = torch.ops.aten.add.Tensor(arg419_1, 1e-05);  arg419_1 = None
        sqrt_212: "f32[1024]" = torch.ops.aten.sqrt.default(add_473);  add_473 = None
        reciprocal_212: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_212);  sqrt_212 = None
        mul_686: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_212, 1);  reciprocal_212 = None
        unsqueeze_1696: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg418_1, -1);  arg418_1 = None
        unsqueeze_1697: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1696, -1);  unsqueeze_1696 = None
        unsqueeze_1698: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_686, -1);  mul_686 = None
        unsqueeze_1699: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1698, -1);  unsqueeze_1698 = None
        sub_262: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_262, unsqueeze_1697);  convolution_262 = unsqueeze_1697 = None
        mul_687: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_1699);  sub_262 = unsqueeze_1699 = None
        unsqueeze_1700: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg420_1, -1);  arg420_1 = None
        unsqueeze_1701: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1700, -1);  unsqueeze_1700 = None
        mul_688: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_687, unsqueeze_1701);  mul_687 = unsqueeze_1701 = None
        unsqueeze_1702: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg421_1, -1);  arg421_1 = None
        unsqueeze_1703: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1702, -1);  unsqueeze_1702 = None
        add_474: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_688, unsqueeze_1703);  mul_688 = unsqueeze_1703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_475: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_474, relu_201);  add_474 = relu_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_205: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_475);  add_475 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_263: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_205, arg422_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg422_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_476: "f32[256]" = torch.ops.aten.add.Tensor(arg424_1, 1e-05);  arg424_1 = None
        sqrt_213: "f32[256]" = torch.ops.aten.sqrt.default(add_476);  add_476 = None
        reciprocal_213: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_213);  sqrt_213 = None
        mul_689: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_213, 1);  reciprocal_213 = None
        unsqueeze_1704: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg423_1, -1);  arg423_1 = None
        unsqueeze_1705: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1704, -1);  unsqueeze_1704 = None
        unsqueeze_1706: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_689, -1);  mul_689 = None
        unsqueeze_1707: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1706, -1);  unsqueeze_1706 = None
        sub_263: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_263, unsqueeze_1705);  convolution_263 = unsqueeze_1705 = None
        mul_690: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_1707);  sub_263 = unsqueeze_1707 = None
        unsqueeze_1708: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg425_1, -1);  arg425_1 = None
        unsqueeze_1709: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1708, -1);  unsqueeze_1708 = None
        mul_691: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_690, unsqueeze_1709);  mul_690 = unsqueeze_1709 = None
        unsqueeze_1710: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg426_1, -1);  arg426_1 = None
        unsqueeze_1711: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1710, -1);  unsqueeze_1710 = None
        add_477: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_691, unsqueeze_1711);  mul_691 = unsqueeze_1711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_206: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_477);  add_477 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_264: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_206, arg427_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_206 = arg427_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_478: "f32[512]" = torch.ops.aten.add.Tensor(arg429_1, 1e-05);  arg429_1 = None
        sqrt_214: "f32[512]" = torch.ops.aten.sqrt.default(add_478);  add_478 = None
        reciprocal_214: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_214);  sqrt_214 = None
        mul_692: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_214, 1);  reciprocal_214 = None
        unsqueeze_1712: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg428_1, -1);  arg428_1 = None
        unsqueeze_1713: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1712, -1);  unsqueeze_1712 = None
        unsqueeze_1714: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_692, -1);  mul_692 = None
        unsqueeze_1715: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1714, -1);  unsqueeze_1714 = None
        sub_264: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_264, unsqueeze_1713);  convolution_264 = unsqueeze_1713 = None
        mul_693: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_1715);  sub_264 = unsqueeze_1715 = None
        unsqueeze_1716: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
        unsqueeze_1717: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1716, -1);  unsqueeze_1716 = None
        mul_694: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_693, unsqueeze_1717);  mul_693 = unsqueeze_1717 = None
        unsqueeze_1718: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg431_1, -1);  arg431_1 = None
        unsqueeze_1719: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1718, -1);  unsqueeze_1718 = None
        add_479: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_694, unsqueeze_1719);  mul_694 = unsqueeze_1719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_207: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_479);  add_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_302: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_207, [8, 2, 256, 16, 16]);  relu_207 = None
        sum_151: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_302, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_51: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_151, [2, 3], True);  sum_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_265: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_51, arg432_1, arg433_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_51 = arg432_1 = arg433_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_480: "f32[128]" = torch.ops.aten.add.Tensor(arg435_1, 1e-05);  arg435_1 = None
        sqrt_215: "f32[128]" = torch.ops.aten.sqrt.default(add_480);  add_480 = None
        reciprocal_215: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_215);  sqrt_215 = None
        mul_695: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_215, 1);  reciprocal_215 = None
        unsqueeze_1720: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg434_1, -1);  arg434_1 = None
        unsqueeze_1721: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1720, -1);  unsqueeze_1720 = None
        unsqueeze_1722: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_695, -1);  mul_695 = None
        unsqueeze_1723: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1722, -1);  unsqueeze_1722 = None
        sub_265: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_265, unsqueeze_1721);  convolution_265 = unsqueeze_1721 = None
        mul_696: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_1723);  sub_265 = unsqueeze_1723 = None
        unsqueeze_1724: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg436_1, -1);  arg436_1 = None
        unsqueeze_1725: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1724, -1);  unsqueeze_1724 = None
        mul_697: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_696, unsqueeze_1725);  mul_696 = unsqueeze_1725 = None
        unsqueeze_1726: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg437_1, -1);  arg437_1 = None
        unsqueeze_1727: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1726, -1);  unsqueeze_1726 = None
        add_481: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_697, unsqueeze_1727);  mul_697 = unsqueeze_1727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_208: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_481);  add_481 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_266: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_208, arg438_1, arg439_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_208 = arg438_1 = arg439_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_303: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_266, [8, 1, 2, -1]);  convolution_266 = None
        permute_51: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_50: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_51, [1], True)
        sub_266: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_51, amax_50);  permute_51 = amax_50 = None
        exp_50: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_266);  sub_266 = None
        sum_152: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_50, [1], True)
        div_50: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_50, sum_152);  exp_50 = sum_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_304: "f32[8, 512]" = torch.ops.aten.view.default(div_50, [8, -1]);  div_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_305: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_304, [8, -1, 1, 1]);  view_304 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_306: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_305, [8, 2, 256, 1, 1]);  view_305 = None
        mul_698: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_302, view_306);  view_302 = view_306 = None
        sum_153: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_698, [1]);  mul_698 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_267: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_153, arg440_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_153 = arg440_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_482: "f32[1024]" = torch.ops.aten.add.Tensor(arg442_1, 1e-05);  arg442_1 = None
        sqrt_216: "f32[1024]" = torch.ops.aten.sqrt.default(add_482);  add_482 = None
        reciprocal_216: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_216);  sqrt_216 = None
        mul_699: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_216, 1);  reciprocal_216 = None
        unsqueeze_1728: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg441_1, -1);  arg441_1 = None
        unsqueeze_1729: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1728, -1);  unsqueeze_1728 = None
        unsqueeze_1730: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_699, -1);  mul_699 = None
        unsqueeze_1731: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1730, -1);  unsqueeze_1730 = None
        sub_267: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_267, unsqueeze_1729);  convolution_267 = unsqueeze_1729 = None
        mul_700: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_1731);  sub_267 = unsqueeze_1731 = None
        unsqueeze_1732: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg443_1, -1);  arg443_1 = None
        unsqueeze_1733: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1732, -1);  unsqueeze_1732 = None
        mul_701: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_700, unsqueeze_1733);  mul_700 = unsqueeze_1733 = None
        unsqueeze_1734: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg444_1, -1);  arg444_1 = None
        unsqueeze_1735: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1734, -1);  unsqueeze_1734 = None
        add_483: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_701, unsqueeze_1735);  mul_701 = unsqueeze_1735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_484: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_483, relu_205);  add_483 = relu_205 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_209: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_484);  add_484 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_268: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_209, arg445_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg445_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_485: "f32[256]" = torch.ops.aten.add.Tensor(arg447_1, 1e-05);  arg447_1 = None
        sqrt_217: "f32[256]" = torch.ops.aten.sqrt.default(add_485);  add_485 = None
        reciprocal_217: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_217);  sqrt_217 = None
        mul_702: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_217, 1);  reciprocal_217 = None
        unsqueeze_1736: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg446_1, -1);  arg446_1 = None
        unsqueeze_1737: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1736, -1);  unsqueeze_1736 = None
        unsqueeze_1738: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_702, -1);  mul_702 = None
        unsqueeze_1739: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1738, -1);  unsqueeze_1738 = None
        sub_268: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_268, unsqueeze_1737);  convolution_268 = unsqueeze_1737 = None
        mul_703: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_1739);  sub_268 = unsqueeze_1739 = None
        unsqueeze_1740: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg448_1, -1);  arg448_1 = None
        unsqueeze_1741: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1740, -1);  unsqueeze_1740 = None
        mul_704: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_703, unsqueeze_1741);  mul_703 = unsqueeze_1741 = None
        unsqueeze_1742: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
        unsqueeze_1743: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1742, -1);  unsqueeze_1742 = None
        add_486: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_704, unsqueeze_1743);  mul_704 = unsqueeze_1743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_210: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_486);  add_486 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_269: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_210, arg450_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_210 = arg450_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_487: "f32[512]" = torch.ops.aten.add.Tensor(arg452_1, 1e-05);  arg452_1 = None
        sqrt_218: "f32[512]" = torch.ops.aten.sqrt.default(add_487);  add_487 = None
        reciprocal_218: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_218);  sqrt_218 = None
        mul_705: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_218, 1);  reciprocal_218 = None
        unsqueeze_1744: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg451_1, -1);  arg451_1 = None
        unsqueeze_1745: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1744, -1);  unsqueeze_1744 = None
        unsqueeze_1746: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_705, -1);  mul_705 = None
        unsqueeze_1747: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1746, -1);  unsqueeze_1746 = None
        sub_269: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_269, unsqueeze_1745);  convolution_269 = unsqueeze_1745 = None
        mul_706: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_1747);  sub_269 = unsqueeze_1747 = None
        unsqueeze_1748: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg453_1, -1);  arg453_1 = None
        unsqueeze_1749: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1748, -1);  unsqueeze_1748 = None
        mul_707: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_706, unsqueeze_1749);  mul_706 = unsqueeze_1749 = None
        unsqueeze_1750: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
        unsqueeze_1751: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1750, -1);  unsqueeze_1750 = None
        add_488: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_707, unsqueeze_1751);  mul_707 = unsqueeze_1751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_211: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_488);  add_488 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_308: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_211, [8, 2, 256, 16, 16]);  relu_211 = None
        sum_154: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_308, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_52: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_154, [2, 3], True);  sum_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_270: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_52, arg455_1, arg456_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_52 = arg455_1 = arg456_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_489: "f32[128]" = torch.ops.aten.add.Tensor(arg458_1, 1e-05);  arg458_1 = None
        sqrt_219: "f32[128]" = torch.ops.aten.sqrt.default(add_489);  add_489 = None
        reciprocal_219: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_219);  sqrt_219 = None
        mul_708: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_219, 1);  reciprocal_219 = None
        unsqueeze_1752: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg457_1, -1);  arg457_1 = None
        unsqueeze_1753: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1752, -1);  unsqueeze_1752 = None
        unsqueeze_1754: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_708, -1);  mul_708 = None
        unsqueeze_1755: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1754, -1);  unsqueeze_1754 = None
        sub_270: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_270, unsqueeze_1753);  convolution_270 = unsqueeze_1753 = None
        mul_709: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_1755);  sub_270 = unsqueeze_1755 = None
        unsqueeze_1756: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg459_1, -1);  arg459_1 = None
        unsqueeze_1757: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1756, -1);  unsqueeze_1756 = None
        mul_710: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_709, unsqueeze_1757);  mul_709 = unsqueeze_1757 = None
        unsqueeze_1758: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg460_1, -1);  arg460_1 = None
        unsqueeze_1759: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1758, -1);  unsqueeze_1758 = None
        add_490: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_710, unsqueeze_1759);  mul_710 = unsqueeze_1759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_212: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_490);  add_490 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_271: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_212, arg461_1, arg462_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_212 = arg461_1 = arg462_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_309: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_271, [8, 1, 2, -1]);  convolution_271 = None
        permute_52: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_51: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_52, [1], True)
        sub_271: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_52, amax_51);  permute_52 = amax_51 = None
        exp_51: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_271);  sub_271 = None
        sum_155: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_51, [1], True)
        div_51: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_51, sum_155);  exp_51 = sum_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_310: "f32[8, 512]" = torch.ops.aten.view.default(div_51, [8, -1]);  div_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_311: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_310, [8, -1, 1, 1]);  view_310 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_312: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_311, [8, 2, 256, 1, 1]);  view_311 = None
        mul_711: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_308, view_312);  view_308 = view_312 = None
        sum_156: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_711, [1]);  mul_711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_272: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_156, arg463_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_156 = arg463_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_491: "f32[1024]" = torch.ops.aten.add.Tensor(arg465_1, 1e-05);  arg465_1 = None
        sqrt_220: "f32[1024]" = torch.ops.aten.sqrt.default(add_491);  add_491 = None
        reciprocal_220: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_220);  sqrt_220 = None
        mul_712: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_220, 1);  reciprocal_220 = None
        unsqueeze_1760: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
        unsqueeze_1761: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1760, -1);  unsqueeze_1760 = None
        unsqueeze_1762: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_712, -1);  mul_712 = None
        unsqueeze_1763: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1762, -1);  unsqueeze_1762 = None
        sub_272: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_272, unsqueeze_1761);  convolution_272 = unsqueeze_1761 = None
        mul_713: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_1763);  sub_272 = unsqueeze_1763 = None
        unsqueeze_1764: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg466_1, -1);  arg466_1 = None
        unsqueeze_1765: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1764, -1);  unsqueeze_1764 = None
        mul_714: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_713, unsqueeze_1765);  mul_713 = unsqueeze_1765 = None
        unsqueeze_1766: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg467_1, -1);  arg467_1 = None
        unsqueeze_1767: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1766, -1);  unsqueeze_1766 = None
        add_492: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_714, unsqueeze_1767);  mul_714 = unsqueeze_1767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_493: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_492, relu_209);  add_492 = relu_209 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_213: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_493);  add_493 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_273: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_213, arg468_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg468_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_494: "f32[256]" = torch.ops.aten.add.Tensor(arg470_1, 1e-05);  arg470_1 = None
        sqrt_221: "f32[256]" = torch.ops.aten.sqrt.default(add_494);  add_494 = None
        reciprocal_221: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_221);  sqrt_221 = None
        mul_715: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_221, 1);  reciprocal_221 = None
        unsqueeze_1768: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
        unsqueeze_1769: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1768, -1);  unsqueeze_1768 = None
        unsqueeze_1770: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_715, -1);  mul_715 = None
        unsqueeze_1771: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1770, -1);  unsqueeze_1770 = None
        sub_273: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_273, unsqueeze_1769);  convolution_273 = unsqueeze_1769 = None
        mul_716: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_1771);  sub_273 = unsqueeze_1771 = None
        unsqueeze_1772: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg471_1, -1);  arg471_1 = None
        unsqueeze_1773: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1772, -1);  unsqueeze_1772 = None
        mul_717: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_716, unsqueeze_1773);  mul_716 = unsqueeze_1773 = None
        unsqueeze_1774: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg472_1, -1);  arg472_1 = None
        unsqueeze_1775: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1774, -1);  unsqueeze_1774 = None
        add_495: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_717, unsqueeze_1775);  mul_717 = unsqueeze_1775 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_214: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_495);  add_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_274: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_214, arg473_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_214 = arg473_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_496: "f32[512]" = torch.ops.aten.add.Tensor(arg475_1, 1e-05);  arg475_1 = None
        sqrt_222: "f32[512]" = torch.ops.aten.sqrt.default(add_496);  add_496 = None
        reciprocal_222: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_222);  sqrt_222 = None
        mul_718: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_222, 1);  reciprocal_222 = None
        unsqueeze_1776: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg474_1, -1);  arg474_1 = None
        unsqueeze_1777: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1776, -1);  unsqueeze_1776 = None
        unsqueeze_1778: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_718, -1);  mul_718 = None
        unsqueeze_1779: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1778, -1);  unsqueeze_1778 = None
        sub_274: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_274, unsqueeze_1777);  convolution_274 = unsqueeze_1777 = None
        mul_719: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_1779);  sub_274 = unsqueeze_1779 = None
        unsqueeze_1780: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg476_1, -1);  arg476_1 = None
        unsqueeze_1781: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1780, -1);  unsqueeze_1780 = None
        mul_720: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_719, unsqueeze_1781);  mul_719 = unsqueeze_1781 = None
        unsqueeze_1782: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg477_1, -1);  arg477_1 = None
        unsqueeze_1783: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1782, -1);  unsqueeze_1782 = None
        add_497: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_720, unsqueeze_1783);  mul_720 = unsqueeze_1783 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_215: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_497);  add_497 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_314: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_215, [8, 2, 256, 16, 16]);  relu_215 = None
        sum_157: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_314, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_53: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_157, [2, 3], True);  sum_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_275: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_53, arg478_1, arg479_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_53 = arg478_1 = arg479_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_498: "f32[128]" = torch.ops.aten.add.Tensor(arg481_1, 1e-05);  arg481_1 = None
        sqrt_223: "f32[128]" = torch.ops.aten.sqrt.default(add_498);  add_498 = None
        reciprocal_223: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_223);  sqrt_223 = None
        mul_721: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_223, 1);  reciprocal_223 = None
        unsqueeze_1784: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg480_1, -1);  arg480_1 = None
        unsqueeze_1785: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1784, -1);  unsqueeze_1784 = None
        unsqueeze_1786: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_721, -1);  mul_721 = None
        unsqueeze_1787: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1786, -1);  unsqueeze_1786 = None
        sub_275: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_275, unsqueeze_1785);  convolution_275 = unsqueeze_1785 = None
        mul_722: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_1787);  sub_275 = unsqueeze_1787 = None
        unsqueeze_1788: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg482_1, -1);  arg482_1 = None
        unsqueeze_1789: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1788, -1);  unsqueeze_1788 = None
        mul_723: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_722, unsqueeze_1789);  mul_722 = unsqueeze_1789 = None
        unsqueeze_1790: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg483_1, -1);  arg483_1 = None
        unsqueeze_1791: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1790, -1);  unsqueeze_1790 = None
        add_499: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_723, unsqueeze_1791);  mul_723 = unsqueeze_1791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_216: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_499);  add_499 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_276: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_216, arg484_1, arg485_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_216 = arg484_1 = arg485_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_315: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_276, [8, 1, 2, -1]);  convolution_276 = None
        permute_53: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_52: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_53, [1], True)
        sub_276: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_53, amax_52);  permute_53 = amax_52 = None
        exp_52: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_276);  sub_276 = None
        sum_158: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_52, [1], True)
        div_52: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_52, sum_158);  exp_52 = sum_158 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_316: "f32[8, 512]" = torch.ops.aten.view.default(div_52, [8, -1]);  div_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_317: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_316, [8, -1, 1, 1]);  view_316 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_318: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_317, [8, 2, 256, 1, 1]);  view_317 = None
        mul_724: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_314, view_318);  view_314 = view_318 = None
        sum_159: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_724, [1]);  mul_724 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_277: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_159, arg486_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_159 = arg486_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_500: "f32[1024]" = torch.ops.aten.add.Tensor(arg488_1, 1e-05);  arg488_1 = None
        sqrt_224: "f32[1024]" = torch.ops.aten.sqrt.default(add_500);  add_500 = None
        reciprocal_224: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_224);  sqrt_224 = None
        mul_725: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_224, 1);  reciprocal_224 = None
        unsqueeze_1792: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg487_1, -1);  arg487_1 = None
        unsqueeze_1793: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1792, -1);  unsqueeze_1792 = None
        unsqueeze_1794: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_725, -1);  mul_725 = None
        unsqueeze_1795: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1794, -1);  unsqueeze_1794 = None
        sub_277: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_277, unsqueeze_1793);  convolution_277 = unsqueeze_1793 = None
        mul_726: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_1795);  sub_277 = unsqueeze_1795 = None
        unsqueeze_1796: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg489_1, -1);  arg489_1 = None
        unsqueeze_1797: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1796, -1);  unsqueeze_1796 = None
        mul_727: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_726, unsqueeze_1797);  mul_726 = unsqueeze_1797 = None
        unsqueeze_1798: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg490_1, -1);  arg490_1 = None
        unsqueeze_1799: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1798, -1);  unsqueeze_1798 = None
        add_501: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_727, unsqueeze_1799);  mul_727 = unsqueeze_1799 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_502: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_501, relu_213);  add_501 = relu_213 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_217: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_502);  add_502 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_278: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_217, arg491_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg491_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_503: "f32[256]" = torch.ops.aten.add.Tensor(arg493_1, 1e-05);  arg493_1 = None
        sqrt_225: "f32[256]" = torch.ops.aten.sqrt.default(add_503);  add_503 = None
        reciprocal_225: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_225);  sqrt_225 = None
        mul_728: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_225, 1);  reciprocal_225 = None
        unsqueeze_1800: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg492_1, -1);  arg492_1 = None
        unsqueeze_1801: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1800, -1);  unsqueeze_1800 = None
        unsqueeze_1802: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_728, -1);  mul_728 = None
        unsqueeze_1803: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1802, -1);  unsqueeze_1802 = None
        sub_278: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_278, unsqueeze_1801);  convolution_278 = unsqueeze_1801 = None
        mul_729: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_1803);  sub_278 = unsqueeze_1803 = None
        unsqueeze_1804: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg494_1, -1);  arg494_1 = None
        unsqueeze_1805: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1804, -1);  unsqueeze_1804 = None
        mul_730: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_729, unsqueeze_1805);  mul_729 = unsqueeze_1805 = None
        unsqueeze_1806: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg495_1, -1);  arg495_1 = None
        unsqueeze_1807: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1806, -1);  unsqueeze_1806 = None
        add_504: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_730, unsqueeze_1807);  mul_730 = unsqueeze_1807 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_218: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_504);  add_504 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_279: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_218, arg496_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_218 = arg496_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_505: "f32[512]" = torch.ops.aten.add.Tensor(arg498_1, 1e-05);  arg498_1 = None
        sqrt_226: "f32[512]" = torch.ops.aten.sqrt.default(add_505);  add_505 = None
        reciprocal_226: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_226);  sqrt_226 = None
        mul_731: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_226, 1);  reciprocal_226 = None
        unsqueeze_1808: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg497_1, -1);  arg497_1 = None
        unsqueeze_1809: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1808, -1);  unsqueeze_1808 = None
        unsqueeze_1810: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_731, -1);  mul_731 = None
        unsqueeze_1811: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1810, -1);  unsqueeze_1810 = None
        sub_279: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_279, unsqueeze_1809);  convolution_279 = unsqueeze_1809 = None
        mul_732: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_1811);  sub_279 = unsqueeze_1811 = None
        unsqueeze_1812: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg499_1, -1);  arg499_1 = None
        unsqueeze_1813: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1812, -1);  unsqueeze_1812 = None
        mul_733: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_732, unsqueeze_1813);  mul_732 = unsqueeze_1813 = None
        unsqueeze_1814: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg500_1, -1);  arg500_1 = None
        unsqueeze_1815: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1814, -1);  unsqueeze_1814 = None
        add_506: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_733, unsqueeze_1815);  mul_733 = unsqueeze_1815 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_219: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_506);  add_506 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_320: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_219, [8, 2, 256, 16, 16]);  relu_219 = None
        sum_160: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_320, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_54: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_160, [2, 3], True);  sum_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_280: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_54, arg501_1, arg502_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_54 = arg501_1 = arg502_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_507: "f32[128]" = torch.ops.aten.add.Tensor(arg504_1, 1e-05);  arg504_1 = None
        sqrt_227: "f32[128]" = torch.ops.aten.sqrt.default(add_507);  add_507 = None
        reciprocal_227: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_227);  sqrt_227 = None
        mul_734: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_227, 1);  reciprocal_227 = None
        unsqueeze_1816: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg503_1, -1);  arg503_1 = None
        unsqueeze_1817: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1816, -1);  unsqueeze_1816 = None
        unsqueeze_1818: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_734, -1);  mul_734 = None
        unsqueeze_1819: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1818, -1);  unsqueeze_1818 = None
        sub_280: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_280, unsqueeze_1817);  convolution_280 = unsqueeze_1817 = None
        mul_735: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_1819);  sub_280 = unsqueeze_1819 = None
        unsqueeze_1820: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg505_1, -1);  arg505_1 = None
        unsqueeze_1821: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1820, -1);  unsqueeze_1820 = None
        mul_736: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_735, unsqueeze_1821);  mul_735 = unsqueeze_1821 = None
        unsqueeze_1822: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg506_1, -1);  arg506_1 = None
        unsqueeze_1823: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1822, -1);  unsqueeze_1822 = None
        add_508: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_736, unsqueeze_1823);  mul_736 = unsqueeze_1823 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_220: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_508);  add_508 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_281: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_220, arg507_1, arg508_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_220 = arg507_1 = arg508_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_321: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_281, [8, 1, 2, -1]);  convolution_281 = None
        permute_54: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_53: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_54, [1], True)
        sub_281: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_54, amax_53);  permute_54 = amax_53 = None
        exp_53: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_281);  sub_281 = None
        sum_161: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_53, [1], True)
        div_53: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_53, sum_161);  exp_53 = sum_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_322: "f32[8, 512]" = torch.ops.aten.view.default(div_53, [8, -1]);  div_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_323: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_322, [8, -1, 1, 1]);  view_322 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_324: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_323, [8, 2, 256, 1, 1]);  view_323 = None
        mul_737: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_320, view_324);  view_320 = view_324 = None
        sum_162: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_737, [1]);  mul_737 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_282: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_162, arg509_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_162 = arg509_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_509: "f32[1024]" = torch.ops.aten.add.Tensor(arg511_1, 1e-05);  arg511_1 = None
        sqrt_228: "f32[1024]" = torch.ops.aten.sqrt.default(add_509);  add_509 = None
        reciprocal_228: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_228);  sqrt_228 = None
        mul_738: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_228, 1);  reciprocal_228 = None
        unsqueeze_1824: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg510_1, -1);  arg510_1 = None
        unsqueeze_1825: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1824, -1);  unsqueeze_1824 = None
        unsqueeze_1826: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_738, -1);  mul_738 = None
        unsqueeze_1827: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1826, -1);  unsqueeze_1826 = None
        sub_282: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_282, unsqueeze_1825);  convolution_282 = unsqueeze_1825 = None
        mul_739: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_1827);  sub_282 = unsqueeze_1827 = None
        unsqueeze_1828: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg512_1, -1);  arg512_1 = None
        unsqueeze_1829: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1828, -1);  unsqueeze_1828 = None
        mul_740: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_739, unsqueeze_1829);  mul_739 = unsqueeze_1829 = None
        unsqueeze_1830: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg513_1, -1);  arg513_1 = None
        unsqueeze_1831: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1830, -1);  unsqueeze_1830 = None
        add_510: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_740, unsqueeze_1831);  mul_740 = unsqueeze_1831 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_511: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_510, relu_217);  add_510 = relu_217 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_221: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_511);  add_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_283: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_221, arg514_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg514_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_512: "f32[256]" = torch.ops.aten.add.Tensor(arg516_1, 1e-05);  arg516_1 = None
        sqrt_229: "f32[256]" = torch.ops.aten.sqrt.default(add_512);  add_512 = None
        reciprocal_229: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_229);  sqrt_229 = None
        mul_741: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_229, 1);  reciprocal_229 = None
        unsqueeze_1832: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg515_1, -1);  arg515_1 = None
        unsqueeze_1833: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1832, -1);  unsqueeze_1832 = None
        unsqueeze_1834: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_741, -1);  mul_741 = None
        unsqueeze_1835: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1834, -1);  unsqueeze_1834 = None
        sub_283: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_283, unsqueeze_1833);  convolution_283 = unsqueeze_1833 = None
        mul_742: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_1835);  sub_283 = unsqueeze_1835 = None
        unsqueeze_1836: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg517_1, -1);  arg517_1 = None
        unsqueeze_1837: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1836, -1);  unsqueeze_1836 = None
        mul_743: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_742, unsqueeze_1837);  mul_742 = unsqueeze_1837 = None
        unsqueeze_1838: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg518_1, -1);  arg518_1 = None
        unsqueeze_1839: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1838, -1);  unsqueeze_1838 = None
        add_513: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_743, unsqueeze_1839);  mul_743 = unsqueeze_1839 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_222: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_513);  add_513 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_284: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_222, arg519_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_222 = arg519_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_514: "f32[512]" = torch.ops.aten.add.Tensor(arg521_1, 1e-05);  arg521_1 = None
        sqrt_230: "f32[512]" = torch.ops.aten.sqrt.default(add_514);  add_514 = None
        reciprocal_230: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_230);  sqrt_230 = None
        mul_744: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_230, 1);  reciprocal_230 = None
        unsqueeze_1840: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg520_1, -1);  arg520_1 = None
        unsqueeze_1841: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1840, -1);  unsqueeze_1840 = None
        unsqueeze_1842: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_744, -1);  mul_744 = None
        unsqueeze_1843: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1842, -1);  unsqueeze_1842 = None
        sub_284: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_284, unsqueeze_1841);  convolution_284 = unsqueeze_1841 = None
        mul_745: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_1843);  sub_284 = unsqueeze_1843 = None
        unsqueeze_1844: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg522_1, -1);  arg522_1 = None
        unsqueeze_1845: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1844, -1);  unsqueeze_1844 = None
        mul_746: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_745, unsqueeze_1845);  mul_745 = unsqueeze_1845 = None
        unsqueeze_1846: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg523_1, -1);  arg523_1 = None
        unsqueeze_1847: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1846, -1);  unsqueeze_1846 = None
        add_515: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_746, unsqueeze_1847);  mul_746 = unsqueeze_1847 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_223: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_515);  add_515 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_326: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_223, [8, 2, 256, 16, 16]);  relu_223 = None
        sum_163: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_326, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_55: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_163, [2, 3], True);  sum_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_285: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_55, arg524_1, arg525_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_55 = arg524_1 = arg525_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_516: "f32[128]" = torch.ops.aten.add.Tensor(arg527_1, 1e-05);  arg527_1 = None
        sqrt_231: "f32[128]" = torch.ops.aten.sqrt.default(add_516);  add_516 = None
        reciprocal_231: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_231);  sqrt_231 = None
        mul_747: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_231, 1);  reciprocal_231 = None
        unsqueeze_1848: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg526_1, -1);  arg526_1 = None
        unsqueeze_1849: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1848, -1);  unsqueeze_1848 = None
        unsqueeze_1850: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_747, -1);  mul_747 = None
        unsqueeze_1851: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1850, -1);  unsqueeze_1850 = None
        sub_285: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_285, unsqueeze_1849);  convolution_285 = unsqueeze_1849 = None
        mul_748: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_1851);  sub_285 = unsqueeze_1851 = None
        unsqueeze_1852: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg528_1, -1);  arg528_1 = None
        unsqueeze_1853: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1852, -1);  unsqueeze_1852 = None
        mul_749: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_748, unsqueeze_1853);  mul_748 = unsqueeze_1853 = None
        unsqueeze_1854: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg529_1, -1);  arg529_1 = None
        unsqueeze_1855: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1854, -1);  unsqueeze_1854 = None
        add_517: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_749, unsqueeze_1855);  mul_749 = unsqueeze_1855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_224: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_517);  add_517 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_286: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_224, arg530_1, arg531_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_224 = arg530_1 = arg531_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_327: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_286, [8, 1, 2, -1]);  convolution_286 = None
        permute_55: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_54: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_55, [1], True)
        sub_286: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_55, amax_54);  permute_55 = amax_54 = None
        exp_54: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_286);  sub_286 = None
        sum_164: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_54, [1], True)
        div_54: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_54, sum_164);  exp_54 = sum_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_328: "f32[8, 512]" = torch.ops.aten.view.default(div_54, [8, -1]);  div_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_329: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_328, [8, -1, 1, 1]);  view_328 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_330: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_329, [8, 2, 256, 1, 1]);  view_329 = None
        mul_750: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_326, view_330);  view_326 = view_330 = None
        sum_165: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_750, [1]);  mul_750 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_287: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_165, arg532_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_165 = arg532_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_518: "f32[1024]" = torch.ops.aten.add.Tensor(arg534_1, 1e-05);  arg534_1 = None
        sqrt_232: "f32[1024]" = torch.ops.aten.sqrt.default(add_518);  add_518 = None
        reciprocal_232: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_232);  sqrt_232 = None
        mul_751: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_232, 1);  reciprocal_232 = None
        unsqueeze_1856: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg533_1, -1);  arg533_1 = None
        unsqueeze_1857: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1856, -1);  unsqueeze_1856 = None
        unsqueeze_1858: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_751, -1);  mul_751 = None
        unsqueeze_1859: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1858, -1);  unsqueeze_1858 = None
        sub_287: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_287, unsqueeze_1857);  convolution_287 = unsqueeze_1857 = None
        mul_752: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_1859);  sub_287 = unsqueeze_1859 = None
        unsqueeze_1860: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg535_1, -1);  arg535_1 = None
        unsqueeze_1861: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1860, -1);  unsqueeze_1860 = None
        mul_753: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_752, unsqueeze_1861);  mul_752 = unsqueeze_1861 = None
        unsqueeze_1862: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg536_1, -1);  arg536_1 = None
        unsqueeze_1863: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1862, -1);  unsqueeze_1862 = None
        add_519: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_753, unsqueeze_1863);  mul_753 = unsqueeze_1863 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_520: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_519, relu_221);  add_519 = relu_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_225: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_520);  add_520 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_288: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_225, arg537_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg537_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_521: "f32[256]" = torch.ops.aten.add.Tensor(arg539_1, 1e-05);  arg539_1 = None
        sqrt_233: "f32[256]" = torch.ops.aten.sqrt.default(add_521);  add_521 = None
        reciprocal_233: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_233);  sqrt_233 = None
        mul_754: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_233, 1);  reciprocal_233 = None
        unsqueeze_1864: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg538_1, -1);  arg538_1 = None
        unsqueeze_1865: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1864, -1);  unsqueeze_1864 = None
        unsqueeze_1866: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_754, -1);  mul_754 = None
        unsqueeze_1867: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1866, -1);  unsqueeze_1866 = None
        sub_288: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_288, unsqueeze_1865);  convolution_288 = unsqueeze_1865 = None
        mul_755: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_1867);  sub_288 = unsqueeze_1867 = None
        unsqueeze_1868: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg540_1, -1);  arg540_1 = None
        unsqueeze_1869: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1868, -1);  unsqueeze_1868 = None
        mul_756: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_755, unsqueeze_1869);  mul_755 = unsqueeze_1869 = None
        unsqueeze_1870: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg541_1, -1);  arg541_1 = None
        unsqueeze_1871: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1870, -1);  unsqueeze_1870 = None
        add_522: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_756, unsqueeze_1871);  mul_756 = unsqueeze_1871 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_226: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_522);  add_522 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_289: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_226, arg542_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_226 = arg542_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_523: "f32[512]" = torch.ops.aten.add.Tensor(arg544_1, 1e-05);  arg544_1 = None
        sqrt_234: "f32[512]" = torch.ops.aten.sqrt.default(add_523);  add_523 = None
        reciprocal_234: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_234);  sqrt_234 = None
        mul_757: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_234, 1);  reciprocal_234 = None
        unsqueeze_1872: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg543_1, -1);  arg543_1 = None
        unsqueeze_1873: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1872, -1);  unsqueeze_1872 = None
        unsqueeze_1874: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_757, -1);  mul_757 = None
        unsqueeze_1875: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1874, -1);  unsqueeze_1874 = None
        sub_289: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_289, unsqueeze_1873);  convolution_289 = unsqueeze_1873 = None
        mul_758: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_1875);  sub_289 = unsqueeze_1875 = None
        unsqueeze_1876: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg545_1, -1);  arg545_1 = None
        unsqueeze_1877: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1876, -1);  unsqueeze_1876 = None
        mul_759: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_758, unsqueeze_1877);  mul_758 = unsqueeze_1877 = None
        unsqueeze_1878: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg546_1, -1);  arg546_1 = None
        unsqueeze_1879: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1878, -1);  unsqueeze_1878 = None
        add_524: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_759, unsqueeze_1879);  mul_759 = unsqueeze_1879 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_227: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_524);  add_524 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_332: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_227, [8, 2, 256, 16, 16]);  relu_227 = None
        sum_166: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_332, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_56: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_166, [2, 3], True);  sum_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_290: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_56, arg547_1, arg548_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_56 = arg547_1 = arg548_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_525: "f32[128]" = torch.ops.aten.add.Tensor(arg550_1, 1e-05);  arg550_1 = None
        sqrt_235: "f32[128]" = torch.ops.aten.sqrt.default(add_525);  add_525 = None
        reciprocal_235: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_235);  sqrt_235 = None
        mul_760: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_235, 1);  reciprocal_235 = None
        unsqueeze_1880: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg549_1, -1);  arg549_1 = None
        unsqueeze_1881: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1880, -1);  unsqueeze_1880 = None
        unsqueeze_1882: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_760, -1);  mul_760 = None
        unsqueeze_1883: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1882, -1);  unsqueeze_1882 = None
        sub_290: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_290, unsqueeze_1881);  convolution_290 = unsqueeze_1881 = None
        mul_761: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_1883);  sub_290 = unsqueeze_1883 = None
        unsqueeze_1884: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg551_1, -1);  arg551_1 = None
        unsqueeze_1885: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1884, -1);  unsqueeze_1884 = None
        mul_762: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_761, unsqueeze_1885);  mul_761 = unsqueeze_1885 = None
        unsqueeze_1886: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg552_1, -1);  arg552_1 = None
        unsqueeze_1887: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1886, -1);  unsqueeze_1886 = None
        add_526: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_762, unsqueeze_1887);  mul_762 = unsqueeze_1887 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_228: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_526);  add_526 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_291: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_228, arg553_1, arg554_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_228 = arg553_1 = arg554_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_333: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_291, [8, 1, 2, -1]);  convolution_291 = None
        permute_56: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_55: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_56, [1], True)
        sub_291: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_56, amax_55);  permute_56 = amax_55 = None
        exp_55: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_291);  sub_291 = None
        sum_167: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_55, [1], True)
        div_55: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_55, sum_167);  exp_55 = sum_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_334: "f32[8, 512]" = torch.ops.aten.view.default(div_55, [8, -1]);  div_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_335: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_334, [8, -1, 1, 1]);  view_334 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_336: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_335, [8, 2, 256, 1, 1]);  view_335 = None
        mul_763: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_332, view_336);  view_332 = view_336 = None
        sum_168: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_763, [1]);  mul_763 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_292: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_168, arg555_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_168 = arg555_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_527: "f32[1024]" = torch.ops.aten.add.Tensor(arg557_1, 1e-05);  arg557_1 = None
        sqrt_236: "f32[1024]" = torch.ops.aten.sqrt.default(add_527);  add_527 = None
        reciprocal_236: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_236);  sqrt_236 = None
        mul_764: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_236, 1);  reciprocal_236 = None
        unsqueeze_1888: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg556_1, -1);  arg556_1 = None
        unsqueeze_1889: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1888, -1);  unsqueeze_1888 = None
        unsqueeze_1890: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_764, -1);  mul_764 = None
        unsqueeze_1891: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1890, -1);  unsqueeze_1890 = None
        sub_292: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_292, unsqueeze_1889);  convolution_292 = unsqueeze_1889 = None
        mul_765: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_1891);  sub_292 = unsqueeze_1891 = None
        unsqueeze_1892: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg558_1, -1);  arg558_1 = None
        unsqueeze_1893: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1892, -1);  unsqueeze_1892 = None
        mul_766: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_765, unsqueeze_1893);  mul_765 = unsqueeze_1893 = None
        unsqueeze_1894: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg559_1, -1);  arg559_1 = None
        unsqueeze_1895: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1894, -1);  unsqueeze_1894 = None
        add_528: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_766, unsqueeze_1895);  mul_766 = unsqueeze_1895 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_529: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_528, relu_225);  add_528 = relu_225 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_229: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_529);  add_529 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_293: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_229, arg560_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg560_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_530: "f32[256]" = torch.ops.aten.add.Tensor(arg562_1, 1e-05);  arg562_1 = None
        sqrt_237: "f32[256]" = torch.ops.aten.sqrt.default(add_530);  add_530 = None
        reciprocal_237: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_237);  sqrt_237 = None
        mul_767: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_237, 1);  reciprocal_237 = None
        unsqueeze_1896: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg561_1, -1);  arg561_1 = None
        unsqueeze_1897: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1896, -1);  unsqueeze_1896 = None
        unsqueeze_1898: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_767, -1);  mul_767 = None
        unsqueeze_1899: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1898, -1);  unsqueeze_1898 = None
        sub_293: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_293, unsqueeze_1897);  convolution_293 = unsqueeze_1897 = None
        mul_768: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_1899);  sub_293 = unsqueeze_1899 = None
        unsqueeze_1900: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg563_1, -1);  arg563_1 = None
        unsqueeze_1901: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1900, -1);  unsqueeze_1900 = None
        mul_769: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_768, unsqueeze_1901);  mul_768 = unsqueeze_1901 = None
        unsqueeze_1902: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg564_1, -1);  arg564_1 = None
        unsqueeze_1903: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1902, -1);  unsqueeze_1902 = None
        add_531: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_769, unsqueeze_1903);  mul_769 = unsqueeze_1903 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_230: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_531);  add_531 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_294: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_230, arg565_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_230 = arg565_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_532: "f32[512]" = torch.ops.aten.add.Tensor(arg567_1, 1e-05);  arg567_1 = None
        sqrt_238: "f32[512]" = torch.ops.aten.sqrt.default(add_532);  add_532 = None
        reciprocal_238: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_238);  sqrt_238 = None
        mul_770: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_238, 1);  reciprocal_238 = None
        unsqueeze_1904: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg566_1, -1);  arg566_1 = None
        unsqueeze_1905: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1904, -1);  unsqueeze_1904 = None
        unsqueeze_1906: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_770, -1);  mul_770 = None
        unsqueeze_1907: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1906, -1);  unsqueeze_1906 = None
        sub_294: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_294, unsqueeze_1905);  convolution_294 = unsqueeze_1905 = None
        mul_771: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_1907);  sub_294 = unsqueeze_1907 = None
        unsqueeze_1908: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg568_1, -1);  arg568_1 = None
        unsqueeze_1909: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1908, -1);  unsqueeze_1908 = None
        mul_772: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_771, unsqueeze_1909);  mul_771 = unsqueeze_1909 = None
        unsqueeze_1910: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg569_1, -1);  arg569_1 = None
        unsqueeze_1911: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1910, -1);  unsqueeze_1910 = None
        add_533: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_772, unsqueeze_1911);  mul_772 = unsqueeze_1911 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_231: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_533);  add_533 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_338: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_231, [8, 2, 256, 16, 16]);  relu_231 = None
        sum_169: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_338, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_57: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_169, [2, 3], True);  sum_169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_295: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_57, arg570_1, arg571_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_57 = arg570_1 = arg571_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_534: "f32[128]" = torch.ops.aten.add.Tensor(arg573_1, 1e-05);  arg573_1 = None
        sqrt_239: "f32[128]" = torch.ops.aten.sqrt.default(add_534);  add_534 = None
        reciprocal_239: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_239);  sqrt_239 = None
        mul_773: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_239, 1);  reciprocal_239 = None
        unsqueeze_1912: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg572_1, -1);  arg572_1 = None
        unsqueeze_1913: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1912, -1);  unsqueeze_1912 = None
        unsqueeze_1914: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_773, -1);  mul_773 = None
        unsqueeze_1915: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1914, -1);  unsqueeze_1914 = None
        sub_295: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_295, unsqueeze_1913);  convolution_295 = unsqueeze_1913 = None
        mul_774: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_1915);  sub_295 = unsqueeze_1915 = None
        unsqueeze_1916: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg574_1, -1);  arg574_1 = None
        unsqueeze_1917: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1916, -1);  unsqueeze_1916 = None
        mul_775: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_774, unsqueeze_1917);  mul_774 = unsqueeze_1917 = None
        unsqueeze_1918: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg575_1, -1);  arg575_1 = None
        unsqueeze_1919: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1918, -1);  unsqueeze_1918 = None
        add_535: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_775, unsqueeze_1919);  mul_775 = unsqueeze_1919 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_232: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_535);  add_535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_296: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_232, arg576_1, arg577_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_232 = arg576_1 = arg577_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_339: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_296, [8, 1, 2, -1]);  convolution_296 = None
        permute_57: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_56: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_57, [1], True)
        sub_296: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_57, amax_56);  permute_57 = amax_56 = None
        exp_56: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_296);  sub_296 = None
        sum_170: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_56, [1], True)
        div_56: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_56, sum_170);  exp_56 = sum_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_340: "f32[8, 512]" = torch.ops.aten.view.default(div_56, [8, -1]);  div_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_341: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_340, [8, -1, 1, 1]);  view_340 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_342: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_341, [8, 2, 256, 1, 1]);  view_341 = None
        mul_776: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_338, view_342);  view_338 = view_342 = None
        sum_171: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_776, [1]);  mul_776 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_297: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_171, arg578_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_171 = arg578_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_536: "f32[1024]" = torch.ops.aten.add.Tensor(arg580_1, 1e-05);  arg580_1 = None
        sqrt_240: "f32[1024]" = torch.ops.aten.sqrt.default(add_536);  add_536 = None
        reciprocal_240: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_240);  sqrt_240 = None
        mul_777: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_240, 1);  reciprocal_240 = None
        unsqueeze_1920: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg579_1, -1);  arg579_1 = None
        unsqueeze_1921: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1920, -1);  unsqueeze_1920 = None
        unsqueeze_1922: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_777, -1);  mul_777 = None
        unsqueeze_1923: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1922, -1);  unsqueeze_1922 = None
        sub_297: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_297, unsqueeze_1921);  convolution_297 = unsqueeze_1921 = None
        mul_778: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_1923);  sub_297 = unsqueeze_1923 = None
        unsqueeze_1924: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg581_1, -1);  arg581_1 = None
        unsqueeze_1925: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1924, -1);  unsqueeze_1924 = None
        mul_779: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_778, unsqueeze_1925);  mul_778 = unsqueeze_1925 = None
        unsqueeze_1926: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg582_1, -1);  arg582_1 = None
        unsqueeze_1927: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1926, -1);  unsqueeze_1926 = None
        add_537: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_779, unsqueeze_1927);  mul_779 = unsqueeze_1927 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_538: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_537, relu_229);  add_537 = relu_229 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_233: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_538);  add_538 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_298: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_233, arg583_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg583_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_539: "f32[256]" = torch.ops.aten.add.Tensor(arg585_1, 1e-05);  arg585_1 = None
        sqrt_241: "f32[256]" = torch.ops.aten.sqrt.default(add_539);  add_539 = None
        reciprocal_241: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_241);  sqrt_241 = None
        mul_780: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_241, 1);  reciprocal_241 = None
        unsqueeze_1928: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg584_1, -1);  arg584_1 = None
        unsqueeze_1929: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1928, -1);  unsqueeze_1928 = None
        unsqueeze_1930: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_780, -1);  mul_780 = None
        unsqueeze_1931: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1930, -1);  unsqueeze_1930 = None
        sub_298: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_298, unsqueeze_1929);  convolution_298 = unsqueeze_1929 = None
        mul_781: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_1931);  sub_298 = unsqueeze_1931 = None
        unsqueeze_1932: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg586_1, -1);  arg586_1 = None
        unsqueeze_1933: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1932, -1);  unsqueeze_1932 = None
        mul_782: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_781, unsqueeze_1933);  mul_781 = unsqueeze_1933 = None
        unsqueeze_1934: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg587_1, -1);  arg587_1 = None
        unsqueeze_1935: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1934, -1);  unsqueeze_1934 = None
        add_540: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_782, unsqueeze_1935);  mul_782 = unsqueeze_1935 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_234: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_540);  add_540 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_299: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_234, arg588_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_234 = arg588_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_541: "f32[512]" = torch.ops.aten.add.Tensor(arg590_1, 1e-05);  arg590_1 = None
        sqrt_242: "f32[512]" = torch.ops.aten.sqrt.default(add_541);  add_541 = None
        reciprocal_242: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_242);  sqrt_242 = None
        mul_783: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_242, 1);  reciprocal_242 = None
        unsqueeze_1936: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg589_1, -1);  arg589_1 = None
        unsqueeze_1937: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1936, -1);  unsqueeze_1936 = None
        unsqueeze_1938: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_783, -1);  mul_783 = None
        unsqueeze_1939: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1938, -1);  unsqueeze_1938 = None
        sub_299: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_299, unsqueeze_1937);  convolution_299 = unsqueeze_1937 = None
        mul_784: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_1939);  sub_299 = unsqueeze_1939 = None
        unsqueeze_1940: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg591_1, -1);  arg591_1 = None
        unsqueeze_1941: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1940, -1);  unsqueeze_1940 = None
        mul_785: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_784, unsqueeze_1941);  mul_784 = unsqueeze_1941 = None
        unsqueeze_1942: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg592_1, -1);  arg592_1 = None
        unsqueeze_1943: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1942, -1);  unsqueeze_1942 = None
        add_542: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_785, unsqueeze_1943);  mul_785 = unsqueeze_1943 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_235: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_542);  add_542 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_344: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_235, [8, 2, 256, 16, 16]);  relu_235 = None
        sum_172: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_344, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_58: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_172, [2, 3], True);  sum_172 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_300: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_58, arg593_1, arg594_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_58 = arg593_1 = arg594_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_543: "f32[128]" = torch.ops.aten.add.Tensor(arg596_1, 1e-05);  arg596_1 = None
        sqrt_243: "f32[128]" = torch.ops.aten.sqrt.default(add_543);  add_543 = None
        reciprocal_243: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_243);  sqrt_243 = None
        mul_786: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_243, 1);  reciprocal_243 = None
        unsqueeze_1944: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg595_1, -1);  arg595_1 = None
        unsqueeze_1945: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1944, -1);  unsqueeze_1944 = None
        unsqueeze_1946: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_786, -1);  mul_786 = None
        unsqueeze_1947: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1946, -1);  unsqueeze_1946 = None
        sub_300: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_300, unsqueeze_1945);  convolution_300 = unsqueeze_1945 = None
        mul_787: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_1947);  sub_300 = unsqueeze_1947 = None
        unsqueeze_1948: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg597_1, -1);  arg597_1 = None
        unsqueeze_1949: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1948, -1);  unsqueeze_1948 = None
        mul_788: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_787, unsqueeze_1949);  mul_787 = unsqueeze_1949 = None
        unsqueeze_1950: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg598_1, -1);  arg598_1 = None
        unsqueeze_1951: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1950, -1);  unsqueeze_1950 = None
        add_544: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_788, unsqueeze_1951);  mul_788 = unsqueeze_1951 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_236: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_544);  add_544 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_301: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_236, arg599_1, arg600_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_236 = arg599_1 = arg600_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_345: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_301, [8, 1, 2, -1]);  convolution_301 = None
        permute_58: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_57: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_58, [1], True)
        sub_301: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_58, amax_57);  permute_58 = amax_57 = None
        exp_57: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_301);  sub_301 = None
        sum_173: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_57, [1], True)
        div_57: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_57, sum_173);  exp_57 = sum_173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_346: "f32[8, 512]" = torch.ops.aten.view.default(div_57, [8, -1]);  div_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_347: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_346, [8, -1, 1, 1]);  view_346 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_348: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_347, [8, 2, 256, 1, 1]);  view_347 = None
        mul_789: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_344, view_348);  view_344 = view_348 = None
        sum_174: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_789, [1]);  mul_789 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_302: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_174, arg601_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_174 = arg601_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_545: "f32[1024]" = torch.ops.aten.add.Tensor(arg603_1, 1e-05);  arg603_1 = None
        sqrt_244: "f32[1024]" = torch.ops.aten.sqrt.default(add_545);  add_545 = None
        reciprocal_244: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_244);  sqrt_244 = None
        mul_790: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_244, 1);  reciprocal_244 = None
        unsqueeze_1952: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg602_1, -1);  arg602_1 = None
        unsqueeze_1953: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1952, -1);  unsqueeze_1952 = None
        unsqueeze_1954: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_790, -1);  mul_790 = None
        unsqueeze_1955: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1954, -1);  unsqueeze_1954 = None
        sub_302: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_302, unsqueeze_1953);  convolution_302 = unsqueeze_1953 = None
        mul_791: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_1955);  sub_302 = unsqueeze_1955 = None
        unsqueeze_1956: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg604_1, -1);  arg604_1 = None
        unsqueeze_1957: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1956, -1);  unsqueeze_1956 = None
        mul_792: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_791, unsqueeze_1957);  mul_791 = unsqueeze_1957 = None
        unsqueeze_1958: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg605_1, -1);  arg605_1 = None
        unsqueeze_1959: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1958, -1);  unsqueeze_1958 = None
        add_546: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_792, unsqueeze_1959);  mul_792 = unsqueeze_1959 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_547: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_546, relu_233);  add_546 = relu_233 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_237: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_547);  add_547 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_303: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_237, arg606_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg606_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_548: "f32[256]" = torch.ops.aten.add.Tensor(arg608_1, 1e-05);  arg608_1 = None
        sqrt_245: "f32[256]" = torch.ops.aten.sqrt.default(add_548);  add_548 = None
        reciprocal_245: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_245);  sqrt_245 = None
        mul_793: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_245, 1);  reciprocal_245 = None
        unsqueeze_1960: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg607_1, -1);  arg607_1 = None
        unsqueeze_1961: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1960, -1);  unsqueeze_1960 = None
        unsqueeze_1962: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_793, -1);  mul_793 = None
        unsqueeze_1963: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1962, -1);  unsqueeze_1962 = None
        sub_303: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_303, unsqueeze_1961);  convolution_303 = unsqueeze_1961 = None
        mul_794: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_1963);  sub_303 = unsqueeze_1963 = None
        unsqueeze_1964: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg609_1, -1);  arg609_1 = None
        unsqueeze_1965: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1964, -1);  unsqueeze_1964 = None
        mul_795: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_794, unsqueeze_1965);  mul_794 = unsqueeze_1965 = None
        unsqueeze_1966: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg610_1, -1);  arg610_1 = None
        unsqueeze_1967: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1966, -1);  unsqueeze_1966 = None
        add_549: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_795, unsqueeze_1967);  mul_795 = unsqueeze_1967 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_238: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_549);  add_549 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_304: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_238, arg611_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_238 = arg611_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_550: "f32[512]" = torch.ops.aten.add.Tensor(arg613_1, 1e-05);  arg613_1 = None
        sqrt_246: "f32[512]" = torch.ops.aten.sqrt.default(add_550);  add_550 = None
        reciprocal_246: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_246);  sqrt_246 = None
        mul_796: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_246, 1);  reciprocal_246 = None
        unsqueeze_1968: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg612_1, -1);  arg612_1 = None
        unsqueeze_1969: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1968, -1);  unsqueeze_1968 = None
        unsqueeze_1970: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_796, -1);  mul_796 = None
        unsqueeze_1971: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1970, -1);  unsqueeze_1970 = None
        sub_304: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_304, unsqueeze_1969);  convolution_304 = unsqueeze_1969 = None
        mul_797: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_1971);  sub_304 = unsqueeze_1971 = None
        unsqueeze_1972: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg614_1, -1);  arg614_1 = None
        unsqueeze_1973: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1972, -1);  unsqueeze_1972 = None
        mul_798: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_797, unsqueeze_1973);  mul_797 = unsqueeze_1973 = None
        unsqueeze_1974: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg615_1, -1);  arg615_1 = None
        unsqueeze_1975: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1974, -1);  unsqueeze_1974 = None
        add_551: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_798, unsqueeze_1975);  mul_798 = unsqueeze_1975 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_239: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_551);  add_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_350: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_239, [8, 2, 256, 16, 16]);  relu_239 = None
        sum_175: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_350, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_59: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_175, [2, 3], True);  sum_175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_305: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_59, arg616_1, arg617_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_59 = arg616_1 = arg617_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_552: "f32[128]" = torch.ops.aten.add.Tensor(arg619_1, 1e-05);  arg619_1 = None
        sqrt_247: "f32[128]" = torch.ops.aten.sqrt.default(add_552);  add_552 = None
        reciprocal_247: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_247);  sqrt_247 = None
        mul_799: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_247, 1);  reciprocal_247 = None
        unsqueeze_1976: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg618_1, -1);  arg618_1 = None
        unsqueeze_1977: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1976, -1);  unsqueeze_1976 = None
        unsqueeze_1978: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_799, -1);  mul_799 = None
        unsqueeze_1979: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1978, -1);  unsqueeze_1978 = None
        sub_305: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_305, unsqueeze_1977);  convolution_305 = unsqueeze_1977 = None
        mul_800: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_1979);  sub_305 = unsqueeze_1979 = None
        unsqueeze_1980: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg620_1, -1);  arg620_1 = None
        unsqueeze_1981: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1980, -1);  unsqueeze_1980 = None
        mul_801: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_800, unsqueeze_1981);  mul_800 = unsqueeze_1981 = None
        unsqueeze_1982: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg621_1, -1);  arg621_1 = None
        unsqueeze_1983: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1982, -1);  unsqueeze_1982 = None
        add_553: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_801, unsqueeze_1983);  mul_801 = unsqueeze_1983 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_240: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_553);  add_553 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_306: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_240, arg622_1, arg623_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_240 = arg622_1 = arg623_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_351: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_306, [8, 1, 2, -1]);  convolution_306 = None
        permute_59: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_58: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_59, [1], True)
        sub_306: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_59, amax_58);  permute_59 = amax_58 = None
        exp_58: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_306);  sub_306 = None
        sum_176: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_58, [1], True)
        div_58: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_58, sum_176);  exp_58 = sum_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_352: "f32[8, 512]" = torch.ops.aten.view.default(div_58, [8, -1]);  div_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_353: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_352, [8, -1, 1, 1]);  view_352 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_354: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_353, [8, 2, 256, 1, 1]);  view_353 = None
        mul_802: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_350, view_354);  view_350 = view_354 = None
        sum_177: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_802, [1]);  mul_802 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_307: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_177, arg624_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_177 = arg624_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_554: "f32[1024]" = torch.ops.aten.add.Tensor(arg626_1, 1e-05);  arg626_1 = None
        sqrt_248: "f32[1024]" = torch.ops.aten.sqrt.default(add_554);  add_554 = None
        reciprocal_248: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_248);  sqrt_248 = None
        mul_803: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_248, 1);  reciprocal_248 = None
        unsqueeze_1984: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg625_1, -1);  arg625_1 = None
        unsqueeze_1985: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1984, -1);  unsqueeze_1984 = None
        unsqueeze_1986: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_803, -1);  mul_803 = None
        unsqueeze_1987: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1986, -1);  unsqueeze_1986 = None
        sub_307: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_307, unsqueeze_1985);  convolution_307 = unsqueeze_1985 = None
        mul_804: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_1987);  sub_307 = unsqueeze_1987 = None
        unsqueeze_1988: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg627_1, -1);  arg627_1 = None
        unsqueeze_1989: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1988, -1);  unsqueeze_1988 = None
        mul_805: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_804, unsqueeze_1989);  mul_804 = unsqueeze_1989 = None
        unsqueeze_1990: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg628_1, -1);  arg628_1 = None
        unsqueeze_1991: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1990, -1);  unsqueeze_1990 = None
        add_555: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_805, unsqueeze_1991);  mul_805 = unsqueeze_1991 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_556: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_555, relu_237);  add_555 = relu_237 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_241: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_556);  add_556 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_308: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_241, arg629_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg629_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_557: "f32[256]" = torch.ops.aten.add.Tensor(arg631_1, 1e-05);  arg631_1 = None
        sqrt_249: "f32[256]" = torch.ops.aten.sqrt.default(add_557);  add_557 = None
        reciprocal_249: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_249);  sqrt_249 = None
        mul_806: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_249, 1);  reciprocal_249 = None
        unsqueeze_1992: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg630_1, -1);  arg630_1 = None
        unsqueeze_1993: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1992, -1);  unsqueeze_1992 = None
        unsqueeze_1994: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_806, -1);  mul_806 = None
        unsqueeze_1995: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1994, -1);  unsqueeze_1994 = None
        sub_308: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_308, unsqueeze_1993);  convolution_308 = unsqueeze_1993 = None
        mul_807: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1995);  sub_308 = unsqueeze_1995 = None
        unsqueeze_1996: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg632_1, -1);  arg632_1 = None
        unsqueeze_1997: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1996, -1);  unsqueeze_1996 = None
        mul_808: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_807, unsqueeze_1997);  mul_807 = unsqueeze_1997 = None
        unsqueeze_1998: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg633_1, -1);  arg633_1 = None
        unsqueeze_1999: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1998, -1);  unsqueeze_1998 = None
        add_558: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_808, unsqueeze_1999);  mul_808 = unsqueeze_1999 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_242: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_558);  add_558 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_309: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_242, arg634_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_242 = arg634_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_559: "f32[512]" = torch.ops.aten.add.Tensor(arg636_1, 1e-05);  arg636_1 = None
        sqrt_250: "f32[512]" = torch.ops.aten.sqrt.default(add_559);  add_559 = None
        reciprocal_250: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_250);  sqrt_250 = None
        mul_809: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_250, 1);  reciprocal_250 = None
        unsqueeze_2000: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg635_1, -1);  arg635_1 = None
        unsqueeze_2001: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2000, -1);  unsqueeze_2000 = None
        unsqueeze_2002: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_809, -1);  mul_809 = None
        unsqueeze_2003: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2002, -1);  unsqueeze_2002 = None
        sub_309: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_309, unsqueeze_2001);  convolution_309 = unsqueeze_2001 = None
        mul_810: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_2003);  sub_309 = unsqueeze_2003 = None
        unsqueeze_2004: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg637_1, -1);  arg637_1 = None
        unsqueeze_2005: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2004, -1);  unsqueeze_2004 = None
        mul_811: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_810, unsqueeze_2005);  mul_810 = unsqueeze_2005 = None
        unsqueeze_2006: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg638_1, -1);  arg638_1 = None
        unsqueeze_2007: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2006, -1);  unsqueeze_2006 = None
        add_560: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_811, unsqueeze_2007);  mul_811 = unsqueeze_2007 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_243: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_560);  add_560 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_356: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_243, [8, 2, 256, 16, 16]);  relu_243 = None
        sum_178: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_356, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_60: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_178, [2, 3], True);  sum_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_310: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_60, arg639_1, arg640_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_60 = arg639_1 = arg640_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_561: "f32[128]" = torch.ops.aten.add.Tensor(arg642_1, 1e-05);  arg642_1 = None
        sqrt_251: "f32[128]" = torch.ops.aten.sqrt.default(add_561);  add_561 = None
        reciprocal_251: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_251);  sqrt_251 = None
        mul_812: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_251, 1);  reciprocal_251 = None
        unsqueeze_2008: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg641_1, -1);  arg641_1 = None
        unsqueeze_2009: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2008, -1);  unsqueeze_2008 = None
        unsqueeze_2010: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_812, -1);  mul_812 = None
        unsqueeze_2011: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2010, -1);  unsqueeze_2010 = None
        sub_310: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_310, unsqueeze_2009);  convolution_310 = unsqueeze_2009 = None
        mul_813: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_2011);  sub_310 = unsqueeze_2011 = None
        unsqueeze_2012: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg643_1, -1);  arg643_1 = None
        unsqueeze_2013: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2012, -1);  unsqueeze_2012 = None
        mul_814: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_813, unsqueeze_2013);  mul_813 = unsqueeze_2013 = None
        unsqueeze_2014: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg644_1, -1);  arg644_1 = None
        unsqueeze_2015: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2014, -1);  unsqueeze_2014 = None
        add_562: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_814, unsqueeze_2015);  mul_814 = unsqueeze_2015 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_244: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_562);  add_562 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_311: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_244, arg645_1, arg646_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_244 = arg645_1 = arg646_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_357: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_311, [8, 1, 2, -1]);  convolution_311 = None
        permute_60: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_59: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_60, [1], True)
        sub_311: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_60, amax_59);  permute_60 = amax_59 = None
        exp_59: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_311);  sub_311 = None
        sum_179: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_59, [1], True)
        div_59: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_59, sum_179);  exp_59 = sum_179 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_358: "f32[8, 512]" = torch.ops.aten.view.default(div_59, [8, -1]);  div_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_359: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_358, [8, -1, 1, 1]);  view_358 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_360: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_359, [8, 2, 256, 1, 1]);  view_359 = None
        mul_815: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_356, view_360);  view_356 = view_360 = None
        sum_180: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_815, [1]);  mul_815 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_312: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_180, arg647_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_180 = arg647_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_563: "f32[1024]" = torch.ops.aten.add.Tensor(arg649_1, 1e-05);  arg649_1 = None
        sqrt_252: "f32[1024]" = torch.ops.aten.sqrt.default(add_563);  add_563 = None
        reciprocal_252: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_252);  sqrt_252 = None
        mul_816: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_252, 1);  reciprocal_252 = None
        unsqueeze_2016: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg648_1, -1);  arg648_1 = None
        unsqueeze_2017: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2016, -1);  unsqueeze_2016 = None
        unsqueeze_2018: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_816, -1);  mul_816 = None
        unsqueeze_2019: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2018, -1);  unsqueeze_2018 = None
        sub_312: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_312, unsqueeze_2017);  convolution_312 = unsqueeze_2017 = None
        mul_817: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_2019);  sub_312 = unsqueeze_2019 = None
        unsqueeze_2020: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg650_1, -1);  arg650_1 = None
        unsqueeze_2021: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2020, -1);  unsqueeze_2020 = None
        mul_818: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_817, unsqueeze_2021);  mul_817 = unsqueeze_2021 = None
        unsqueeze_2022: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg651_1, -1);  arg651_1 = None
        unsqueeze_2023: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2022, -1);  unsqueeze_2022 = None
        add_564: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_818, unsqueeze_2023);  mul_818 = unsqueeze_2023 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_565: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_564, relu_241);  add_564 = relu_241 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_245: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_565);  add_565 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_313: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_245, arg652_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg652_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_566: "f32[256]" = torch.ops.aten.add.Tensor(arg654_1, 1e-05);  arg654_1 = None
        sqrt_253: "f32[256]" = torch.ops.aten.sqrt.default(add_566);  add_566 = None
        reciprocal_253: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_253);  sqrt_253 = None
        mul_819: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_253, 1);  reciprocal_253 = None
        unsqueeze_2024: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg653_1, -1);  arg653_1 = None
        unsqueeze_2025: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2024, -1);  unsqueeze_2024 = None
        unsqueeze_2026: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_819, -1);  mul_819 = None
        unsqueeze_2027: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2026, -1);  unsqueeze_2026 = None
        sub_313: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_313, unsqueeze_2025);  convolution_313 = unsqueeze_2025 = None
        mul_820: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_2027);  sub_313 = unsqueeze_2027 = None
        unsqueeze_2028: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg655_1, -1);  arg655_1 = None
        unsqueeze_2029: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2028, -1);  unsqueeze_2028 = None
        mul_821: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_820, unsqueeze_2029);  mul_820 = unsqueeze_2029 = None
        unsqueeze_2030: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg656_1, -1);  arg656_1 = None
        unsqueeze_2031: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2030, -1);  unsqueeze_2030 = None
        add_567: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_821, unsqueeze_2031);  mul_821 = unsqueeze_2031 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_246: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_567);  add_567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_314: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_246, arg657_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_246 = arg657_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_568: "f32[512]" = torch.ops.aten.add.Tensor(arg659_1, 1e-05);  arg659_1 = None
        sqrt_254: "f32[512]" = torch.ops.aten.sqrt.default(add_568);  add_568 = None
        reciprocal_254: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_254);  sqrt_254 = None
        mul_822: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_254, 1);  reciprocal_254 = None
        unsqueeze_2032: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg658_1, -1);  arg658_1 = None
        unsqueeze_2033: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2032, -1);  unsqueeze_2032 = None
        unsqueeze_2034: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_822, -1);  mul_822 = None
        unsqueeze_2035: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2034, -1);  unsqueeze_2034 = None
        sub_314: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_314, unsqueeze_2033);  convolution_314 = unsqueeze_2033 = None
        mul_823: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_2035);  sub_314 = unsqueeze_2035 = None
        unsqueeze_2036: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg660_1, -1);  arg660_1 = None
        unsqueeze_2037: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2036, -1);  unsqueeze_2036 = None
        mul_824: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_823, unsqueeze_2037);  mul_823 = unsqueeze_2037 = None
        unsqueeze_2038: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg661_1, -1);  arg661_1 = None
        unsqueeze_2039: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2038, -1);  unsqueeze_2038 = None
        add_569: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_824, unsqueeze_2039);  mul_824 = unsqueeze_2039 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_247: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_569);  add_569 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_362: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_247, [8, 2, 256, 16, 16]);  relu_247 = None
        sum_181: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_362, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_61: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_181, [2, 3], True);  sum_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_315: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_61, arg662_1, arg663_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_61 = arg662_1 = arg663_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_570: "f32[128]" = torch.ops.aten.add.Tensor(arg665_1, 1e-05);  arg665_1 = None
        sqrt_255: "f32[128]" = torch.ops.aten.sqrt.default(add_570);  add_570 = None
        reciprocal_255: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_255);  sqrt_255 = None
        mul_825: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_255, 1);  reciprocal_255 = None
        unsqueeze_2040: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg664_1, -1);  arg664_1 = None
        unsqueeze_2041: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2040, -1);  unsqueeze_2040 = None
        unsqueeze_2042: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_825, -1);  mul_825 = None
        unsqueeze_2043: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2042, -1);  unsqueeze_2042 = None
        sub_315: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_315, unsqueeze_2041);  convolution_315 = unsqueeze_2041 = None
        mul_826: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_2043);  sub_315 = unsqueeze_2043 = None
        unsqueeze_2044: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg666_1, -1);  arg666_1 = None
        unsqueeze_2045: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2044, -1);  unsqueeze_2044 = None
        mul_827: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_826, unsqueeze_2045);  mul_826 = unsqueeze_2045 = None
        unsqueeze_2046: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg667_1, -1);  arg667_1 = None
        unsqueeze_2047: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2046, -1);  unsqueeze_2046 = None
        add_571: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_827, unsqueeze_2047);  mul_827 = unsqueeze_2047 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_248: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_571);  add_571 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_316: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_248, arg668_1, arg669_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_248 = arg668_1 = arg669_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_363: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_316, [8, 1, 2, -1]);  convolution_316 = None
        permute_61: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_363, [0, 2, 1, 3]);  view_363 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_60: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_61, [1], True)
        sub_316: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_61, amax_60);  permute_61 = amax_60 = None
        exp_60: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_316);  sub_316 = None
        sum_182: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_60, [1], True)
        div_60: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_60, sum_182);  exp_60 = sum_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_364: "f32[8, 512]" = torch.ops.aten.view.default(div_60, [8, -1]);  div_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_365: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_364, [8, -1, 1, 1]);  view_364 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_366: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_365, [8, 2, 256, 1, 1]);  view_365 = None
        mul_828: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_362, view_366);  view_362 = view_366 = None
        sum_183: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_828, [1]);  mul_828 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_317: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_183, arg670_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_183 = arg670_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_572: "f32[1024]" = torch.ops.aten.add.Tensor(arg672_1, 1e-05);  arg672_1 = None
        sqrt_256: "f32[1024]" = torch.ops.aten.sqrt.default(add_572);  add_572 = None
        reciprocal_256: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_256);  sqrt_256 = None
        mul_829: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_256, 1);  reciprocal_256 = None
        unsqueeze_2048: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg671_1, -1);  arg671_1 = None
        unsqueeze_2049: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2048, -1);  unsqueeze_2048 = None
        unsqueeze_2050: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_829, -1);  mul_829 = None
        unsqueeze_2051: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2050, -1);  unsqueeze_2050 = None
        sub_317: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_317, unsqueeze_2049);  convolution_317 = unsqueeze_2049 = None
        mul_830: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_2051);  sub_317 = unsqueeze_2051 = None
        unsqueeze_2052: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg673_1, -1);  arg673_1 = None
        unsqueeze_2053: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2052, -1);  unsqueeze_2052 = None
        mul_831: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_830, unsqueeze_2053);  mul_830 = unsqueeze_2053 = None
        unsqueeze_2054: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg674_1, -1);  arg674_1 = None
        unsqueeze_2055: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2054, -1);  unsqueeze_2054 = None
        add_573: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_831, unsqueeze_2055);  mul_831 = unsqueeze_2055 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_574: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_573, relu_245);  add_573 = relu_245 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_249: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_574);  add_574 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_318: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_249, arg675_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg675_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_575: "f32[256]" = torch.ops.aten.add.Tensor(arg677_1, 1e-05);  arg677_1 = None
        sqrt_257: "f32[256]" = torch.ops.aten.sqrt.default(add_575);  add_575 = None
        reciprocal_257: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_257);  sqrt_257 = None
        mul_832: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_257, 1);  reciprocal_257 = None
        unsqueeze_2056: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg676_1, -1);  arg676_1 = None
        unsqueeze_2057: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2056, -1);  unsqueeze_2056 = None
        unsqueeze_2058: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_832, -1);  mul_832 = None
        unsqueeze_2059: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2058, -1);  unsqueeze_2058 = None
        sub_318: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_318, unsqueeze_2057);  convolution_318 = unsqueeze_2057 = None
        mul_833: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_2059);  sub_318 = unsqueeze_2059 = None
        unsqueeze_2060: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg678_1, -1);  arg678_1 = None
        unsqueeze_2061: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2060, -1);  unsqueeze_2060 = None
        mul_834: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_833, unsqueeze_2061);  mul_833 = unsqueeze_2061 = None
        unsqueeze_2062: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg679_1, -1);  arg679_1 = None
        unsqueeze_2063: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2062, -1);  unsqueeze_2062 = None
        add_576: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_834, unsqueeze_2063);  mul_834 = unsqueeze_2063 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_250: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_576);  add_576 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_319: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_250, arg680_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_250 = arg680_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_577: "f32[512]" = torch.ops.aten.add.Tensor(arg682_1, 1e-05);  arg682_1 = None
        sqrt_258: "f32[512]" = torch.ops.aten.sqrt.default(add_577);  add_577 = None
        reciprocal_258: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_258);  sqrt_258 = None
        mul_835: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_258, 1);  reciprocal_258 = None
        unsqueeze_2064: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg681_1, -1);  arg681_1 = None
        unsqueeze_2065: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2064, -1);  unsqueeze_2064 = None
        unsqueeze_2066: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_835, -1);  mul_835 = None
        unsqueeze_2067: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2066, -1);  unsqueeze_2066 = None
        sub_319: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_319, unsqueeze_2065);  convolution_319 = unsqueeze_2065 = None
        mul_836: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_2067);  sub_319 = unsqueeze_2067 = None
        unsqueeze_2068: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg683_1, -1);  arg683_1 = None
        unsqueeze_2069: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2068, -1);  unsqueeze_2068 = None
        mul_837: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_836, unsqueeze_2069);  mul_836 = unsqueeze_2069 = None
        unsqueeze_2070: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg684_1, -1);  arg684_1 = None
        unsqueeze_2071: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2070, -1);  unsqueeze_2070 = None
        add_578: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_837, unsqueeze_2071);  mul_837 = unsqueeze_2071 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_251: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_578);  add_578 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_368: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_251, [8, 2, 256, 16, 16]);  relu_251 = None
        sum_184: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_368, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_62: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_184, [2, 3], True);  sum_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_320: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_62, arg685_1, arg686_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_62 = arg685_1 = arg686_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_579: "f32[128]" = torch.ops.aten.add.Tensor(arg688_1, 1e-05);  arg688_1 = None
        sqrt_259: "f32[128]" = torch.ops.aten.sqrt.default(add_579);  add_579 = None
        reciprocal_259: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_259);  sqrt_259 = None
        mul_838: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_259, 1);  reciprocal_259 = None
        unsqueeze_2072: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg687_1, -1);  arg687_1 = None
        unsqueeze_2073: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2072, -1);  unsqueeze_2072 = None
        unsqueeze_2074: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_838, -1);  mul_838 = None
        unsqueeze_2075: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2074, -1);  unsqueeze_2074 = None
        sub_320: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_320, unsqueeze_2073);  convolution_320 = unsqueeze_2073 = None
        mul_839: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_2075);  sub_320 = unsqueeze_2075 = None
        unsqueeze_2076: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg689_1, -1);  arg689_1 = None
        unsqueeze_2077: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2076, -1);  unsqueeze_2076 = None
        mul_840: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_839, unsqueeze_2077);  mul_839 = unsqueeze_2077 = None
        unsqueeze_2078: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg690_1, -1);  arg690_1 = None
        unsqueeze_2079: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2078, -1);  unsqueeze_2078 = None
        add_580: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_840, unsqueeze_2079);  mul_840 = unsqueeze_2079 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_252: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_580);  add_580 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_321: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_252, arg691_1, arg692_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_252 = arg691_1 = arg692_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_369: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_321, [8, 1, 2, -1]);  convolution_321 = None
        permute_62: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_61: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_62, [1], True)
        sub_321: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_62, amax_61);  permute_62 = amax_61 = None
        exp_61: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_321);  sub_321 = None
        sum_185: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_61, [1], True)
        div_61: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_61, sum_185);  exp_61 = sum_185 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_370: "f32[8, 512]" = torch.ops.aten.view.default(div_61, [8, -1]);  div_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_371: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_370, [8, -1, 1, 1]);  view_370 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_372: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_371, [8, 2, 256, 1, 1]);  view_371 = None
        mul_841: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_368, view_372);  view_368 = view_372 = None
        sum_186: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_841, [1]);  mul_841 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_322: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_186, arg693_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_186 = arg693_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_581: "f32[1024]" = torch.ops.aten.add.Tensor(arg695_1, 1e-05);  arg695_1 = None
        sqrt_260: "f32[1024]" = torch.ops.aten.sqrt.default(add_581);  add_581 = None
        reciprocal_260: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_260);  sqrt_260 = None
        mul_842: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_260, 1);  reciprocal_260 = None
        unsqueeze_2080: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg694_1, -1);  arg694_1 = None
        unsqueeze_2081: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2080, -1);  unsqueeze_2080 = None
        unsqueeze_2082: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_842, -1);  mul_842 = None
        unsqueeze_2083: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2082, -1);  unsqueeze_2082 = None
        sub_322: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_322, unsqueeze_2081);  convolution_322 = unsqueeze_2081 = None
        mul_843: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_2083);  sub_322 = unsqueeze_2083 = None
        unsqueeze_2084: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg696_1, -1);  arg696_1 = None
        unsqueeze_2085: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2084, -1);  unsqueeze_2084 = None
        mul_844: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_843, unsqueeze_2085);  mul_843 = unsqueeze_2085 = None
        unsqueeze_2086: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg697_1, -1);  arg697_1 = None
        unsqueeze_2087: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2086, -1);  unsqueeze_2086 = None
        add_582: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_844, unsqueeze_2087);  mul_844 = unsqueeze_2087 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_583: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_582, relu_249);  add_582 = relu_249 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_253: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_583);  add_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_323: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_253, arg698_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg698_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_584: "f32[256]" = torch.ops.aten.add.Tensor(arg700_1, 1e-05);  arg700_1 = None
        sqrt_261: "f32[256]" = torch.ops.aten.sqrt.default(add_584);  add_584 = None
        reciprocal_261: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_261);  sqrt_261 = None
        mul_845: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_261, 1);  reciprocal_261 = None
        unsqueeze_2088: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg699_1, -1);  arg699_1 = None
        unsqueeze_2089: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2088, -1);  unsqueeze_2088 = None
        unsqueeze_2090: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_845, -1);  mul_845 = None
        unsqueeze_2091: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2090, -1);  unsqueeze_2090 = None
        sub_323: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_323, unsqueeze_2089);  convolution_323 = unsqueeze_2089 = None
        mul_846: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_2091);  sub_323 = unsqueeze_2091 = None
        unsqueeze_2092: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg701_1, -1);  arg701_1 = None
        unsqueeze_2093: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2092, -1);  unsqueeze_2092 = None
        mul_847: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_846, unsqueeze_2093);  mul_846 = unsqueeze_2093 = None
        unsqueeze_2094: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg702_1, -1);  arg702_1 = None
        unsqueeze_2095: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2094, -1);  unsqueeze_2094 = None
        add_585: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_847, unsqueeze_2095);  mul_847 = unsqueeze_2095 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_254: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_585);  add_585 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_324: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_254, arg703_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_254 = arg703_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_586: "f32[512]" = torch.ops.aten.add.Tensor(arg705_1, 1e-05);  arg705_1 = None
        sqrt_262: "f32[512]" = torch.ops.aten.sqrt.default(add_586);  add_586 = None
        reciprocal_262: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_262);  sqrt_262 = None
        mul_848: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_262, 1);  reciprocal_262 = None
        unsqueeze_2096: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg704_1, -1);  arg704_1 = None
        unsqueeze_2097: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2096, -1);  unsqueeze_2096 = None
        unsqueeze_2098: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_848, -1);  mul_848 = None
        unsqueeze_2099: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2098, -1);  unsqueeze_2098 = None
        sub_324: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_324, unsqueeze_2097);  convolution_324 = unsqueeze_2097 = None
        mul_849: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_2099);  sub_324 = unsqueeze_2099 = None
        unsqueeze_2100: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg706_1, -1);  arg706_1 = None
        unsqueeze_2101: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2100, -1);  unsqueeze_2100 = None
        mul_850: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_849, unsqueeze_2101);  mul_849 = unsqueeze_2101 = None
        unsqueeze_2102: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg707_1, -1);  arg707_1 = None
        unsqueeze_2103: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2102, -1);  unsqueeze_2102 = None
        add_587: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_850, unsqueeze_2103);  mul_850 = unsqueeze_2103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_255: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_587);  add_587 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_374: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_255, [8, 2, 256, 16, 16]);  relu_255 = None
        sum_187: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_374, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_63: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_187, [2, 3], True);  sum_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_325: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_63, arg708_1, arg709_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_63 = arg708_1 = arg709_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_588: "f32[128]" = torch.ops.aten.add.Tensor(arg711_1, 1e-05);  arg711_1 = None
        sqrt_263: "f32[128]" = torch.ops.aten.sqrt.default(add_588);  add_588 = None
        reciprocal_263: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_263);  sqrt_263 = None
        mul_851: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_263, 1);  reciprocal_263 = None
        unsqueeze_2104: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg710_1, -1);  arg710_1 = None
        unsqueeze_2105: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2104, -1);  unsqueeze_2104 = None
        unsqueeze_2106: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_851, -1);  mul_851 = None
        unsqueeze_2107: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2106, -1);  unsqueeze_2106 = None
        sub_325: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_325, unsqueeze_2105);  convolution_325 = unsqueeze_2105 = None
        mul_852: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_2107);  sub_325 = unsqueeze_2107 = None
        unsqueeze_2108: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg712_1, -1);  arg712_1 = None
        unsqueeze_2109: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2108, -1);  unsqueeze_2108 = None
        mul_853: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_852, unsqueeze_2109);  mul_852 = unsqueeze_2109 = None
        unsqueeze_2110: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg713_1, -1);  arg713_1 = None
        unsqueeze_2111: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2110, -1);  unsqueeze_2110 = None
        add_589: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_853, unsqueeze_2111);  mul_853 = unsqueeze_2111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_256: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_589);  add_589 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_326: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_256, arg714_1, arg715_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_256 = arg714_1 = arg715_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_375: "f32[8, 1, 2, 256]" = torch.ops.aten.view.default(convolution_326, [8, 1, 2, -1]);  convolution_326 = None
        permute_63: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_62: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_63, [1], True)
        sub_326: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_63, amax_62);  permute_63 = amax_62 = None
        exp_62: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_326);  sub_326 = None
        sum_188: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_62, [1], True)
        div_62: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_62, sum_188);  exp_62 = sum_188 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_376: "f32[8, 512]" = torch.ops.aten.view.default(div_62, [8, -1]);  div_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_377: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_376, [8, -1, 1, 1]);  view_376 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_378: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_377, [8, 2, 256, 1, 1]);  view_377 = None
        mul_854: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_374, view_378);  view_374 = view_378 = None
        sum_189: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_854, [1]);  mul_854 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_327: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_189, arg716_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_189 = arg716_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_590: "f32[1024]" = torch.ops.aten.add.Tensor(arg718_1, 1e-05);  arg718_1 = None
        sqrt_264: "f32[1024]" = torch.ops.aten.sqrt.default(add_590);  add_590 = None
        reciprocal_264: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_264);  sqrt_264 = None
        mul_855: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_264, 1);  reciprocal_264 = None
        unsqueeze_2112: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg717_1, -1);  arg717_1 = None
        unsqueeze_2113: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2112, -1);  unsqueeze_2112 = None
        unsqueeze_2114: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_855, -1);  mul_855 = None
        unsqueeze_2115: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2114, -1);  unsqueeze_2114 = None
        sub_327: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_327, unsqueeze_2113);  convolution_327 = unsqueeze_2113 = None
        mul_856: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_2115);  sub_327 = unsqueeze_2115 = None
        unsqueeze_2116: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg719_1, -1);  arg719_1 = None
        unsqueeze_2117: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2116, -1);  unsqueeze_2116 = None
        mul_857: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_856, unsqueeze_2117);  mul_856 = unsqueeze_2117 = None
        unsqueeze_2118: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg720_1, -1);  arg720_1 = None
        unsqueeze_2119: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2118, -1);  unsqueeze_2118 = None
        add_591: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_857, unsqueeze_2119);  mul_857 = unsqueeze_2119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_592: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_591, relu_253);  add_591 = relu_253 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_257: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_592);  add_592 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_328: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_257, arg721_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg721_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_593: "f32[512]" = torch.ops.aten.add.Tensor(arg723_1, 1e-05);  arg723_1 = None
        sqrt_265: "f32[512]" = torch.ops.aten.sqrt.default(add_593);  add_593 = None
        reciprocal_265: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_265);  sqrt_265 = None
        mul_858: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_265, 1);  reciprocal_265 = None
        unsqueeze_2120: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg722_1, -1);  arg722_1 = None
        unsqueeze_2121: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2120, -1);  unsqueeze_2120 = None
        unsqueeze_2122: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_858, -1);  mul_858 = None
        unsqueeze_2123: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2122, -1);  unsqueeze_2122 = None
        sub_328: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_328, unsqueeze_2121);  convolution_328 = unsqueeze_2121 = None
        mul_859: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_2123);  sub_328 = unsqueeze_2123 = None
        unsqueeze_2124: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg724_1, -1);  arg724_1 = None
        unsqueeze_2125: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2124, -1);  unsqueeze_2124 = None
        mul_860: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_859, unsqueeze_2125);  mul_859 = unsqueeze_2125 = None
        unsqueeze_2126: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg725_1, -1);  arg725_1 = None
        unsqueeze_2127: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2126, -1);  unsqueeze_2126 = None
        add_594: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_860, unsqueeze_2127);  mul_860 = unsqueeze_2127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_258: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_594);  add_594 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_329: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(relu_258, arg726_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_258 = arg726_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_595: "f32[1024]" = torch.ops.aten.add.Tensor(arg728_1, 1e-05);  arg728_1 = None
        sqrt_266: "f32[1024]" = torch.ops.aten.sqrt.default(add_595);  add_595 = None
        reciprocal_266: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_266);  sqrt_266 = None
        mul_861: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_266, 1);  reciprocal_266 = None
        unsqueeze_2128: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg727_1, -1);  arg727_1 = None
        unsqueeze_2129: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2128, -1);  unsqueeze_2128 = None
        unsqueeze_2130: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_861, -1);  mul_861 = None
        unsqueeze_2131: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2130, -1);  unsqueeze_2130 = None
        sub_329: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_329, unsqueeze_2129);  convolution_329 = unsqueeze_2129 = None
        mul_862: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_2131);  sub_329 = unsqueeze_2131 = None
        unsqueeze_2132: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg729_1, -1);  arg729_1 = None
        unsqueeze_2133: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2132, -1);  unsqueeze_2132 = None
        mul_863: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_862, unsqueeze_2133);  mul_862 = unsqueeze_2133 = None
        unsqueeze_2134: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg730_1, -1);  arg730_1 = None
        unsqueeze_2135: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2134, -1);  unsqueeze_2134 = None
        add_596: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_863, unsqueeze_2135);  mul_863 = unsqueeze_2135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_259: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_596);  add_596 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_380: "f32[8, 2, 512, 16, 16]" = torch.ops.aten.view.default(relu_259, [8, 2, 512, 16, 16]);  relu_259 = None
        sum_190: "f32[8, 512, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_380, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_64: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(sum_190, [2, 3], True);  sum_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_330: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_64, arg731_1, arg732_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_64 = arg731_1 = arg732_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_597: "f32[256]" = torch.ops.aten.add.Tensor(arg734_1, 1e-05);  arg734_1 = None
        sqrt_267: "f32[256]" = torch.ops.aten.sqrt.default(add_597);  add_597 = None
        reciprocal_267: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_267);  sqrt_267 = None
        mul_864: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_267, 1);  reciprocal_267 = None
        unsqueeze_2136: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg733_1, -1);  arg733_1 = None
        unsqueeze_2137: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2136, -1);  unsqueeze_2136 = None
        unsqueeze_2138: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_864, -1);  mul_864 = None
        unsqueeze_2139: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2138, -1);  unsqueeze_2138 = None
        sub_330: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_330, unsqueeze_2137);  convolution_330 = unsqueeze_2137 = None
        mul_865: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_2139);  sub_330 = unsqueeze_2139 = None
        unsqueeze_2140: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg735_1, -1);  arg735_1 = None
        unsqueeze_2141: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2140, -1);  unsqueeze_2140 = None
        mul_866: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(mul_865, unsqueeze_2141);  mul_865 = unsqueeze_2141 = None
        unsqueeze_2142: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg736_1, -1);  arg736_1 = None
        unsqueeze_2143: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2142, -1);  unsqueeze_2142 = None
        add_598: "f32[8, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_866, unsqueeze_2143);  mul_866 = unsqueeze_2143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_260: "f32[8, 256, 1, 1]" = torch.ops.aten.relu.default(add_598);  add_598 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_331: "f32[8, 1024, 1, 1]" = torch.ops.aten.convolution.default(relu_260, arg737_1, arg738_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_260 = arg737_1 = arg738_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_381: "f32[8, 1, 2, 512]" = torch.ops.aten.view.default(convolution_331, [8, 1, 2, -1]);  convolution_331 = None
        permute_64: "f32[8, 2, 1, 512]" = torch.ops.aten.permute.default(view_381, [0, 2, 1, 3]);  view_381 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_63: "f32[8, 1, 1, 512]" = torch.ops.aten.amax.default(permute_64, [1], True)
        sub_331: "f32[8, 2, 1, 512]" = torch.ops.aten.sub.Tensor(permute_64, amax_63);  permute_64 = amax_63 = None
        exp_63: "f32[8, 2, 1, 512]" = torch.ops.aten.exp.default(sub_331);  sub_331 = None
        sum_191: "f32[8, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(exp_63, [1], True)
        div_63: "f32[8, 2, 1, 512]" = torch.ops.aten.div.Tensor(exp_63, sum_191);  exp_63 = sum_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_382: "f32[8, 1024]" = torch.ops.aten.view.default(div_63, [8, -1]);  div_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_383: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(view_382, [8, -1, 1, 1]);  view_382 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_384: "f32[8, 2, 512, 1, 1]" = torch.ops.aten.view.default(view_383, [8, 2, 512, 1, 1]);  view_383 = None
        mul_867: "f32[8, 2, 512, 16, 16]" = torch.ops.aten.mul.Tensor(view_380, view_384);  view_380 = view_384 = None
        sum_192: "f32[8, 512, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_867, [1]);  mul_867 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:107 in forward, code: out = self.avd_last(out)
        avg_pool2d_10: "f32[8, 512, 8, 8]" = torch.ops.aten.avg_pool2d.default(sum_192, [3, 3], [2, 2], [1, 1]);  sum_192 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_332: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(avg_pool2d_10, arg739_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_10 = arg739_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_599: "f32[2048]" = torch.ops.aten.add.Tensor(arg741_1, 1e-05);  arg741_1 = None
        sqrt_268: "f32[2048]" = torch.ops.aten.sqrt.default(add_599);  add_599 = None
        reciprocal_268: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_268);  sqrt_268 = None
        mul_868: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_268, 1);  reciprocal_268 = None
        unsqueeze_2144: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg740_1, -1);  arg740_1 = None
        unsqueeze_2145: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2144, -1);  unsqueeze_2144 = None
        unsqueeze_2146: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_868, -1);  mul_868 = None
        unsqueeze_2147: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2146, -1);  unsqueeze_2146 = None
        sub_332: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_332, unsqueeze_2145);  convolution_332 = unsqueeze_2145 = None
        mul_869: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_2147);  sub_332 = unsqueeze_2147 = None
        unsqueeze_2148: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg742_1, -1);  arg742_1 = None
        unsqueeze_2149: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2148, -1);  unsqueeze_2148 = None
        mul_870: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_869, unsqueeze_2149);  mul_869 = unsqueeze_2149 = None
        unsqueeze_2150: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg743_1, -1);  arg743_1 = None
        unsqueeze_2151: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2150, -1);  unsqueeze_2150 = None
        add_600: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_870, unsqueeze_2151);  mul_870 = unsqueeze_2151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:113 in forward, code: shortcut = self.downsample(x)
        avg_pool2d_11: "f32[8, 1024, 8, 8]" = torch.ops.aten.avg_pool2d.default(relu_257, [2, 2], [2, 2], [0, 0], True, False);  relu_257 = None
        convolution_333: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(avg_pool2d_11, arg744_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_11 = arg744_1 = None
        add_601: "f32[2048]" = torch.ops.aten.add.Tensor(arg746_1, 1e-05);  arg746_1 = None
        sqrt_269: "f32[2048]" = torch.ops.aten.sqrt.default(add_601);  add_601 = None
        reciprocal_269: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_269);  sqrt_269 = None
        mul_871: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_269, 1);  reciprocal_269 = None
        unsqueeze_2152: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg745_1, -1);  arg745_1 = None
        unsqueeze_2153: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2152, -1);  unsqueeze_2152 = None
        unsqueeze_2154: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_871, -1);  mul_871 = None
        unsqueeze_2155: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2154, -1);  unsqueeze_2154 = None
        sub_333: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_333, unsqueeze_2153);  convolution_333 = unsqueeze_2153 = None
        mul_872: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_2155);  sub_333 = unsqueeze_2155 = None
        unsqueeze_2156: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg747_1, -1);  arg747_1 = None
        unsqueeze_2157: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2156, -1);  unsqueeze_2156 = None
        mul_873: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_872, unsqueeze_2157);  mul_872 = unsqueeze_2157 = None
        unsqueeze_2158: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg748_1, -1);  arg748_1 = None
        unsqueeze_2159: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2158, -1);  unsqueeze_2158 = None
        add_602: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_873, unsqueeze_2159);  mul_873 = unsqueeze_2159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_603: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_600, add_602);  add_600 = add_602 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_261: "f32[8, 2048, 8, 8]" = torch.ops.aten.relu.default(add_603);  add_603 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_334: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(relu_261, arg749_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg749_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_604: "f32[512]" = torch.ops.aten.add.Tensor(arg751_1, 1e-05);  arg751_1 = None
        sqrt_270: "f32[512]" = torch.ops.aten.sqrt.default(add_604);  add_604 = None
        reciprocal_270: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_270);  sqrt_270 = None
        mul_874: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_270, 1);  reciprocal_270 = None
        unsqueeze_2160: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg750_1, -1);  arg750_1 = None
        unsqueeze_2161: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2160, -1);  unsqueeze_2160 = None
        unsqueeze_2162: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_874, -1);  mul_874 = None
        unsqueeze_2163: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2162, -1);  unsqueeze_2162 = None
        sub_334: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_334, unsqueeze_2161);  convolution_334 = unsqueeze_2161 = None
        mul_875: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_2163);  sub_334 = unsqueeze_2163 = None
        unsqueeze_2164: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg752_1, -1);  arg752_1 = None
        unsqueeze_2165: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2164, -1);  unsqueeze_2164 = None
        mul_876: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_875, unsqueeze_2165);  mul_875 = unsqueeze_2165 = None
        unsqueeze_2166: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg753_1, -1);  arg753_1 = None
        unsqueeze_2167: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2166, -1);  unsqueeze_2166 = None
        add_605: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_876, unsqueeze_2167);  mul_876 = unsqueeze_2167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_262: "f32[8, 512, 8, 8]" = torch.ops.aten.relu.default(add_605);  add_605 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_335: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(relu_262, arg754_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_262 = arg754_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_606: "f32[1024]" = torch.ops.aten.add.Tensor(arg756_1, 1e-05);  arg756_1 = None
        sqrt_271: "f32[1024]" = torch.ops.aten.sqrt.default(add_606);  add_606 = None
        reciprocal_271: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_271);  sqrt_271 = None
        mul_877: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_271, 1);  reciprocal_271 = None
        unsqueeze_2168: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg755_1, -1);  arg755_1 = None
        unsqueeze_2169: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2168, -1);  unsqueeze_2168 = None
        unsqueeze_2170: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_877, -1);  mul_877 = None
        unsqueeze_2171: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2170, -1);  unsqueeze_2170 = None
        sub_335: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_335, unsqueeze_2169);  convolution_335 = unsqueeze_2169 = None
        mul_878: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_2171);  sub_335 = unsqueeze_2171 = None
        unsqueeze_2172: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg757_1, -1);  arg757_1 = None
        unsqueeze_2173: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2172, -1);  unsqueeze_2172 = None
        mul_879: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_878, unsqueeze_2173);  mul_878 = unsqueeze_2173 = None
        unsqueeze_2174: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg758_1, -1);  arg758_1 = None
        unsqueeze_2175: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2174, -1);  unsqueeze_2174 = None
        add_607: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_879, unsqueeze_2175);  mul_879 = unsqueeze_2175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_263: "f32[8, 1024, 8, 8]" = torch.ops.aten.relu.default(add_607);  add_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_386: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.view.default(relu_263, [8, 2, 512, 8, 8]);  relu_263 = None
        sum_193: "f32[8, 512, 8, 8]" = torch.ops.aten.sum.dim_IntList(view_386, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_65: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(sum_193, [2, 3], True);  sum_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_336: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_65, arg759_1, arg760_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_65 = arg759_1 = arg760_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_608: "f32[256]" = torch.ops.aten.add.Tensor(arg762_1, 1e-05);  arg762_1 = None
        sqrt_272: "f32[256]" = torch.ops.aten.sqrt.default(add_608);  add_608 = None
        reciprocal_272: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_272);  sqrt_272 = None
        mul_880: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_272, 1);  reciprocal_272 = None
        unsqueeze_2176: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg761_1, -1);  arg761_1 = None
        unsqueeze_2177: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2176, -1);  unsqueeze_2176 = None
        unsqueeze_2178: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_880, -1);  mul_880 = None
        unsqueeze_2179: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2178, -1);  unsqueeze_2178 = None
        sub_336: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_336, unsqueeze_2177);  convolution_336 = unsqueeze_2177 = None
        mul_881: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_2179);  sub_336 = unsqueeze_2179 = None
        unsqueeze_2180: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg763_1, -1);  arg763_1 = None
        unsqueeze_2181: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2180, -1);  unsqueeze_2180 = None
        mul_882: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(mul_881, unsqueeze_2181);  mul_881 = unsqueeze_2181 = None
        unsqueeze_2182: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg764_1, -1);  arg764_1 = None
        unsqueeze_2183: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2182, -1);  unsqueeze_2182 = None
        add_609: "f32[8, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_882, unsqueeze_2183);  mul_882 = unsqueeze_2183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_264: "f32[8, 256, 1, 1]" = torch.ops.aten.relu.default(add_609);  add_609 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_337: "f32[8, 1024, 1, 1]" = torch.ops.aten.convolution.default(relu_264, arg765_1, arg766_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_264 = arg765_1 = arg766_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_387: "f32[8, 1, 2, 512]" = torch.ops.aten.view.default(convolution_337, [8, 1, 2, -1]);  convolution_337 = None
        permute_65: "f32[8, 2, 1, 512]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_64: "f32[8, 1, 1, 512]" = torch.ops.aten.amax.default(permute_65, [1], True)
        sub_337: "f32[8, 2, 1, 512]" = torch.ops.aten.sub.Tensor(permute_65, amax_64);  permute_65 = amax_64 = None
        exp_64: "f32[8, 2, 1, 512]" = torch.ops.aten.exp.default(sub_337);  sub_337 = None
        sum_194: "f32[8, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(exp_64, [1], True)
        div_64: "f32[8, 2, 1, 512]" = torch.ops.aten.div.Tensor(exp_64, sum_194);  exp_64 = sum_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_388: "f32[8, 1024]" = torch.ops.aten.view.default(div_64, [8, -1]);  div_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_389: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(view_388, [8, -1, 1, 1]);  view_388 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_390: "f32[8, 2, 512, 1, 1]" = torch.ops.aten.view.default(view_389, [8, 2, 512, 1, 1]);  view_389 = None
        mul_883: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.mul.Tensor(view_386, view_390);  view_386 = view_390 = None
        sum_195: "f32[8, 512, 8, 8]" = torch.ops.aten.sum.dim_IntList(mul_883, [1]);  mul_883 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_338: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(sum_195, arg767_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_195 = arg767_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_610: "f32[2048]" = torch.ops.aten.add.Tensor(arg769_1, 1e-05);  arg769_1 = None
        sqrt_273: "f32[2048]" = torch.ops.aten.sqrt.default(add_610);  add_610 = None
        reciprocal_273: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_273);  sqrt_273 = None
        mul_884: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_273, 1);  reciprocal_273 = None
        unsqueeze_2184: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg768_1, -1);  arg768_1 = None
        unsqueeze_2185: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2184, -1);  unsqueeze_2184 = None
        unsqueeze_2186: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_884, -1);  mul_884 = None
        unsqueeze_2187: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2186, -1);  unsqueeze_2186 = None
        sub_338: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_338, unsqueeze_2185);  convolution_338 = unsqueeze_2185 = None
        mul_885: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_2187);  sub_338 = unsqueeze_2187 = None
        unsqueeze_2188: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg770_1, -1);  arg770_1 = None
        unsqueeze_2189: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2188, -1);  unsqueeze_2188 = None
        mul_886: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_885, unsqueeze_2189);  mul_885 = unsqueeze_2189 = None
        unsqueeze_2190: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg771_1, -1);  arg771_1 = None
        unsqueeze_2191: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2190, -1);  unsqueeze_2190 = None
        add_611: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_886, unsqueeze_2191);  mul_886 = unsqueeze_2191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_612: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_611, relu_261);  add_611 = relu_261 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_265: "f32[8, 2048, 8, 8]" = torch.ops.aten.relu.default(add_612);  add_612 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:94 in forward, code: out = self.conv1(x)
        convolution_339: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(relu_265, arg772_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg772_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:95 in forward, code: out = self.bn1(out)
        add_613: "f32[512]" = torch.ops.aten.add.Tensor(arg774_1, 1e-05);  arg774_1 = None
        sqrt_274: "f32[512]" = torch.ops.aten.sqrt.default(add_613);  add_613 = None
        reciprocal_274: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_274);  sqrt_274 = None
        mul_887: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_274, 1);  reciprocal_274 = None
        unsqueeze_2192: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg773_1, -1);  arg773_1 = None
        unsqueeze_2193: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2192, -1);  unsqueeze_2192 = None
        unsqueeze_2194: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_887, -1);  mul_887 = None
        unsqueeze_2195: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2194, -1);  unsqueeze_2194 = None
        sub_339: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_339, unsqueeze_2193);  convolution_339 = unsqueeze_2193 = None
        mul_888: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_2195);  sub_339 = unsqueeze_2195 = None
        unsqueeze_2196: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg775_1, -1);  arg775_1 = None
        unsqueeze_2197: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2196, -1);  unsqueeze_2196 = None
        mul_889: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_888, unsqueeze_2197);  mul_888 = unsqueeze_2197 = None
        unsqueeze_2198: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg776_1, -1);  arg776_1 = None
        unsqueeze_2199: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2198, -1);  unsqueeze_2198 = None
        add_614: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_889, unsqueeze_2199);  mul_889 = unsqueeze_2199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:96 in forward, code: out = self.act1(out)
        relu_266: "f32[8, 512, 8, 8]" = torch.ops.aten.relu.default(add_614);  add_614 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:62 in forward, code: x = self.conv(x)
        convolution_340: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(relu_266, arg777_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_266 = arg777_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:63 in forward, code: x = self.bn0(x)
        add_615: "f32[1024]" = torch.ops.aten.add.Tensor(arg779_1, 1e-05);  arg779_1 = None
        sqrt_275: "f32[1024]" = torch.ops.aten.sqrt.default(add_615);  add_615 = None
        reciprocal_275: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_275);  sqrt_275 = None
        mul_890: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_275, 1);  reciprocal_275 = None
        unsqueeze_2200: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg778_1, -1);  arg778_1 = None
        unsqueeze_2201: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2200, -1);  unsqueeze_2200 = None
        unsqueeze_2202: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_890, -1);  mul_890 = None
        unsqueeze_2203: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2202, -1);  unsqueeze_2202 = None
        sub_340: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_340, unsqueeze_2201);  convolution_340 = unsqueeze_2201 = None
        mul_891: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_2203);  sub_340 = unsqueeze_2203 = None
        unsqueeze_2204: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg780_1, -1);  arg780_1 = None
        unsqueeze_2205: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2204, -1);  unsqueeze_2204 = None
        mul_892: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_891, unsqueeze_2205);  mul_891 = unsqueeze_2205 = None
        unsqueeze_2206: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg781_1, -1);  arg781_1 = None
        unsqueeze_2207: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2206, -1);  unsqueeze_2206 = None
        add_616: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_892, unsqueeze_2207);  mul_892 = unsqueeze_2207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:65 in forward, code: x = self.act0(x)
        relu_267: "f32[8, 1024, 8, 8]" = torch.ops.aten.relu.default(add_616);  add_616 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:70 in forward, code: x_gap = x.sum(dim=1)
        view_392: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.view.default(relu_267, [8, 2, 512, 8, 8]);  relu_267 = None
        sum_196: "f32[8, 512, 8, 8]" = torch.ops.aten.sum.dim_IntList(view_392, [1])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:73 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_66: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(sum_196, [2, 3], True);  sum_196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:74 in forward, code: x_gap = self.fc1(x_gap)
        convolution_341: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_66, arg782_1, arg783_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_66 = arg782_1 = arg783_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:75 in forward, code: x_gap = self.bn1(x_gap)
        add_617: "f32[256]" = torch.ops.aten.add.Tensor(arg785_1, 1e-05);  arg785_1 = None
        sqrt_276: "f32[256]" = torch.ops.aten.sqrt.default(add_617);  add_617 = None
        reciprocal_276: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_276);  sqrt_276 = None
        mul_893: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_276, 1);  reciprocal_276 = None
        unsqueeze_2208: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg784_1, -1);  arg784_1 = None
        unsqueeze_2209: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2208, -1);  unsqueeze_2208 = None
        unsqueeze_2210: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_893, -1);  mul_893 = None
        unsqueeze_2211: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2210, -1);  unsqueeze_2210 = None
        sub_341: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_341, unsqueeze_2209);  convolution_341 = unsqueeze_2209 = None
        mul_894: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_2211);  sub_341 = unsqueeze_2211 = None
        unsqueeze_2212: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg786_1, -1);  arg786_1 = None
        unsqueeze_2213: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2212, -1);  unsqueeze_2212 = None
        mul_895: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(mul_894, unsqueeze_2213);  mul_894 = unsqueeze_2213 = None
        unsqueeze_2214: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg787_1, -1);  arg787_1 = None
        unsqueeze_2215: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2214, -1);  unsqueeze_2214 = None
        add_618: "f32[8, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_895, unsqueeze_2215);  mul_895 = unsqueeze_2215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:76 in forward, code: x_gap = self.act1(x_gap)
        relu_268: "f32[8, 256, 1, 1]" = torch.ops.aten.relu.default(add_618);  add_618 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:77 in forward, code: x_attn = self.fc2(x_gap)
        convolution_342: "f32[8, 1024, 1, 1]" = torch.ops.aten.convolution.default(relu_268, arg788_1, arg789_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_268 = arg788_1 = arg789_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:25 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_393: "f32[8, 1, 2, 512]" = torch.ops.aten.view.default(convolution_342, [8, 1, 2, -1]);  convolution_342 = None
        permute_66: "f32[8, 2, 1, 512]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:26 in forward, code: x = F.softmax(x, dim=1)
        amax_65: "f32[8, 1, 1, 512]" = torch.ops.aten.amax.default(permute_66, [1], True)
        sub_342: "f32[8, 2, 1, 512]" = torch.ops.aten.sub.Tensor(permute_66, amax_65);  permute_66 = amax_65 = None
        exp_65: "f32[8, 2, 1, 512]" = torch.ops.aten.exp.default(sub_342);  sub_342 = None
        sum_197: "f32[8, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(exp_65, [1], True)
        div_65: "f32[8, 2, 1, 512]" = torch.ops.aten.div.Tensor(exp_65, sum_197);  exp_65 = sum_197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.reshape(batch, -1)
        view_394: "f32[8, 1024]" = torch.ops.aten.view.default(div_65, [8, -1]);  div_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:79 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_395: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(view_394, [8, -1, 1, 1]);  view_394 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/split_attn.py:81 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_396: "f32[8, 2, 512, 1, 1]" = torch.ops.aten.view.default(view_395, [8, 2, 512, 1, 1]);  view_395 = None
        mul_896: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.mul.Tensor(view_392, view_396);  view_392 = view_396 = None
        sum_198: "f32[8, 512, 8, 8]" = torch.ops.aten.sum.dim_IntList(mul_896, [1]);  mul_896 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:109 in forward, code: out = self.conv3(out)
        convolution_343: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(sum_198, arg790_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_198 = arg790_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:110 in forward, code: out = self.bn3(out)
        add_619: "f32[2048]" = torch.ops.aten.add.Tensor(arg792_1, 1e-05);  arg792_1 = None
        sqrt_277: "f32[2048]" = torch.ops.aten.sqrt.default(add_619);  add_619 = None
        reciprocal_277: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_277);  sqrt_277 = None
        mul_897: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_277, 1);  reciprocal_277 = None
        unsqueeze_2216: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg791_1, -1);  arg791_1 = None
        unsqueeze_2217: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2216, -1);  unsqueeze_2216 = None
        unsqueeze_2218: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_897, -1);  mul_897 = None
        unsqueeze_2219: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2218, -1);  unsqueeze_2218 = None
        sub_343: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_343, unsqueeze_2217);  convolution_343 = unsqueeze_2217 = None
        mul_898: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_2219);  sub_343 = unsqueeze_2219 = None
        unsqueeze_2220: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg793_1, -1);  arg793_1 = None
        unsqueeze_2221: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2220, -1);  unsqueeze_2220 = None
        mul_899: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_898, unsqueeze_2221);  mul_898 = unsqueeze_2221 = None
        unsqueeze_2222: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg794_1, -1);  arg794_1 = None
        unsqueeze_2223: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2222, -1);  unsqueeze_2222 = None
        add_620: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_899, unsqueeze_2223);  mul_899 = unsqueeze_2223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:115 in forward, code: out += shortcut
        add_621: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_620, relu_265);  add_620 = relu_265 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnest.py:116 in forward, code: out = self.act3(out)
        relu_269: "f32[8, 2048, 8, 8]" = torch.ops.aten.relu.default(add_621);  add_621 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_67: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_269, [-1, -2], True);  relu_269 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_397: "f32[8, 2048]" = torch.ops.aten.view.default(mean_67, [8, 2048]);  mean_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/resnet.py:633 in forward_head, code: return x if pre_logits else self.fc(x)
        permute_67: "f32[2048, 1000]" = torch.ops.aten.permute.default(arg795_1, [1, 0]);  arg795_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg796_1, view_397, permute_67);  arg796_1 = view_397 = permute_67 = None
        return (addmm_1,)
        