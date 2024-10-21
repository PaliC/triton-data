class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 384, 384]", arg1_1: "f32[768, 3, 16, 16]", arg2_1: "f32[768]", arg3_1: "f32[1, 576, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[2304, 768]", arg8_1: "f32[2304]", arg9_1: "f32[16, 16]", arg10_1: "f32[16]", arg11_1: "f32[16, 16]", arg12_1: "f32[16]", arg13_1: "f32[768, 768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[3072, 768]", arg19_1: "f32[3072]", arg20_1: "f32[768, 3072]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[2304, 768]", arg26_1: "f32[2304]", arg27_1: "f32[16, 16]", arg28_1: "f32[16]", arg29_1: "f32[16, 16]", arg30_1: "f32[16]", arg31_1: "f32[768, 768]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[3072, 768]", arg37_1: "f32[3072]", arg38_1: "f32[768, 3072]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[768]", arg43_1: "f32[2304, 768]", arg44_1: "f32[2304]", arg45_1: "f32[16, 16]", arg46_1: "f32[16]", arg47_1: "f32[16, 16]", arg48_1: "f32[16]", arg49_1: "f32[768, 768]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[3072, 768]", arg55_1: "f32[3072]", arg56_1: "f32[768, 3072]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[2304, 768]", arg62_1: "f32[2304]", arg63_1: "f32[16, 16]", arg64_1: "f32[16]", arg65_1: "f32[16, 16]", arg66_1: "f32[16]", arg67_1: "f32[768, 768]", arg68_1: "f32[768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[3072, 768]", arg73_1: "f32[3072]", arg74_1: "f32[768, 3072]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[2304, 768]", arg80_1: "f32[2304]", arg81_1: "f32[16, 16]", arg82_1: "f32[16]", arg83_1: "f32[16, 16]", arg84_1: "f32[16]", arg85_1: "f32[768, 768]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[768]", arg90_1: "f32[3072, 768]", arg91_1: "f32[3072]", arg92_1: "f32[768, 3072]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[768]", arg97_1: "f32[2304, 768]", arg98_1: "f32[2304]", arg99_1: "f32[16, 16]", arg100_1: "f32[16]", arg101_1: "f32[16, 16]", arg102_1: "f32[16]", arg103_1: "f32[768, 768]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg106_1: "f32[768]", arg107_1: "f32[768]", arg108_1: "f32[3072, 768]", arg109_1: "f32[3072]", arg110_1: "f32[768, 3072]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[2304, 768]", arg116_1: "f32[2304]", arg117_1: "f32[16, 16]", arg118_1: "f32[16]", arg119_1: "f32[16, 16]", arg120_1: "f32[16]", arg121_1: "f32[768, 768]", arg122_1: "f32[768]", arg123_1: "f32[768]", arg124_1: "f32[768]", arg125_1: "f32[768]", arg126_1: "f32[3072, 768]", arg127_1: "f32[3072]", arg128_1: "f32[768, 3072]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[768]", arg132_1: "f32[768]", arg133_1: "f32[2304, 768]", arg134_1: "f32[2304]", arg135_1: "f32[16, 16]", arg136_1: "f32[16]", arg137_1: "f32[16, 16]", arg138_1: "f32[16]", arg139_1: "f32[768, 768]", arg140_1: "f32[768]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[768]", arg144_1: "f32[3072, 768]", arg145_1: "f32[3072]", arg146_1: "f32[768, 3072]", arg147_1: "f32[768]", arg148_1: "f32[768]", arg149_1: "f32[768]", arg150_1: "f32[768]", arg151_1: "f32[2304, 768]", arg152_1: "f32[2304]", arg153_1: "f32[16, 16]", arg154_1: "f32[16]", arg155_1: "f32[16, 16]", arg156_1: "f32[16]", arg157_1: "f32[768, 768]", arg158_1: "f32[768]", arg159_1: "f32[768]", arg160_1: "f32[768]", arg161_1: "f32[768]", arg162_1: "f32[3072, 768]", arg163_1: "f32[3072]", arg164_1: "f32[768, 3072]", arg165_1: "f32[768]", arg166_1: "f32[768]", arg167_1: "f32[768]", arg168_1: "f32[768]", arg169_1: "f32[2304, 768]", arg170_1: "f32[2304]", arg171_1: "f32[16, 16]", arg172_1: "f32[16]", arg173_1: "f32[16, 16]", arg174_1: "f32[16]", arg175_1: "f32[768, 768]", arg176_1: "f32[768]", arg177_1: "f32[768]", arg178_1: "f32[768]", arg179_1: "f32[768]", arg180_1: "f32[3072, 768]", arg181_1: "f32[3072]", arg182_1: "f32[768, 3072]", arg183_1: "f32[768]", arg184_1: "f32[768]", arg185_1: "f32[768]", arg186_1: "f32[768]", arg187_1: "f32[2304, 768]", arg188_1: "f32[2304]", arg189_1: "f32[16, 16]", arg190_1: "f32[16]", arg191_1: "f32[16, 16]", arg192_1: "f32[16]", arg193_1: "f32[768, 768]", arg194_1: "f32[768]", arg195_1: "f32[768]", arg196_1: "f32[768]", arg197_1: "f32[768]", arg198_1: "f32[3072, 768]", arg199_1: "f32[3072]", arg200_1: "f32[768, 3072]", arg201_1: "f32[768]", arg202_1: "f32[768]", arg203_1: "f32[768]", arg204_1: "f32[768]", arg205_1: "f32[2304, 768]", arg206_1: "f32[2304]", arg207_1: "f32[16, 16]", arg208_1: "f32[16]", arg209_1: "f32[16, 16]", arg210_1: "f32[16]", arg211_1: "f32[768, 768]", arg212_1: "f32[768]", arg213_1: "f32[768]", arg214_1: "f32[768]", arg215_1: "f32[768]", arg216_1: "f32[3072, 768]", arg217_1: "f32[3072]", arg218_1: "f32[768, 3072]", arg219_1: "f32[768]", arg220_1: "f32[768]", arg221_1: "f32[768]", arg222_1: "f32[768]", arg223_1: "f32[2304, 768]", arg224_1: "f32[2304]", arg225_1: "f32[16, 16]", arg226_1: "f32[16]", arg227_1: "f32[16, 16]", arg228_1: "f32[16]", arg229_1: "f32[768, 768]", arg230_1: "f32[768]", arg231_1: "f32[768]", arg232_1: "f32[768]", arg233_1: "f32[768]", arg234_1: "f32[3072, 768]", arg235_1: "f32[3072]", arg236_1: "f32[768, 3072]", arg237_1: "f32[768]", arg238_1: "f32[768]", arg239_1: "f32[768]", arg240_1: "f32[768]", arg241_1: "f32[2304, 768]", arg242_1: "f32[2304]", arg243_1: "f32[16, 16]", arg244_1: "f32[16]", arg245_1: "f32[16, 16]", arg246_1: "f32[16]", arg247_1: "f32[768, 768]", arg248_1: "f32[768]", arg249_1: "f32[768]", arg250_1: "f32[768]", arg251_1: "f32[768]", arg252_1: "f32[3072, 768]", arg253_1: "f32[3072]", arg254_1: "f32[768, 3072]", arg255_1: "f32[768]", arg256_1: "f32[768]", arg257_1: "f32[768]", arg258_1: "f32[768]", arg259_1: "f32[2304, 768]", arg260_1: "f32[2304]", arg261_1: "f32[16, 16]", arg262_1: "f32[16]", arg263_1: "f32[16, 16]", arg264_1: "f32[16]", arg265_1: "f32[768, 768]", arg266_1: "f32[768]", arg267_1: "f32[768]", arg268_1: "f32[768]", arg269_1: "f32[768]", arg270_1: "f32[3072, 768]", arg271_1: "f32[3072]", arg272_1: "f32[768, 3072]", arg273_1: "f32[768]", arg274_1: "f32[768]", arg275_1: "f32[768]", arg276_1: "f32[768]", arg277_1: "f32[2304, 768]", arg278_1: "f32[2304]", arg279_1: "f32[16, 16]", arg280_1: "f32[16]", arg281_1: "f32[16, 16]", arg282_1: "f32[16]", arg283_1: "f32[768, 768]", arg284_1: "f32[768]", arg285_1: "f32[768]", arg286_1: "f32[768]", arg287_1: "f32[768]", arg288_1: "f32[3072, 768]", arg289_1: "f32[3072]", arg290_1: "f32[768, 3072]", arg291_1: "f32[768]", arg292_1: "f32[768]", arg293_1: "f32[768]", arg294_1: "f32[768]", arg295_1: "f32[2304, 768]", arg296_1: "f32[2304]", arg297_1: "f32[16, 16]", arg298_1: "f32[16]", arg299_1: "f32[16, 16]", arg300_1: "f32[16]", arg301_1: "f32[768, 768]", arg302_1: "f32[768]", arg303_1: "f32[768]", arg304_1: "f32[768]", arg305_1: "f32[768]", arg306_1: "f32[3072, 768]", arg307_1: "f32[3072]", arg308_1: "f32[768, 3072]", arg309_1: "f32[768]", arg310_1: "f32[768]", arg311_1: "f32[768]", arg312_1: "f32[768]", arg313_1: "f32[2304, 768]", arg314_1: "f32[2304]", arg315_1: "f32[16, 16]", arg316_1: "f32[16]", arg317_1: "f32[16, 16]", arg318_1: "f32[16]", arg319_1: "f32[768, 768]", arg320_1: "f32[768]", arg321_1: "f32[768]", arg322_1: "f32[768]", arg323_1: "f32[768]", arg324_1: "f32[3072, 768]", arg325_1: "f32[3072]", arg326_1: "f32[768, 3072]", arg327_1: "f32[768]", arg328_1: "f32[768]", arg329_1: "f32[768]", arg330_1: "f32[768]", arg331_1: "f32[2304, 768]", arg332_1: "f32[2304]", arg333_1: "f32[16, 16]", arg334_1: "f32[16]", arg335_1: "f32[16, 16]", arg336_1: "f32[16]", arg337_1: "f32[768, 768]", arg338_1: "f32[768]", arg339_1: "f32[768]", arg340_1: "f32[768]", arg341_1: "f32[768]", arg342_1: "f32[3072, 768]", arg343_1: "f32[3072]", arg344_1: "f32[768, 3072]", arg345_1: "f32[768]", arg346_1: "f32[768]", arg347_1: "f32[768]", arg348_1: "f32[768]", arg349_1: "f32[2304, 768]", arg350_1: "f32[2304]", arg351_1: "f32[16, 16]", arg352_1: "f32[16]", arg353_1: "f32[16, 16]", arg354_1: "f32[16]", arg355_1: "f32[768, 768]", arg356_1: "f32[768]", arg357_1: "f32[768]", arg358_1: "f32[768]", arg359_1: "f32[768]", arg360_1: "f32[3072, 768]", arg361_1: "f32[3072]", arg362_1: "f32[768, 3072]", arg363_1: "f32[768]", arg364_1: "f32[768]", arg365_1: "f32[768]", arg366_1: "f32[768]", arg367_1: "f32[2304, 768]", arg368_1: "f32[2304]", arg369_1: "f32[16, 16]", arg370_1: "f32[16]", arg371_1: "f32[16, 16]", arg372_1: "f32[16]", arg373_1: "f32[768, 768]", arg374_1: "f32[768]", arg375_1: "f32[768]", arg376_1: "f32[768]", arg377_1: "f32[768]", arg378_1: "f32[3072, 768]", arg379_1: "f32[3072]", arg380_1: "f32[768, 3072]", arg381_1: "f32[768]", arg382_1: "f32[768]", arg383_1: "f32[768]", arg384_1: "f32[768]", arg385_1: "f32[2304, 768]", arg386_1: "f32[2304]", arg387_1: "f32[16, 16]", arg388_1: "f32[16]", arg389_1: "f32[16, 16]", arg390_1: "f32[16]", arg391_1: "f32[768, 768]", arg392_1: "f32[768]", arg393_1: "f32[768]", arg394_1: "f32[768]", arg395_1: "f32[768]", arg396_1: "f32[3072, 768]", arg397_1: "f32[3072]", arg398_1: "f32[768, 3072]", arg399_1: "f32[768]", arg400_1: "f32[768]", arg401_1: "f32[768]", arg402_1: "f32[768]", arg403_1: "f32[2304, 768]", arg404_1: "f32[2304]", arg405_1: "f32[16, 16]", arg406_1: "f32[16]", arg407_1: "f32[16, 16]", arg408_1: "f32[16]", arg409_1: "f32[768, 768]", arg410_1: "f32[768]", arg411_1: "f32[768]", arg412_1: "f32[768]", arg413_1: "f32[768]", arg414_1: "f32[3072, 768]", arg415_1: "f32[3072]", arg416_1: "f32[768, 3072]", arg417_1: "f32[768]", arg418_1: "f32[768]", arg419_1: "f32[768]", arg420_1: "f32[768]", arg421_1: "f32[2304, 768]", arg422_1: "f32[2304]", arg423_1: "f32[16, 16]", arg424_1: "f32[16]", arg425_1: "f32[16, 16]", arg426_1: "f32[16]", arg427_1: "f32[768, 768]", arg428_1: "f32[768]", arg429_1: "f32[768]", arg430_1: "f32[768]", arg431_1: "f32[768]", arg432_1: "f32[3072, 768]", arg433_1: "f32[3072]", arg434_1: "f32[768, 3072]", arg435_1: "f32[768]", arg436_1: "f32[768]", arg437_1: "f32[768]", arg438_1: "f32[768]", arg439_1: "f32[2304, 768]", arg440_1: "f32[2304]", arg441_1: "f32[16, 16]", arg442_1: "f32[16]", arg443_1: "f32[16, 16]", arg444_1: "f32[16]", arg445_1: "f32[768, 768]", arg446_1: "f32[768]", arg447_1: "f32[768]", arg448_1: "f32[768]", arg449_1: "f32[768]", arg450_1: "f32[3072, 768]", arg451_1: "f32[3072]", arg452_1: "f32[768, 3072]", arg453_1: "f32[768]", arg454_1: "f32[768]", arg455_1: "f32[768]", arg456_1: "f32[768]", arg457_1: "f32[2304, 768]", arg458_1: "f32[2304]", arg459_1: "f32[16, 16]", arg460_1: "f32[16]", arg461_1: "f32[16, 16]", arg462_1: "f32[16]", arg463_1: "f32[768, 768]", arg464_1: "f32[768]", arg465_1: "f32[768]", arg466_1: "f32[768]", arg467_1: "f32[768]", arg468_1: "f32[3072, 768]", arg469_1: "f32[3072]", arg470_1: "f32[768, 3072]", arg471_1: "f32[768]", arg472_1: "f32[768]", arg473_1: "f32[768]", arg474_1: "f32[768]", arg475_1: "f32[2304, 768]", arg476_1: "f32[2304]", arg477_1: "f32[16, 16]", arg478_1: "f32[16]", arg479_1: "f32[16, 16]", arg480_1: "f32[16]", arg481_1: "f32[768, 768]", arg482_1: "f32[768]", arg483_1: "f32[768]", arg484_1: "f32[768]", arg485_1: "f32[768]", arg486_1: "f32[3072, 768]", arg487_1: "f32[3072]", arg488_1: "f32[768, 3072]", arg489_1: "f32[768]", arg490_1: "f32[768]", arg491_1: "f32[768]", arg492_1: "f32[768]", arg493_1: "f32[2304, 768]", arg494_1: "f32[2304]", arg495_1: "f32[16, 16]", arg496_1: "f32[16]", arg497_1: "f32[16, 16]", arg498_1: "f32[16]", arg499_1: "f32[768, 768]", arg500_1: "f32[768]", arg501_1: "f32[768]", arg502_1: "f32[768]", arg503_1: "f32[768]", arg504_1: "f32[3072, 768]", arg505_1: "f32[3072]", arg506_1: "f32[768, 3072]", arg507_1: "f32[768]", arg508_1: "f32[768]", arg509_1: "f32[768]", arg510_1: "f32[768]", arg511_1: "f32[2304, 768]", arg512_1: "f32[2304]", arg513_1: "f32[16, 16]", arg514_1: "f32[16]", arg515_1: "f32[16, 16]", arg516_1: "f32[16]", arg517_1: "f32[768, 768]", arg518_1: "f32[768]", arg519_1: "f32[768]", arg520_1: "f32[768]", arg521_1: "f32[768]", arg522_1: "f32[3072, 768]", arg523_1: "f32[3072]", arg524_1: "f32[768, 3072]", arg525_1: "f32[768]", arg526_1: "f32[768]", arg527_1: "f32[768]", arg528_1: "f32[768]", arg529_1: "f32[2304, 768]", arg530_1: "f32[2304]", arg531_1: "f32[16, 16]", arg532_1: "f32[16]", arg533_1: "f32[16, 16]", arg534_1: "f32[16]", arg535_1: "f32[768, 768]", arg536_1: "f32[768]", arg537_1: "f32[768]", arg538_1: "f32[768]", arg539_1: "f32[768]", arg540_1: "f32[3072, 768]", arg541_1: "f32[3072]", arg542_1: "f32[768, 3072]", arg543_1: "f32[768]", arg544_1: "f32[768]", arg545_1: "f32[768]", arg546_1: "f32[768]", arg547_1: "f32[2304, 768]", arg548_1: "f32[2304]", arg549_1: "f32[16, 16]", arg550_1: "f32[16]", arg551_1: "f32[16, 16]", arg552_1: "f32[16]", arg553_1: "f32[768, 768]", arg554_1: "f32[768]", arg555_1: "f32[768]", arg556_1: "f32[768]", arg557_1: "f32[768]", arg558_1: "f32[3072, 768]", arg559_1: "f32[3072]", arg560_1: "f32[768, 3072]", arg561_1: "f32[768]", arg562_1: "f32[768]", arg563_1: "f32[768]", arg564_1: "f32[768]", arg565_1: "f32[2304, 768]", arg566_1: "f32[2304]", arg567_1: "f32[16, 16]", arg568_1: "f32[16]", arg569_1: "f32[16, 16]", arg570_1: "f32[16]", arg571_1: "f32[768, 768]", arg572_1: "f32[768]", arg573_1: "f32[768]", arg574_1: "f32[768]", arg575_1: "f32[768]", arg576_1: "f32[3072, 768]", arg577_1: "f32[3072]", arg578_1: "f32[768, 3072]", arg579_1: "f32[768]", arg580_1: "f32[768]", arg581_1: "f32[768]", arg582_1: "f32[768]", arg583_1: "f32[2304, 768]", arg584_1: "f32[2304]", arg585_1: "f32[16, 16]", arg586_1: "f32[16]", arg587_1: "f32[16, 16]", arg588_1: "f32[16]", arg589_1: "f32[768, 768]", arg590_1: "f32[768]", arg591_1: "f32[768]", arg592_1: "f32[768]", arg593_1: "f32[768]", arg594_1: "f32[3072, 768]", arg595_1: "f32[3072]", arg596_1: "f32[768, 3072]", arg597_1: "f32[768]", arg598_1: "f32[768]", arg599_1: "f32[768]", arg600_1: "f32[768]", arg601_1: "f32[2304, 768]", arg602_1: "f32[2304]", arg603_1: "f32[16, 16]", arg604_1: "f32[16]", arg605_1: "f32[16, 16]", arg606_1: "f32[16]", arg607_1: "f32[768, 768]", arg608_1: "f32[768]", arg609_1: "f32[768]", arg610_1: "f32[768]", arg611_1: "f32[768]", arg612_1: "f32[3072, 768]", arg613_1: "f32[3072]", arg614_1: "f32[768, 3072]", arg615_1: "f32[768]", arg616_1: "f32[768]", arg617_1: "f32[768]", arg618_1: "f32[768]", arg619_1: "f32[2304, 768]", arg620_1: "f32[2304]", arg621_1: "f32[16, 16]", arg622_1: "f32[16]", arg623_1: "f32[16, 16]", arg624_1: "f32[16]", arg625_1: "f32[768, 768]", arg626_1: "f32[768]", arg627_1: "f32[768]", arg628_1: "f32[768]", arg629_1: "f32[768]", arg630_1: "f32[3072, 768]", arg631_1: "f32[3072]", arg632_1: "f32[768, 3072]", arg633_1: "f32[768]", arg634_1: "f32[768]", arg635_1: "f32[768]", arg636_1: "f32[768]", arg637_1: "f32[2304, 768]", arg638_1: "f32[2304]", arg639_1: "f32[16, 16]", arg640_1: "f32[16]", arg641_1: "f32[16, 16]", arg642_1: "f32[16]", arg643_1: "f32[768, 768]", arg644_1: "f32[768]", arg645_1: "f32[768]", arg646_1: "f32[768]", arg647_1: "f32[768]", arg648_1: "f32[3072, 768]", arg649_1: "f32[3072]", arg650_1: "f32[768, 3072]", arg651_1: "f32[768]", arg652_1: "f32[1, 1, 768]", arg653_1: "f32[768]", arg654_1: "f32[768]", arg655_1: "f32[768]", arg656_1: "f32[768, 768]", arg657_1: "f32[768]", arg658_1: "f32[768, 768]", arg659_1: "f32[768]", arg660_1: "f32[768, 768]", arg661_1: "f32[768]", arg662_1: "f32[768, 768]", arg663_1: "f32[768]", arg664_1: "f32[768]", arg665_1: "f32[768]", arg666_1: "f32[768]", arg667_1: "f32[3072, 768]", arg668_1: "f32[3072]", arg669_1: "f32[768, 3072]", arg670_1: "f32[768]", arg671_1: "f32[768]", arg672_1: "f32[768]", arg673_1: "f32[768]", arg674_1: "f32[768, 768]", arg675_1: "f32[768]", arg676_1: "f32[768, 768]", arg677_1: "f32[768]", arg678_1: "f32[768, 768]", arg679_1: "f32[768]", arg680_1: "f32[768, 768]", arg681_1: "f32[768]", arg682_1: "f32[768]", arg683_1: "f32[768]", arg684_1: "f32[768]", arg685_1: "f32[3072, 768]", arg686_1: "f32[3072]", arg687_1: "f32[768, 3072]", arg688_1: "f32[768]", arg689_1: "f32[768]", arg690_1: "f32[768]", arg691_1: "f32[1000, 768]", arg692_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_1: "f32[8, 768, 24, 24]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:133 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view_749: "f32[8, 768, 576]" = torch.ops.aten.reshape.default(convolution_1, [8, 768, 576]);  convolution_1 = None
        permute_490: "f32[8, 576, 768]" = torch.ops.aten.permute.default(view_749, [0, 2, 1]);  view_749 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:418 in forward_features, code: x = x + self.pos_embed
        add_341: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(permute_490, arg3_1);  permute_490 = arg3_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_513: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_341, memory_format = torch.contiguous_format)
        var_mean_77 = torch.ops.aten.var_mean.correction(clone_513, [2], correction = 0, keepdim = True)
        getitem_162: "f32[8, 576, 1]" = var_mean_77[0]
        getitem_163: "f32[8, 576, 1]" = var_mean_77[1];  var_mean_77 = None
        sub_113: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_513, getitem_163);  clone_513 = getitem_163 = None
        add_342: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
        rsqrt_77: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_342);  add_342 = None
        mul_380: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_77);  sub_113 = rsqrt_77 = None
        mul_381: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_380, arg5_1);  mul_380 = arg5_1 = None
        add_343: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_381, arg6_1);  mul_381 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_750: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_343, [4608, 768]);  add_343 = None
        permute_491: "f32[768, 2304]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        
        # No stacktrace found for following nodes
        mm_default_149: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_750, permute_491);  view_750 = permute_491 = None
        add_tensor_149: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_149, arg8_1);  mm_default_149 = arg8_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_751: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_149, [8, 576, 2304]);  add_tensor_149 = None
        view_752: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_751, [8, 576, 3, 16, 48]);  view_751 = None
        permute_492: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_752, [2, 0, 3, 1, 4]);  view_752 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_111: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_492, 0, 0)
        mul_382: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_111, 0.14433756729740643);  select_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_145: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_382, [8, 16, 576, 48]);  mul_382 = None
        clone_514: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_753: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_514, [128, 576, 48]);  clone_514 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_112: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_492, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_493: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_112, [0, 1, 3, 2]);  select_112 = None
        expand_146: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_493, [8, 16, 48, 576]);  permute_493 = None
        clone_515: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_146, memory_format = torch.contiguous_format);  expand_146 = None
        view_754: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_515, [128, 48, 576]);  clone_515 = None
        bmm_72: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_753, view_754);  view_753 = view_754 = None
        view_755: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_72, [8, 16, 576, 576]);  bmm_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_494: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_755, [0, 2, 3, 1]);  view_755 = None
        clone_516: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_494, memory_format = torch.contiguous_format);  permute_494 = None
        view_756: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_516, [2654208, 16]);  clone_516 = None
        permute_495: "f32[16, 16]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_72: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_756, permute_495);  view_756 = permute_495 = None
        view_757: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_72, [8, 576, 576, 16]);  mm_72 = None
        add_344: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_757, arg10_1);  view_757 = arg10_1 = None
        permute_496: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_344, [0, 3, 1, 2]);  add_344 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_517: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_496, memory_format = torch.contiguous_format);  permute_496 = None
        amax_36: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_517, [-1], True)
        sub_114: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_517, amax_36);  clone_517 = amax_36 = None
        exp_36: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_114);  sub_114 = None
        sum_37: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_36, [-1], True)
        div_36: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_36, sum_37);  exp_36 = sum_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_497: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_36, [0, 2, 3, 1]);  div_36 = None
        clone_518: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_497, memory_format = torch.contiguous_format);  permute_497 = None
        view_758: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_518, [2654208, 16]);  clone_518 = None
        permute_498: "f32[16, 16]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_73: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_758, permute_498);  view_758 = permute_498 = None
        view_759: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_73, [8, 576, 576, 16]);  mm_73 = None
        add_345: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_759, arg12_1);  view_759 = arg12_1 = None
        permute_499: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_345, [0, 3, 1, 2]);  add_345 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_147: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_499, [8, 16, 576, 576]);  permute_499 = None
        clone_520: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_147, memory_format = torch.contiguous_format);  expand_147 = None
        view_760: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_520, [128, 576, 576]);  clone_520 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_113: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_492, 0, 2);  permute_492 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_148: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_113, [8, 16, 576, 48]);  select_113 = None
        clone_521: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_148, memory_format = torch.contiguous_format);  expand_148 = None
        view_761: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_521, [128, 576, 48]);  clone_521 = None
        bmm_73: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_760, view_761);  view_760 = view_761 = None
        view_762: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_73, [8, 16, 576, 48]);  bmm_73 = None
        permute_500: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_762, [0, 2, 1, 3]);  view_762 = None
        clone_522: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
        view_763: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_522, [8, 576, 768]);  clone_522 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_764: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_763, [4608, 768]);  view_763 = None
        permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        
        # No stacktrace found for following nodes
        mm_default_148: "f32[4608, 768]" = torch.ops.aten.mm.default(view_764, permute_501);  view_764 = permute_501 = None
        add_tensor_148: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_148, arg14_1);  mm_default_148 = arg14_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_765: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_148, [8, 576, 768]);  add_tensor_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_383: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg4_1, view_765);  arg4_1 = view_765 = None
        add_346: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_341, mul_383);  add_341 = mul_383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_524: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_346, memory_format = torch.contiguous_format)
        var_mean_78 = torch.ops.aten.var_mean.correction(clone_524, [2], correction = 0, keepdim = True)
        getitem_164: "f32[8, 576, 1]" = var_mean_78[0]
        getitem_165: "f32[8, 576, 1]" = var_mean_78[1];  var_mean_78 = None
        sub_115: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_524, getitem_165);  clone_524 = getitem_165 = None
        add_347: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-06);  getitem_164 = None
        rsqrt_78: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_347);  add_347 = None
        mul_384: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_78);  sub_115 = rsqrt_78 = None
        mul_385: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_384, arg16_1);  mul_384 = arg16_1 = None
        add_348: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_385, arg17_1);  mul_385 = arg17_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_766: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_348, [4608, 768]);  add_348 = None
        permute_502: "f32[768, 3072]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        
        # No stacktrace found for following nodes
        mm_default_147: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_766, permute_502);  view_766 = permute_502 = None
        add_tensor_147: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_147, arg19_1);  mm_default_147 = arg19_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_767: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_147, [8, 576, 3072]);  add_tensor_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_386: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_767, 0.5)
        mul_387: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_767, 0.7071067811865476);  view_767 = None
        erf_38: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_387);  mul_387 = None
        add_349: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_388: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_386, add_349);  mul_386 = add_349 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_768: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_388, [4608, 3072]);  mul_388 = None
        permute_503: "f32[3072, 768]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        
        # No stacktrace found for following nodes
        mm_default_146: "f32[4608, 768]" = torch.ops.aten.mm.default(view_768, permute_503);  view_768 = permute_503 = None
        add_tensor_146: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_146, arg21_1);  mm_default_146 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_769: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_146, [8, 576, 768]);  add_tensor_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_389: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg15_1, view_769);  arg15_1 = view_769 = None
        add_350: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_346, mul_389);  add_346 = mul_389 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_527: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_350, memory_format = torch.contiguous_format)
        var_mean_79 = torch.ops.aten.var_mean.correction(clone_527, [2], correction = 0, keepdim = True)
        getitem_166: "f32[8, 576, 1]" = var_mean_79[0]
        getitem_167: "f32[8, 576, 1]" = var_mean_79[1];  var_mean_79 = None
        sub_116: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_527, getitem_167);  clone_527 = getitem_167 = None
        add_351: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-06);  getitem_166 = None
        rsqrt_79: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
        mul_390: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_79);  sub_116 = rsqrt_79 = None
        mul_391: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_390, arg23_1);  mul_390 = arg23_1 = None
        add_352: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_391, arg24_1);  mul_391 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_770: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_352, [4608, 768]);  add_352 = None
        permute_504: "f32[768, 2304]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        
        # No stacktrace found for following nodes
        mm_default_145: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_770, permute_504);  view_770 = permute_504 = None
        add_tensor_145: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_145, arg26_1);  mm_default_145 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_771: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_145, [8, 576, 2304]);  add_tensor_145 = None
        view_772: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_771, [8, 576, 3, 16, 48]);  view_771 = None
        permute_505: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_772, [2, 0, 3, 1, 4]);  view_772 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_114: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_505, 0, 0)
        mul_392: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_114, 0.14433756729740643);  select_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_149: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_392, [8, 16, 576, 48]);  mul_392 = None
        clone_528: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_149, memory_format = torch.contiguous_format);  expand_149 = None
        view_773: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_528, [128, 576, 48]);  clone_528 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_115: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_505, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_506: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_115, [0, 1, 3, 2]);  select_115 = None
        expand_150: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_506, [8, 16, 48, 576]);  permute_506 = None
        clone_529: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_150, memory_format = torch.contiguous_format);  expand_150 = None
        view_774: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_529, [128, 48, 576]);  clone_529 = None
        bmm_74: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_773, view_774);  view_773 = view_774 = None
        view_775: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_74, [8, 16, 576, 576]);  bmm_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_507: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_775, [0, 2, 3, 1]);  view_775 = None
        clone_530: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_507, memory_format = torch.contiguous_format);  permute_507 = None
        view_776: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_530, [2654208, 16]);  clone_530 = None
        permute_508: "f32[16, 16]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        mm_74: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_776, permute_508);  view_776 = permute_508 = None
        view_777: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_74, [8, 576, 576, 16]);  mm_74 = None
        add_353: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_777, arg28_1);  view_777 = arg28_1 = None
        permute_509: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_353, [0, 3, 1, 2]);  add_353 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_531: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_509, memory_format = torch.contiguous_format);  permute_509 = None
        amax_37: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_531, [-1], True)
        sub_117: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_531, amax_37);  clone_531 = amax_37 = None
        exp_37: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_117);  sub_117 = None
        sum_38: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_37, [-1], True)
        div_37: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_37, sum_38);  exp_37 = sum_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_510: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_37, [0, 2, 3, 1]);  div_37 = None
        clone_532: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_510, memory_format = torch.contiguous_format);  permute_510 = None
        view_778: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_532, [2654208, 16]);  clone_532 = None
        permute_511: "f32[16, 16]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_75: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_778, permute_511);  view_778 = permute_511 = None
        view_779: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_75, [8, 576, 576, 16]);  mm_75 = None
        add_354: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_779, arg30_1);  view_779 = arg30_1 = None
        permute_512: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_354, [0, 3, 1, 2]);  add_354 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_151: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_512, [8, 16, 576, 576]);  permute_512 = None
        clone_534: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_151, memory_format = torch.contiguous_format);  expand_151 = None
        view_780: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_534, [128, 576, 576]);  clone_534 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_116: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_505, 0, 2);  permute_505 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_152: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_116, [8, 16, 576, 48]);  select_116 = None
        clone_535: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_152, memory_format = torch.contiguous_format);  expand_152 = None
        view_781: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_535, [128, 576, 48]);  clone_535 = None
        bmm_75: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_780, view_781);  view_780 = view_781 = None
        view_782: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_75, [8, 16, 576, 48]);  bmm_75 = None
        permute_513: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_782, [0, 2, 1, 3]);  view_782 = None
        clone_536: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_513, memory_format = torch.contiguous_format);  permute_513 = None
        view_783: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_536, [8, 576, 768]);  clone_536 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_784: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_783, [4608, 768]);  view_783 = None
        permute_514: "f32[768, 768]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        
        # No stacktrace found for following nodes
        mm_default_144: "f32[4608, 768]" = torch.ops.aten.mm.default(view_784, permute_514);  view_784 = permute_514 = None
        add_tensor_144: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_144, arg32_1);  mm_default_144 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_785: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_144, [8, 576, 768]);  add_tensor_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_393: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg22_1, view_785);  arg22_1 = view_785 = None
        add_355: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_350, mul_393);  add_350 = mul_393 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_538: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_355, memory_format = torch.contiguous_format)
        var_mean_80 = torch.ops.aten.var_mean.correction(clone_538, [2], correction = 0, keepdim = True)
        getitem_168: "f32[8, 576, 1]" = var_mean_80[0]
        getitem_169: "f32[8, 576, 1]" = var_mean_80[1];  var_mean_80 = None
        sub_118: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_538, getitem_169);  clone_538 = getitem_169 = None
        add_356: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
        rsqrt_80: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_356);  add_356 = None
        mul_394: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_80);  sub_118 = rsqrt_80 = None
        mul_395: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_394, arg34_1);  mul_394 = arg34_1 = None
        add_357: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_395, arg35_1);  mul_395 = arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_786: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_357, [4608, 768]);  add_357 = None
        permute_515: "f32[768, 3072]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        
        # No stacktrace found for following nodes
        mm_default_143: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_786, permute_515);  view_786 = permute_515 = None
        add_tensor_143: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_143, arg37_1);  mm_default_143 = arg37_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_787: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_143, [8, 576, 3072]);  add_tensor_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_396: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_787, 0.5)
        mul_397: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_787, 0.7071067811865476);  view_787 = None
        erf_39: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_397);  mul_397 = None
        add_358: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_398: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_396, add_358);  mul_396 = add_358 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_788: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_398, [4608, 3072]);  mul_398 = None
        permute_516: "f32[3072, 768]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        
        # No stacktrace found for following nodes
        mm_default_142: "f32[4608, 768]" = torch.ops.aten.mm.default(view_788, permute_516);  view_788 = permute_516 = None
        add_tensor_142: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_142, arg39_1);  mm_default_142 = arg39_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_789: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_142, [8, 576, 768]);  add_tensor_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_399: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg33_1, view_789);  arg33_1 = view_789 = None
        add_359: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_355, mul_399);  add_355 = mul_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_541: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_359, memory_format = torch.contiguous_format)
        var_mean_81 = torch.ops.aten.var_mean.correction(clone_541, [2], correction = 0, keepdim = True)
        getitem_170: "f32[8, 576, 1]" = var_mean_81[0]
        getitem_171: "f32[8, 576, 1]" = var_mean_81[1];  var_mean_81 = None
        sub_119: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_541, getitem_171);  clone_541 = getitem_171 = None
        add_360: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-06);  getitem_170 = None
        rsqrt_81: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_360);  add_360 = None
        mul_400: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_81);  sub_119 = rsqrt_81 = None
        mul_401: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_400, arg41_1);  mul_400 = arg41_1 = None
        add_361: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_401, arg42_1);  mul_401 = arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_790: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_361, [4608, 768]);  add_361 = None
        permute_517: "f32[768, 2304]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        
        # No stacktrace found for following nodes
        mm_default_141: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_790, permute_517);  view_790 = permute_517 = None
        add_tensor_141: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_141, arg44_1);  mm_default_141 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_791: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_141, [8, 576, 2304]);  add_tensor_141 = None
        view_792: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_791, [8, 576, 3, 16, 48]);  view_791 = None
        permute_518: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_792, [2, 0, 3, 1, 4]);  view_792 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_117: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_518, 0, 0)
        mul_402: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_117, 0.14433756729740643);  select_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_153: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_402, [8, 16, 576, 48]);  mul_402 = None
        clone_542: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_793: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_542, [128, 576, 48]);  clone_542 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_118: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_518, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_519: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_118, [0, 1, 3, 2]);  select_118 = None
        expand_154: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_519, [8, 16, 48, 576]);  permute_519 = None
        clone_543: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_154, memory_format = torch.contiguous_format);  expand_154 = None
        view_794: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_543, [128, 48, 576]);  clone_543 = None
        bmm_76: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_793, view_794);  view_793 = view_794 = None
        view_795: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_76, [8, 16, 576, 576]);  bmm_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_520: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_795, [0, 2, 3, 1]);  view_795 = None
        clone_544: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
        view_796: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_544, [2654208, 16]);  clone_544 = None
        permute_521: "f32[16, 16]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_76: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_796, permute_521);  view_796 = permute_521 = None
        view_797: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_76, [8, 576, 576, 16]);  mm_76 = None
        add_362: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_797, arg46_1);  view_797 = arg46_1 = None
        permute_522: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_362, [0, 3, 1, 2]);  add_362 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_545: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_522, memory_format = torch.contiguous_format);  permute_522 = None
        amax_38: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_545, [-1], True)
        sub_120: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_545, amax_38);  clone_545 = amax_38 = None
        exp_38: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_120);  sub_120 = None
        sum_39: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_38, [-1], True)
        div_38: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_38, sum_39);  exp_38 = sum_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_523: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_38, [0, 2, 3, 1]);  div_38 = None
        clone_546: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
        view_798: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_546, [2654208, 16]);  clone_546 = None
        permute_524: "f32[16, 16]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_77: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_798, permute_524);  view_798 = permute_524 = None
        view_799: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_77, [8, 576, 576, 16]);  mm_77 = None
        add_363: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_799, arg48_1);  view_799 = arg48_1 = None
        permute_525: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_363, [0, 3, 1, 2]);  add_363 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_155: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_525, [8, 16, 576, 576]);  permute_525 = None
        clone_548: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_155, memory_format = torch.contiguous_format);  expand_155 = None
        view_800: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_548, [128, 576, 576]);  clone_548 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_119: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_518, 0, 2);  permute_518 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_156: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_119, [8, 16, 576, 48]);  select_119 = None
        clone_549: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_156, memory_format = torch.contiguous_format);  expand_156 = None
        view_801: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_549, [128, 576, 48]);  clone_549 = None
        bmm_77: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_800, view_801);  view_800 = view_801 = None
        view_802: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_77, [8, 16, 576, 48]);  bmm_77 = None
        permute_526: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None
        clone_550: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_526, memory_format = torch.contiguous_format);  permute_526 = None
        view_803: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_550, [8, 576, 768]);  clone_550 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_804: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_803, [4608, 768]);  view_803 = None
        permute_527: "f32[768, 768]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        
        # No stacktrace found for following nodes
        mm_default_140: "f32[4608, 768]" = torch.ops.aten.mm.default(view_804, permute_527);  view_804 = permute_527 = None
        add_tensor_140: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_140, arg50_1);  mm_default_140 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_805: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_140, [8, 576, 768]);  add_tensor_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_403: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg40_1, view_805);  arg40_1 = view_805 = None
        add_364: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_359, mul_403);  add_359 = mul_403 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_552: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_364, memory_format = torch.contiguous_format)
        var_mean_82 = torch.ops.aten.var_mean.correction(clone_552, [2], correction = 0, keepdim = True)
        getitem_172: "f32[8, 576, 1]" = var_mean_82[0]
        getitem_173: "f32[8, 576, 1]" = var_mean_82[1];  var_mean_82 = None
        sub_121: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_552, getitem_173);  clone_552 = getitem_173 = None
        add_365: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-06);  getitem_172 = None
        rsqrt_82: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_365);  add_365 = None
        mul_404: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_82);  sub_121 = rsqrt_82 = None
        mul_405: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_404, arg52_1);  mul_404 = arg52_1 = None
        add_366: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_405, arg53_1);  mul_405 = arg53_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_806: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_366, [4608, 768]);  add_366 = None
        permute_528: "f32[768, 3072]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        
        # No stacktrace found for following nodes
        mm_default_139: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_806, permute_528);  view_806 = permute_528 = None
        add_tensor_139: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_139, arg55_1);  mm_default_139 = arg55_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_807: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_139, [8, 576, 3072]);  add_tensor_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_406: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_807, 0.5)
        mul_407: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_807, 0.7071067811865476);  view_807 = None
        erf_40: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_407);  mul_407 = None
        add_367: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_408: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_406, add_367);  mul_406 = add_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_808: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_408, [4608, 3072]);  mul_408 = None
        permute_529: "f32[3072, 768]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        
        # No stacktrace found for following nodes
        mm_default_138: "f32[4608, 768]" = torch.ops.aten.mm.default(view_808, permute_529);  view_808 = permute_529 = None
        add_tensor_138: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_138, arg57_1);  mm_default_138 = arg57_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_809: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_138, [8, 576, 768]);  add_tensor_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_409: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg51_1, view_809);  arg51_1 = view_809 = None
        add_368: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_364, mul_409);  add_364 = mul_409 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_555: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_368, memory_format = torch.contiguous_format)
        var_mean_83 = torch.ops.aten.var_mean.correction(clone_555, [2], correction = 0, keepdim = True)
        getitem_174: "f32[8, 576, 1]" = var_mean_83[0]
        getitem_175: "f32[8, 576, 1]" = var_mean_83[1];  var_mean_83 = None
        sub_122: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_555, getitem_175);  clone_555 = getitem_175 = None
        add_369: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
        rsqrt_83: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_369);  add_369 = None
        mul_410: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_83);  sub_122 = rsqrt_83 = None
        mul_411: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_410, arg59_1);  mul_410 = arg59_1 = None
        add_370: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_411, arg60_1);  mul_411 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_810: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_370, [4608, 768]);  add_370 = None
        permute_530: "f32[768, 2304]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        
        # No stacktrace found for following nodes
        mm_default_137: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_810, permute_530);  view_810 = permute_530 = None
        add_tensor_137: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_137, arg62_1);  mm_default_137 = arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_811: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_137, [8, 576, 2304]);  add_tensor_137 = None
        view_812: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_811, [8, 576, 3, 16, 48]);  view_811 = None
        permute_531: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_812, [2, 0, 3, 1, 4]);  view_812 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_120: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_531, 0, 0)
        mul_412: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_120, 0.14433756729740643);  select_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_157: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_412, [8, 16, 576, 48]);  mul_412 = None
        clone_556: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_157, memory_format = torch.contiguous_format);  expand_157 = None
        view_813: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_556, [128, 576, 48]);  clone_556 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_121: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_531, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_532: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_121, [0, 1, 3, 2]);  select_121 = None
        expand_158: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_532, [8, 16, 48, 576]);  permute_532 = None
        clone_557: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_158, memory_format = torch.contiguous_format);  expand_158 = None
        view_814: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_557, [128, 48, 576]);  clone_557 = None
        bmm_78: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_813, view_814);  view_813 = view_814 = None
        view_815: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_78, [8, 16, 576, 576]);  bmm_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_533: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_815, [0, 2, 3, 1]);  view_815 = None
        clone_558: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_533, memory_format = torch.contiguous_format);  permute_533 = None
        view_816: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_558, [2654208, 16]);  clone_558 = None
        permute_534: "f32[16, 16]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_78: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_816, permute_534);  view_816 = permute_534 = None
        view_817: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_78, [8, 576, 576, 16]);  mm_78 = None
        add_371: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_817, arg64_1);  view_817 = arg64_1 = None
        permute_535: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_371, [0, 3, 1, 2]);  add_371 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_559: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_535, memory_format = torch.contiguous_format);  permute_535 = None
        amax_39: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_559, [-1], True)
        sub_123: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_559, amax_39);  clone_559 = amax_39 = None
        exp_39: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_123);  sub_123 = None
        sum_40: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_39, [-1], True)
        div_39: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_39, sum_40);  exp_39 = sum_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_536: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_39, [0, 2, 3, 1]);  div_39 = None
        clone_560: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_536, memory_format = torch.contiguous_format);  permute_536 = None
        view_818: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_560, [2654208, 16]);  clone_560 = None
        permute_537: "f32[16, 16]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_79: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_818, permute_537);  view_818 = permute_537 = None
        view_819: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_79, [8, 576, 576, 16]);  mm_79 = None
        add_372: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_819, arg66_1);  view_819 = arg66_1 = None
        permute_538: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_372, [0, 3, 1, 2]);  add_372 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_159: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_538, [8, 16, 576, 576]);  permute_538 = None
        clone_562: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_159, memory_format = torch.contiguous_format);  expand_159 = None
        view_820: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_562, [128, 576, 576]);  clone_562 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_122: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_531, 0, 2);  permute_531 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_160: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_122, [8, 16, 576, 48]);  select_122 = None
        clone_563: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_160, memory_format = torch.contiguous_format);  expand_160 = None
        view_821: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_563, [128, 576, 48]);  clone_563 = None
        bmm_79: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_820, view_821);  view_820 = view_821 = None
        view_822: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_79, [8, 16, 576, 48]);  bmm_79 = None
        permute_539: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_822, [0, 2, 1, 3]);  view_822 = None
        clone_564: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_539, memory_format = torch.contiguous_format);  permute_539 = None
        view_823: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_564, [8, 576, 768]);  clone_564 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_824: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_823, [4608, 768]);  view_823 = None
        permute_540: "f32[768, 768]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        
        # No stacktrace found for following nodes
        mm_default_136: "f32[4608, 768]" = torch.ops.aten.mm.default(view_824, permute_540);  view_824 = permute_540 = None
        add_tensor_136: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_136, arg68_1);  mm_default_136 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_825: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_136, [8, 576, 768]);  add_tensor_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_413: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg58_1, view_825);  arg58_1 = view_825 = None
        add_373: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_368, mul_413);  add_368 = mul_413 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_566: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_373, memory_format = torch.contiguous_format)
        var_mean_84 = torch.ops.aten.var_mean.correction(clone_566, [2], correction = 0, keepdim = True)
        getitem_176: "f32[8, 576, 1]" = var_mean_84[0]
        getitem_177: "f32[8, 576, 1]" = var_mean_84[1];  var_mean_84 = None
        sub_124: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_566, getitem_177);  clone_566 = getitem_177 = None
        add_374: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
        rsqrt_84: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_374);  add_374 = None
        mul_414: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_84);  sub_124 = rsqrt_84 = None
        mul_415: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_414, arg70_1);  mul_414 = arg70_1 = None
        add_375: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_415, arg71_1);  mul_415 = arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_826: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_375, [4608, 768]);  add_375 = None
        permute_541: "f32[768, 3072]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        
        # No stacktrace found for following nodes
        mm_default_135: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_826, permute_541);  view_826 = permute_541 = None
        add_tensor_135: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_135, arg73_1);  mm_default_135 = arg73_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_827: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_135, [8, 576, 3072]);  add_tensor_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_416: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_827, 0.5)
        mul_417: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_827, 0.7071067811865476);  view_827 = None
        erf_41: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_417);  mul_417 = None
        add_376: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_418: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_416, add_376);  mul_416 = add_376 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_828: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_418, [4608, 3072]);  mul_418 = None
        permute_542: "f32[3072, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        
        # No stacktrace found for following nodes
        mm_default_134: "f32[4608, 768]" = torch.ops.aten.mm.default(view_828, permute_542);  view_828 = permute_542 = None
        add_tensor_134: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_134, arg75_1);  mm_default_134 = arg75_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_829: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_134, [8, 576, 768]);  add_tensor_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_419: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg69_1, view_829);  arg69_1 = view_829 = None
        add_377: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_373, mul_419);  add_373 = mul_419 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_569: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_377, memory_format = torch.contiguous_format)
        var_mean_85 = torch.ops.aten.var_mean.correction(clone_569, [2], correction = 0, keepdim = True)
        getitem_178: "f32[8, 576, 1]" = var_mean_85[0]
        getitem_179: "f32[8, 576, 1]" = var_mean_85[1];  var_mean_85 = None
        sub_125: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_569, getitem_179);  clone_569 = getitem_179 = None
        add_378: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-06);  getitem_178 = None
        rsqrt_85: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_378);  add_378 = None
        mul_420: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_85);  sub_125 = rsqrt_85 = None
        mul_421: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_420, arg77_1);  mul_420 = arg77_1 = None
        add_379: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_421, arg78_1);  mul_421 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_830: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_379, [4608, 768]);  add_379 = None
        permute_543: "f32[768, 2304]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        
        # No stacktrace found for following nodes
        mm_default_133: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_830, permute_543);  view_830 = permute_543 = None
        add_tensor_133: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_133, arg80_1);  mm_default_133 = arg80_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_831: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_133, [8, 576, 2304]);  add_tensor_133 = None
        view_832: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_831, [8, 576, 3, 16, 48]);  view_831 = None
        permute_544: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_832, [2, 0, 3, 1, 4]);  view_832 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_123: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_544, 0, 0)
        mul_422: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_123, 0.14433756729740643);  select_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_161: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_422, [8, 16, 576, 48]);  mul_422 = None
        clone_570: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_833: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_570, [128, 576, 48]);  clone_570 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_124: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_544, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_545: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_124, [0, 1, 3, 2]);  select_124 = None
        expand_162: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_545, [8, 16, 48, 576]);  permute_545 = None
        clone_571: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_162, memory_format = torch.contiguous_format);  expand_162 = None
        view_834: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_571, [128, 48, 576]);  clone_571 = None
        bmm_80: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_833, view_834);  view_833 = view_834 = None
        view_835: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_80, [8, 16, 576, 576]);  bmm_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_546: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_835, [0, 2, 3, 1]);  view_835 = None
        clone_572: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_546, memory_format = torch.contiguous_format);  permute_546 = None
        view_836: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_572, [2654208, 16]);  clone_572 = None
        permute_547: "f32[16, 16]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        mm_80: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_836, permute_547);  view_836 = permute_547 = None
        view_837: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_80, [8, 576, 576, 16]);  mm_80 = None
        add_380: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_837, arg82_1);  view_837 = arg82_1 = None
        permute_548: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_380, [0, 3, 1, 2]);  add_380 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_573: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_548, memory_format = torch.contiguous_format);  permute_548 = None
        amax_40: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_573, [-1], True)
        sub_126: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_573, amax_40);  clone_573 = amax_40 = None
        exp_40: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_126);  sub_126 = None
        sum_41: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_40, [-1], True)
        div_40: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_40, sum_41);  exp_40 = sum_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_549: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_40, [0, 2, 3, 1]);  div_40 = None
        clone_574: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_549, memory_format = torch.contiguous_format);  permute_549 = None
        view_838: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_574, [2654208, 16]);  clone_574 = None
        permute_550: "f32[16, 16]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_81: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_838, permute_550);  view_838 = permute_550 = None
        view_839: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_81, [8, 576, 576, 16]);  mm_81 = None
        add_381: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_839, arg84_1);  view_839 = arg84_1 = None
        permute_551: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_381, [0, 3, 1, 2]);  add_381 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_163: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_551, [8, 16, 576, 576]);  permute_551 = None
        clone_576: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_163, memory_format = torch.contiguous_format);  expand_163 = None
        view_840: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_576, [128, 576, 576]);  clone_576 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_125: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_544, 0, 2);  permute_544 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_164: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_125, [8, 16, 576, 48]);  select_125 = None
        clone_577: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_164, memory_format = torch.contiguous_format);  expand_164 = None
        view_841: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_577, [128, 576, 48]);  clone_577 = None
        bmm_81: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_840, view_841);  view_840 = view_841 = None
        view_842: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_81, [8, 16, 576, 48]);  bmm_81 = None
        permute_552: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_842, [0, 2, 1, 3]);  view_842 = None
        clone_578: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_552, memory_format = torch.contiguous_format);  permute_552 = None
        view_843: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_578, [8, 576, 768]);  clone_578 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_844: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_843, [4608, 768]);  view_843 = None
        permute_553: "f32[768, 768]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        
        # No stacktrace found for following nodes
        mm_default_132: "f32[4608, 768]" = torch.ops.aten.mm.default(view_844, permute_553);  view_844 = permute_553 = None
        add_tensor_132: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_132, arg86_1);  mm_default_132 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_845: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_132, [8, 576, 768]);  add_tensor_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_423: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg76_1, view_845);  arg76_1 = view_845 = None
        add_382: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_377, mul_423);  add_377 = mul_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_580: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_382, memory_format = torch.contiguous_format)
        var_mean_86 = torch.ops.aten.var_mean.correction(clone_580, [2], correction = 0, keepdim = True)
        getitem_180: "f32[8, 576, 1]" = var_mean_86[0]
        getitem_181: "f32[8, 576, 1]" = var_mean_86[1];  var_mean_86 = None
        sub_127: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_580, getitem_181);  clone_580 = getitem_181 = None
        add_383: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-06);  getitem_180 = None
        rsqrt_86: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_383);  add_383 = None
        mul_424: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_86);  sub_127 = rsqrt_86 = None
        mul_425: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_424, arg88_1);  mul_424 = arg88_1 = None
        add_384: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_425, arg89_1);  mul_425 = arg89_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_846: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_384, [4608, 768]);  add_384 = None
        permute_554: "f32[768, 3072]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        
        # No stacktrace found for following nodes
        mm_default_131: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_846, permute_554);  view_846 = permute_554 = None
        add_tensor_131: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_131, arg91_1);  mm_default_131 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_847: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_131, [8, 576, 3072]);  add_tensor_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_426: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_847, 0.5)
        mul_427: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_847, 0.7071067811865476);  view_847 = None
        erf_42: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_427);  mul_427 = None
        add_385: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_428: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_426, add_385);  mul_426 = add_385 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_848: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_428, [4608, 3072]);  mul_428 = None
        permute_555: "f32[3072, 768]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        
        # No stacktrace found for following nodes
        mm_default_130: "f32[4608, 768]" = torch.ops.aten.mm.default(view_848, permute_555);  view_848 = permute_555 = None
        add_tensor_130: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_130, arg93_1);  mm_default_130 = arg93_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_849: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_130, [8, 576, 768]);  add_tensor_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_429: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg87_1, view_849);  arg87_1 = view_849 = None
        add_386: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_382, mul_429);  add_382 = mul_429 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_583: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_386, memory_format = torch.contiguous_format)
        var_mean_87 = torch.ops.aten.var_mean.correction(clone_583, [2], correction = 0, keepdim = True)
        getitem_182: "f32[8, 576, 1]" = var_mean_87[0]
        getitem_183: "f32[8, 576, 1]" = var_mean_87[1];  var_mean_87 = None
        sub_128: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_583, getitem_183);  clone_583 = getitem_183 = None
        add_387: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-06);  getitem_182 = None
        rsqrt_87: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_387);  add_387 = None
        mul_430: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_87);  sub_128 = rsqrt_87 = None
        mul_431: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_430, arg95_1);  mul_430 = arg95_1 = None
        add_388: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_431, arg96_1);  mul_431 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_850: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_388, [4608, 768]);  add_388 = None
        permute_556: "f32[768, 2304]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        
        # No stacktrace found for following nodes
        mm_default_129: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_850, permute_556);  view_850 = permute_556 = None
        add_tensor_129: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_129, arg98_1);  mm_default_129 = arg98_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_851: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_129, [8, 576, 2304]);  add_tensor_129 = None
        view_852: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_851, [8, 576, 3, 16, 48]);  view_851 = None
        permute_557: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_852, [2, 0, 3, 1, 4]);  view_852 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_126: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_557, 0, 0)
        mul_432: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_126, 0.14433756729740643);  select_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_165: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_432, [8, 16, 576, 48]);  mul_432 = None
        clone_584: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_165, memory_format = torch.contiguous_format);  expand_165 = None
        view_853: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_584, [128, 576, 48]);  clone_584 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_127: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_557, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_558: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_127, [0, 1, 3, 2]);  select_127 = None
        expand_166: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_558, [8, 16, 48, 576]);  permute_558 = None
        clone_585: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_166, memory_format = torch.contiguous_format);  expand_166 = None
        view_854: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_585, [128, 48, 576]);  clone_585 = None
        bmm_82: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_853, view_854);  view_853 = view_854 = None
        view_855: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_82, [8, 16, 576, 576]);  bmm_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_559: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_855, [0, 2, 3, 1]);  view_855 = None
        clone_586: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
        view_856: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_586, [2654208, 16]);  clone_586 = None
        permute_560: "f32[16, 16]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_82: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_856, permute_560);  view_856 = permute_560 = None
        view_857: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_82, [8, 576, 576, 16]);  mm_82 = None
        add_389: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_857, arg100_1);  view_857 = arg100_1 = None
        permute_561: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_389, [0, 3, 1, 2]);  add_389 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_587: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_561, memory_format = torch.contiguous_format);  permute_561 = None
        amax_41: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_587, [-1], True)
        sub_129: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_587, amax_41);  clone_587 = amax_41 = None
        exp_41: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_129);  sub_129 = None
        sum_42: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_41, [-1], True)
        div_41: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_41, sum_42);  exp_41 = sum_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_562: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_41, [0, 2, 3, 1]);  div_41 = None
        clone_588: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_562, memory_format = torch.contiguous_format);  permute_562 = None
        view_858: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_588, [2654208, 16]);  clone_588 = None
        permute_563: "f32[16, 16]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_83: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_858, permute_563);  view_858 = permute_563 = None
        view_859: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_83, [8, 576, 576, 16]);  mm_83 = None
        add_390: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_859, arg102_1);  view_859 = arg102_1 = None
        permute_564: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_390, [0, 3, 1, 2]);  add_390 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_167: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_564, [8, 16, 576, 576]);  permute_564 = None
        clone_590: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_167, memory_format = torch.contiguous_format);  expand_167 = None
        view_860: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_590, [128, 576, 576]);  clone_590 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_128: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_557, 0, 2);  permute_557 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_168: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_128, [8, 16, 576, 48]);  select_128 = None
        clone_591: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_168, memory_format = torch.contiguous_format);  expand_168 = None
        view_861: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_591, [128, 576, 48]);  clone_591 = None
        bmm_83: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_860, view_861);  view_860 = view_861 = None
        view_862: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_83, [8, 16, 576, 48]);  bmm_83 = None
        permute_565: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_862, [0, 2, 1, 3]);  view_862 = None
        clone_592: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_565, memory_format = torch.contiguous_format);  permute_565 = None
        view_863: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_592, [8, 576, 768]);  clone_592 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_864: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_863, [4608, 768]);  view_863 = None
        permute_566: "f32[768, 768]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        
        # No stacktrace found for following nodes
        mm_default_128: "f32[4608, 768]" = torch.ops.aten.mm.default(view_864, permute_566);  view_864 = permute_566 = None
        add_tensor_128: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_128, arg104_1);  mm_default_128 = arg104_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_865: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_128, [8, 576, 768]);  add_tensor_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_433: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg94_1, view_865);  arg94_1 = view_865 = None
        add_391: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_386, mul_433);  add_386 = mul_433 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_594: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_391, memory_format = torch.contiguous_format)
        var_mean_88 = torch.ops.aten.var_mean.correction(clone_594, [2], correction = 0, keepdim = True)
        getitem_184: "f32[8, 576, 1]" = var_mean_88[0]
        getitem_185: "f32[8, 576, 1]" = var_mean_88[1];  var_mean_88 = None
        sub_130: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_594, getitem_185);  clone_594 = getitem_185 = None
        add_392: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-06);  getitem_184 = None
        rsqrt_88: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_392);  add_392 = None
        mul_434: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_88);  sub_130 = rsqrt_88 = None
        mul_435: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_434, arg106_1);  mul_434 = arg106_1 = None
        add_393: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_435, arg107_1);  mul_435 = arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_866: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_393, [4608, 768]);  add_393 = None
        permute_567: "f32[768, 3072]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        
        # No stacktrace found for following nodes
        mm_default_127: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_866, permute_567);  view_866 = permute_567 = None
        add_tensor_127: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_127, arg109_1);  mm_default_127 = arg109_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_867: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_127, [8, 576, 3072]);  add_tensor_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_436: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_867, 0.5)
        mul_437: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_867, 0.7071067811865476);  view_867 = None
        erf_43: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_437);  mul_437 = None
        add_394: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_438: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_436, add_394);  mul_436 = add_394 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_868: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_438, [4608, 3072]);  mul_438 = None
        permute_568: "f32[3072, 768]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        
        # No stacktrace found for following nodes
        mm_default_126: "f32[4608, 768]" = torch.ops.aten.mm.default(view_868, permute_568);  view_868 = permute_568 = None
        add_tensor_126: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_126, arg111_1);  mm_default_126 = arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_869: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_126, [8, 576, 768]);  add_tensor_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_439: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg105_1, view_869);  arg105_1 = view_869 = None
        add_395: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_391, mul_439);  add_391 = mul_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_597: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_395, memory_format = torch.contiguous_format)
        var_mean_89 = torch.ops.aten.var_mean.correction(clone_597, [2], correction = 0, keepdim = True)
        getitem_186: "f32[8, 576, 1]" = var_mean_89[0]
        getitem_187: "f32[8, 576, 1]" = var_mean_89[1];  var_mean_89 = None
        sub_131: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_597, getitem_187);  clone_597 = getitem_187 = None
        add_396: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-06);  getitem_186 = None
        rsqrt_89: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_396);  add_396 = None
        mul_440: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_89);  sub_131 = rsqrt_89 = None
        mul_441: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_440, arg113_1);  mul_440 = arg113_1 = None
        add_397: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_441, arg114_1);  mul_441 = arg114_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_870: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_397, [4608, 768]);  add_397 = None
        permute_569: "f32[768, 2304]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        
        # No stacktrace found for following nodes
        mm_default_125: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_870, permute_569);  view_870 = permute_569 = None
        add_tensor_125: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_125, arg116_1);  mm_default_125 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_871: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_125, [8, 576, 2304]);  add_tensor_125 = None
        view_872: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_871, [8, 576, 3, 16, 48]);  view_871 = None
        permute_570: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_872, [2, 0, 3, 1, 4]);  view_872 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_129: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_570, 0, 0)
        mul_442: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_129, 0.14433756729740643);  select_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_169: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_442, [8, 16, 576, 48]);  mul_442 = None
        clone_598: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_873: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_598, [128, 576, 48]);  clone_598 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_130: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_570, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_571: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_130, [0, 1, 3, 2]);  select_130 = None
        expand_170: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_571, [8, 16, 48, 576]);  permute_571 = None
        clone_599: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_170, memory_format = torch.contiguous_format);  expand_170 = None
        view_874: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_599, [128, 48, 576]);  clone_599 = None
        bmm_84: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_873, view_874);  view_873 = view_874 = None
        view_875: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_84, [8, 16, 576, 576]);  bmm_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_572: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_875, [0, 2, 3, 1]);  view_875 = None
        clone_600: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_572, memory_format = torch.contiguous_format);  permute_572 = None
        view_876: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_600, [2654208, 16]);  clone_600 = None
        permute_573: "f32[16, 16]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_84: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_876, permute_573);  view_876 = permute_573 = None
        view_877: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_84, [8, 576, 576, 16]);  mm_84 = None
        add_398: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_877, arg118_1);  view_877 = arg118_1 = None
        permute_574: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_398, [0, 3, 1, 2]);  add_398 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_601: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_574, memory_format = torch.contiguous_format);  permute_574 = None
        amax_42: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_601, [-1], True)
        sub_132: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_601, amax_42);  clone_601 = amax_42 = None
        exp_42: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_132);  sub_132 = None
        sum_43: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_42, [-1], True)
        div_42: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_42, sum_43);  exp_42 = sum_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_575: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_42, [0, 2, 3, 1]);  div_42 = None
        clone_602: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_575, memory_format = torch.contiguous_format);  permute_575 = None
        view_878: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_602, [2654208, 16]);  clone_602 = None
        permute_576: "f32[16, 16]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_85: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_878, permute_576);  view_878 = permute_576 = None
        view_879: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_85, [8, 576, 576, 16]);  mm_85 = None
        add_399: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_879, arg120_1);  view_879 = arg120_1 = None
        permute_577: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_399, [0, 3, 1, 2]);  add_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_171: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_577, [8, 16, 576, 576]);  permute_577 = None
        clone_604: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_171, memory_format = torch.contiguous_format);  expand_171 = None
        view_880: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_604, [128, 576, 576]);  clone_604 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_131: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_570, 0, 2);  permute_570 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_172: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_131, [8, 16, 576, 48]);  select_131 = None
        clone_605: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_172, memory_format = torch.contiguous_format);  expand_172 = None
        view_881: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_605, [128, 576, 48]);  clone_605 = None
        bmm_85: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_880, view_881);  view_880 = view_881 = None
        view_882: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_85, [8, 16, 576, 48]);  bmm_85 = None
        permute_578: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_882, [0, 2, 1, 3]);  view_882 = None
        clone_606: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_578, memory_format = torch.contiguous_format);  permute_578 = None
        view_883: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_606, [8, 576, 768]);  clone_606 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_884: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_883, [4608, 768]);  view_883 = None
        permute_579: "f32[768, 768]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        
        # No stacktrace found for following nodes
        mm_default_124: "f32[4608, 768]" = torch.ops.aten.mm.default(view_884, permute_579);  view_884 = permute_579 = None
        add_tensor_124: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_124, arg122_1);  mm_default_124 = arg122_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_885: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_124, [8, 576, 768]);  add_tensor_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_443: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg112_1, view_885);  arg112_1 = view_885 = None
        add_400: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_395, mul_443);  add_395 = mul_443 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_608: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_400, memory_format = torch.contiguous_format)
        var_mean_90 = torch.ops.aten.var_mean.correction(clone_608, [2], correction = 0, keepdim = True)
        getitem_188: "f32[8, 576, 1]" = var_mean_90[0]
        getitem_189: "f32[8, 576, 1]" = var_mean_90[1];  var_mean_90 = None
        sub_133: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_608, getitem_189);  clone_608 = getitem_189 = None
        add_401: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-06);  getitem_188 = None
        rsqrt_90: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_401);  add_401 = None
        mul_444: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_90);  sub_133 = rsqrt_90 = None
        mul_445: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_444, arg124_1);  mul_444 = arg124_1 = None
        add_402: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_445, arg125_1);  mul_445 = arg125_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_886: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_402, [4608, 768]);  add_402 = None
        permute_580: "f32[768, 3072]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        
        # No stacktrace found for following nodes
        mm_default_123: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_886, permute_580);  view_886 = permute_580 = None
        add_tensor_123: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_123, arg127_1);  mm_default_123 = arg127_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_887: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_123, [8, 576, 3072]);  add_tensor_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_446: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_887, 0.5)
        mul_447: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_887, 0.7071067811865476);  view_887 = None
        erf_44: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_447);  mul_447 = None
        add_403: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_448: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_446, add_403);  mul_446 = add_403 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_888: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_448, [4608, 3072]);  mul_448 = None
        permute_581: "f32[3072, 768]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        
        # No stacktrace found for following nodes
        mm_default_122: "f32[4608, 768]" = torch.ops.aten.mm.default(view_888, permute_581);  view_888 = permute_581 = None
        add_tensor_122: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_122, arg129_1);  mm_default_122 = arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_889: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_122, [8, 576, 768]);  add_tensor_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_449: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg123_1, view_889);  arg123_1 = view_889 = None
        add_404: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_400, mul_449);  add_400 = mul_449 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_611: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_404, memory_format = torch.contiguous_format)
        var_mean_91 = torch.ops.aten.var_mean.correction(clone_611, [2], correction = 0, keepdim = True)
        getitem_190: "f32[8, 576, 1]" = var_mean_91[0]
        getitem_191: "f32[8, 576, 1]" = var_mean_91[1];  var_mean_91 = None
        sub_134: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_611, getitem_191);  clone_611 = getitem_191 = None
        add_405: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-06);  getitem_190 = None
        rsqrt_91: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_405);  add_405 = None
        mul_450: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_91);  sub_134 = rsqrt_91 = None
        mul_451: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_450, arg131_1);  mul_450 = arg131_1 = None
        add_406: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_451, arg132_1);  mul_451 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_890: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_406, [4608, 768]);  add_406 = None
        permute_582: "f32[768, 2304]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        
        # No stacktrace found for following nodes
        mm_default_121: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_890, permute_582);  view_890 = permute_582 = None
        add_tensor_121: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_121, arg134_1);  mm_default_121 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_891: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_121, [8, 576, 2304]);  add_tensor_121 = None
        view_892: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_891, [8, 576, 3, 16, 48]);  view_891 = None
        permute_583: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_892, [2, 0, 3, 1, 4]);  view_892 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_132: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_583, 0, 0)
        mul_452: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_132, 0.14433756729740643);  select_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_173: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_452, [8, 16, 576, 48]);  mul_452 = None
        clone_612: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_173, memory_format = torch.contiguous_format);  expand_173 = None
        view_893: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_612, [128, 576, 48]);  clone_612 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_133: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_583, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_584: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_133, [0, 1, 3, 2]);  select_133 = None
        expand_174: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_584, [8, 16, 48, 576]);  permute_584 = None
        clone_613: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_174, memory_format = torch.contiguous_format);  expand_174 = None
        view_894: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_613, [128, 48, 576]);  clone_613 = None
        bmm_86: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_893, view_894);  view_893 = view_894 = None
        view_895: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_86, [8, 16, 576, 576]);  bmm_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_585: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_895, [0, 2, 3, 1]);  view_895 = None
        clone_614: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_585, memory_format = torch.contiguous_format);  permute_585 = None
        view_896: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_614, [2654208, 16]);  clone_614 = None
        permute_586: "f32[16, 16]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        mm_86: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_896, permute_586);  view_896 = permute_586 = None
        view_897: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_86, [8, 576, 576, 16]);  mm_86 = None
        add_407: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_897, arg136_1);  view_897 = arg136_1 = None
        permute_587: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_407, [0, 3, 1, 2]);  add_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_615: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
        amax_43: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_615, [-1], True)
        sub_135: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_615, amax_43);  clone_615 = amax_43 = None
        exp_43: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_135);  sub_135 = None
        sum_44: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_43, [-1], True)
        div_43: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_43, sum_44);  exp_43 = sum_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_588: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_43, [0, 2, 3, 1]);  div_43 = None
        clone_616: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_588, memory_format = torch.contiguous_format);  permute_588 = None
        view_898: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_616, [2654208, 16]);  clone_616 = None
        permute_589: "f32[16, 16]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_87: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_898, permute_589);  view_898 = permute_589 = None
        view_899: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_87, [8, 576, 576, 16]);  mm_87 = None
        add_408: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_899, arg138_1);  view_899 = arg138_1 = None
        permute_590: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_408, [0, 3, 1, 2]);  add_408 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_175: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_590, [8, 16, 576, 576]);  permute_590 = None
        clone_618: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_175, memory_format = torch.contiguous_format);  expand_175 = None
        view_900: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_618, [128, 576, 576]);  clone_618 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_134: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_583, 0, 2);  permute_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_176: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_134, [8, 16, 576, 48]);  select_134 = None
        clone_619: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_176, memory_format = torch.contiguous_format);  expand_176 = None
        view_901: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_619, [128, 576, 48]);  clone_619 = None
        bmm_87: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_900, view_901);  view_900 = view_901 = None
        view_902: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_87, [8, 16, 576, 48]);  bmm_87 = None
        permute_591: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_902, [0, 2, 1, 3]);  view_902 = None
        clone_620: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_591, memory_format = torch.contiguous_format);  permute_591 = None
        view_903: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_620, [8, 576, 768]);  clone_620 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_904: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_903, [4608, 768]);  view_903 = None
        permute_592: "f32[768, 768]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        
        # No stacktrace found for following nodes
        mm_default_120: "f32[4608, 768]" = torch.ops.aten.mm.default(view_904, permute_592);  view_904 = permute_592 = None
        add_tensor_120: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_120, arg140_1);  mm_default_120 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_905: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_120, [8, 576, 768]);  add_tensor_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_453: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg130_1, view_905);  arg130_1 = view_905 = None
        add_409: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_404, mul_453);  add_404 = mul_453 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_622: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_409, memory_format = torch.contiguous_format)
        var_mean_92 = torch.ops.aten.var_mean.correction(clone_622, [2], correction = 0, keepdim = True)
        getitem_192: "f32[8, 576, 1]" = var_mean_92[0]
        getitem_193: "f32[8, 576, 1]" = var_mean_92[1];  var_mean_92 = None
        sub_136: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_622, getitem_193);  clone_622 = getitem_193 = None
        add_410: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
        rsqrt_92: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_410);  add_410 = None
        mul_454: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_136, rsqrt_92);  sub_136 = rsqrt_92 = None
        mul_455: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_454, arg142_1);  mul_454 = arg142_1 = None
        add_411: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_455, arg143_1);  mul_455 = arg143_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_906: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_411, [4608, 768]);  add_411 = None
        permute_593: "f32[768, 3072]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        
        # No stacktrace found for following nodes
        mm_default_119: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_906, permute_593);  view_906 = permute_593 = None
        add_tensor_119: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_119, arg145_1);  mm_default_119 = arg145_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_907: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_119, [8, 576, 3072]);  add_tensor_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_456: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_907, 0.5)
        mul_457: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_907, 0.7071067811865476);  view_907 = None
        erf_45: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_457);  mul_457 = None
        add_412: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_458: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_456, add_412);  mul_456 = add_412 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_908: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_458, [4608, 3072]);  mul_458 = None
        permute_594: "f32[3072, 768]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        
        # No stacktrace found for following nodes
        mm_default_118: "f32[4608, 768]" = torch.ops.aten.mm.default(view_908, permute_594);  view_908 = permute_594 = None
        add_tensor_118: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_118, arg147_1);  mm_default_118 = arg147_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_909: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_118, [8, 576, 768]);  add_tensor_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_459: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg141_1, view_909);  arg141_1 = view_909 = None
        add_413: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_409, mul_459);  add_409 = mul_459 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_625: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_413, memory_format = torch.contiguous_format)
        var_mean_93 = torch.ops.aten.var_mean.correction(clone_625, [2], correction = 0, keepdim = True)
        getitem_194: "f32[8, 576, 1]" = var_mean_93[0]
        getitem_195: "f32[8, 576, 1]" = var_mean_93[1];  var_mean_93 = None
        sub_137: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_625, getitem_195);  clone_625 = getitem_195 = None
        add_414: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-06);  getitem_194 = None
        rsqrt_93: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_414);  add_414 = None
        mul_460: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_93);  sub_137 = rsqrt_93 = None
        mul_461: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_460, arg149_1);  mul_460 = arg149_1 = None
        add_415: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_461, arg150_1);  mul_461 = arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_910: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_415, [4608, 768]);  add_415 = None
        permute_595: "f32[768, 2304]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        
        # No stacktrace found for following nodes
        mm_default_117: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_910, permute_595);  view_910 = permute_595 = None
        add_tensor_117: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_117, arg152_1);  mm_default_117 = arg152_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_911: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_117, [8, 576, 2304]);  add_tensor_117 = None
        view_912: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_911, [8, 576, 3, 16, 48]);  view_911 = None
        permute_596: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_912, [2, 0, 3, 1, 4]);  view_912 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_135: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_596, 0, 0)
        mul_462: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_135, 0.14433756729740643);  select_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_177: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_462, [8, 16, 576, 48]);  mul_462 = None
        clone_626: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_913: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_626, [128, 576, 48]);  clone_626 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_136: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_596, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_597: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_136, [0, 1, 3, 2]);  select_136 = None
        expand_178: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_597, [8, 16, 48, 576]);  permute_597 = None
        clone_627: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_178, memory_format = torch.contiguous_format);  expand_178 = None
        view_914: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_627, [128, 48, 576]);  clone_627 = None
        bmm_88: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_913, view_914);  view_913 = view_914 = None
        view_915: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_88, [8, 16, 576, 576]);  bmm_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_598: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_915, [0, 2, 3, 1]);  view_915 = None
        clone_628: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_598, memory_format = torch.contiguous_format);  permute_598 = None
        view_916: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_628, [2654208, 16]);  clone_628 = None
        permute_599: "f32[16, 16]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        mm_88: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_916, permute_599);  view_916 = permute_599 = None
        view_917: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_88, [8, 576, 576, 16]);  mm_88 = None
        add_416: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_917, arg154_1);  view_917 = arg154_1 = None
        permute_600: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_416, [0, 3, 1, 2]);  add_416 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_629: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_600, memory_format = torch.contiguous_format);  permute_600 = None
        amax_44: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_629, [-1], True)
        sub_138: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_629, amax_44);  clone_629 = amax_44 = None
        exp_44: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_138);  sub_138 = None
        sum_45: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_44, [-1], True)
        div_44: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_44, sum_45);  exp_44 = sum_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_601: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_44, [0, 2, 3, 1]);  div_44 = None
        clone_630: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_601, memory_format = torch.contiguous_format);  permute_601 = None
        view_918: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_630, [2654208, 16]);  clone_630 = None
        permute_602: "f32[16, 16]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        mm_89: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_918, permute_602);  view_918 = permute_602 = None
        view_919: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_89, [8, 576, 576, 16]);  mm_89 = None
        add_417: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_919, arg156_1);  view_919 = arg156_1 = None
        permute_603: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_417, [0, 3, 1, 2]);  add_417 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_179: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_603, [8, 16, 576, 576]);  permute_603 = None
        clone_632: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_179, memory_format = torch.contiguous_format);  expand_179 = None
        view_920: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_632, [128, 576, 576]);  clone_632 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_137: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_596, 0, 2);  permute_596 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_180: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_137, [8, 16, 576, 48]);  select_137 = None
        clone_633: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_180, memory_format = torch.contiguous_format);  expand_180 = None
        view_921: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_633, [128, 576, 48]);  clone_633 = None
        bmm_89: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_920, view_921);  view_920 = view_921 = None
        view_922: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_89, [8, 16, 576, 48]);  bmm_89 = None
        permute_604: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_922, [0, 2, 1, 3]);  view_922 = None
        clone_634: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_604, memory_format = torch.contiguous_format);  permute_604 = None
        view_923: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_634, [8, 576, 768]);  clone_634 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_924: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_923, [4608, 768]);  view_923 = None
        permute_605: "f32[768, 768]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        
        # No stacktrace found for following nodes
        mm_default_116: "f32[4608, 768]" = torch.ops.aten.mm.default(view_924, permute_605);  view_924 = permute_605 = None
        add_tensor_116: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_116, arg158_1);  mm_default_116 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_925: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_116, [8, 576, 768]);  add_tensor_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_463: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg148_1, view_925);  arg148_1 = view_925 = None
        add_418: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_413, mul_463);  add_413 = mul_463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_636: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_418, memory_format = torch.contiguous_format)
        var_mean_94 = torch.ops.aten.var_mean.correction(clone_636, [2], correction = 0, keepdim = True)
        getitem_196: "f32[8, 576, 1]" = var_mean_94[0]
        getitem_197: "f32[8, 576, 1]" = var_mean_94[1];  var_mean_94 = None
        sub_139: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_636, getitem_197);  clone_636 = getitem_197 = None
        add_419: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-06);  getitem_196 = None
        rsqrt_94: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_419);  add_419 = None
        mul_464: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_139, rsqrt_94);  sub_139 = rsqrt_94 = None
        mul_465: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_464, arg160_1);  mul_464 = arg160_1 = None
        add_420: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_465, arg161_1);  mul_465 = arg161_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_926: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_420, [4608, 768]);  add_420 = None
        permute_606: "f32[768, 3072]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        
        # No stacktrace found for following nodes
        mm_default_115: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_926, permute_606);  view_926 = permute_606 = None
        add_tensor_115: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_115, arg163_1);  mm_default_115 = arg163_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_927: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_115, [8, 576, 3072]);  add_tensor_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_466: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_927, 0.5)
        mul_467: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_927, 0.7071067811865476);  view_927 = None
        erf_46: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_467);  mul_467 = None
        add_421: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_468: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_466, add_421);  mul_466 = add_421 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_928: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_468, [4608, 3072]);  mul_468 = None
        permute_607: "f32[3072, 768]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        
        # No stacktrace found for following nodes
        mm_default_114: "f32[4608, 768]" = torch.ops.aten.mm.default(view_928, permute_607);  view_928 = permute_607 = None
        add_tensor_114: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_114, arg165_1);  mm_default_114 = arg165_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_929: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_114, [8, 576, 768]);  add_tensor_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_469: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg159_1, view_929);  arg159_1 = view_929 = None
        add_422: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_418, mul_469);  add_418 = mul_469 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_639: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_422, memory_format = torch.contiguous_format)
        var_mean_95 = torch.ops.aten.var_mean.correction(clone_639, [2], correction = 0, keepdim = True)
        getitem_198: "f32[8, 576, 1]" = var_mean_95[0]
        getitem_199: "f32[8, 576, 1]" = var_mean_95[1];  var_mean_95 = None
        sub_140: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_639, getitem_199);  clone_639 = getitem_199 = None
        add_423: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-06);  getitem_198 = None
        rsqrt_95: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_423);  add_423 = None
        mul_470: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_95);  sub_140 = rsqrt_95 = None
        mul_471: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_470, arg167_1);  mul_470 = arg167_1 = None
        add_424: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_471, arg168_1);  mul_471 = arg168_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_930: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_424, [4608, 768]);  add_424 = None
        permute_608: "f32[768, 2304]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        
        # No stacktrace found for following nodes
        mm_default_113: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_930, permute_608);  view_930 = permute_608 = None
        add_tensor_113: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_113, arg170_1);  mm_default_113 = arg170_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_931: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_113, [8, 576, 2304]);  add_tensor_113 = None
        view_932: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_931, [8, 576, 3, 16, 48]);  view_931 = None
        permute_609: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_932, [2, 0, 3, 1, 4]);  view_932 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_138: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_609, 0, 0)
        mul_472: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_138, 0.14433756729740643);  select_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_181: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_472, [8, 16, 576, 48]);  mul_472 = None
        clone_640: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_181, memory_format = torch.contiguous_format);  expand_181 = None
        view_933: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_640, [128, 576, 48]);  clone_640 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_139: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_609, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_610: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_139, [0, 1, 3, 2]);  select_139 = None
        expand_182: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_610, [8, 16, 48, 576]);  permute_610 = None
        clone_641: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_182, memory_format = torch.contiguous_format);  expand_182 = None
        view_934: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_641, [128, 48, 576]);  clone_641 = None
        bmm_90: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_933, view_934);  view_933 = view_934 = None
        view_935: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_90, [8, 16, 576, 576]);  bmm_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_611: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_935, [0, 2, 3, 1]);  view_935 = None
        clone_642: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_611, memory_format = torch.contiguous_format);  permute_611 = None
        view_936: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_642, [2654208, 16]);  clone_642 = None
        permute_612: "f32[16, 16]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        mm_90: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_936, permute_612);  view_936 = permute_612 = None
        view_937: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_90, [8, 576, 576, 16]);  mm_90 = None
        add_425: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_937, arg172_1);  view_937 = arg172_1 = None
        permute_613: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_425, [0, 3, 1, 2]);  add_425 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_643: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_613, memory_format = torch.contiguous_format);  permute_613 = None
        amax_45: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_643, [-1], True)
        sub_141: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_643, amax_45);  clone_643 = amax_45 = None
        exp_45: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_141);  sub_141 = None
        sum_46: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_45, [-1], True)
        div_45: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_45, sum_46);  exp_45 = sum_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_614: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_45, [0, 2, 3, 1]);  div_45 = None
        clone_644: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_614, memory_format = torch.contiguous_format);  permute_614 = None
        view_938: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_644, [2654208, 16]);  clone_644 = None
        permute_615: "f32[16, 16]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        mm_91: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_938, permute_615);  view_938 = permute_615 = None
        view_939: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_91, [8, 576, 576, 16]);  mm_91 = None
        add_426: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_939, arg174_1);  view_939 = arg174_1 = None
        permute_616: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_426, [0, 3, 1, 2]);  add_426 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_183: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_616, [8, 16, 576, 576]);  permute_616 = None
        clone_646: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_183, memory_format = torch.contiguous_format);  expand_183 = None
        view_940: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_646, [128, 576, 576]);  clone_646 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_140: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_609, 0, 2);  permute_609 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_184: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_140, [8, 16, 576, 48]);  select_140 = None
        clone_647: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_184, memory_format = torch.contiguous_format);  expand_184 = None
        view_941: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_647, [128, 576, 48]);  clone_647 = None
        bmm_91: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_940, view_941);  view_940 = view_941 = None
        view_942: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_91, [8, 16, 576, 48]);  bmm_91 = None
        permute_617: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_942, [0, 2, 1, 3]);  view_942 = None
        clone_648: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_617, memory_format = torch.contiguous_format);  permute_617 = None
        view_943: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_648, [8, 576, 768]);  clone_648 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_944: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_943, [4608, 768]);  view_943 = None
        permute_618: "f32[768, 768]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        
        # No stacktrace found for following nodes
        mm_default_112: "f32[4608, 768]" = torch.ops.aten.mm.default(view_944, permute_618);  view_944 = permute_618 = None
        add_tensor_112: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_112, arg176_1);  mm_default_112 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_945: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_112, [8, 576, 768]);  add_tensor_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_473: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg166_1, view_945);  arg166_1 = view_945 = None
        add_427: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_422, mul_473);  add_422 = mul_473 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_650: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_427, memory_format = torch.contiguous_format)
        var_mean_96 = torch.ops.aten.var_mean.correction(clone_650, [2], correction = 0, keepdim = True)
        getitem_200: "f32[8, 576, 1]" = var_mean_96[0]
        getitem_201: "f32[8, 576, 1]" = var_mean_96[1];  var_mean_96 = None
        sub_142: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_650, getitem_201);  clone_650 = getitem_201 = None
        add_428: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-06);  getitem_200 = None
        rsqrt_96: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_428);  add_428 = None
        mul_474: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_142, rsqrt_96);  sub_142 = rsqrt_96 = None
        mul_475: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_474, arg178_1);  mul_474 = arg178_1 = None
        add_429: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_475, arg179_1);  mul_475 = arg179_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_946: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_429, [4608, 768]);  add_429 = None
        permute_619: "f32[768, 3072]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        
        # No stacktrace found for following nodes
        mm_default_111: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_946, permute_619);  view_946 = permute_619 = None
        add_tensor_111: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_111, arg181_1);  mm_default_111 = arg181_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_947: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_111, [8, 576, 3072]);  add_tensor_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_476: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_947, 0.5)
        mul_477: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_947, 0.7071067811865476);  view_947 = None
        erf_47: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_477);  mul_477 = None
        add_430: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_478: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_476, add_430);  mul_476 = add_430 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_948: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_478, [4608, 3072]);  mul_478 = None
        permute_620: "f32[3072, 768]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        
        # No stacktrace found for following nodes
        mm_default_110: "f32[4608, 768]" = torch.ops.aten.mm.default(view_948, permute_620);  view_948 = permute_620 = None
        add_tensor_110: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_110, arg183_1);  mm_default_110 = arg183_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_949: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_110, [8, 576, 768]);  add_tensor_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_479: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg177_1, view_949);  arg177_1 = view_949 = None
        add_431: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_427, mul_479);  add_427 = mul_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_653: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_431, memory_format = torch.contiguous_format)
        var_mean_97 = torch.ops.aten.var_mean.correction(clone_653, [2], correction = 0, keepdim = True)
        getitem_202: "f32[8, 576, 1]" = var_mean_97[0]
        getitem_203: "f32[8, 576, 1]" = var_mean_97[1];  var_mean_97 = None
        sub_143: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_653, getitem_203);  clone_653 = getitem_203 = None
        add_432: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_202, 1e-06);  getitem_202 = None
        rsqrt_97: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_432);  add_432 = None
        mul_480: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_143, rsqrt_97);  sub_143 = rsqrt_97 = None
        mul_481: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_480, arg185_1);  mul_480 = arg185_1 = None
        add_433: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_481, arg186_1);  mul_481 = arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_950: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_433, [4608, 768]);  add_433 = None
        permute_621: "f32[768, 2304]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        
        # No stacktrace found for following nodes
        mm_default_109: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_950, permute_621);  view_950 = permute_621 = None
        add_tensor_109: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_109, arg188_1);  mm_default_109 = arg188_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_951: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_109, [8, 576, 2304]);  add_tensor_109 = None
        view_952: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_951, [8, 576, 3, 16, 48]);  view_951 = None
        permute_622: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_952, [2, 0, 3, 1, 4]);  view_952 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_141: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_622, 0, 0)
        mul_482: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_141, 0.14433756729740643);  select_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_185: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_482, [8, 16, 576, 48]);  mul_482 = None
        clone_654: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_953: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_654, [128, 576, 48]);  clone_654 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_142: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_622, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_623: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_142, [0, 1, 3, 2]);  select_142 = None
        expand_186: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_623, [8, 16, 48, 576]);  permute_623 = None
        clone_655: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_186, memory_format = torch.contiguous_format);  expand_186 = None
        view_954: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_655, [128, 48, 576]);  clone_655 = None
        bmm_92: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_953, view_954);  view_953 = view_954 = None
        view_955: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_92, [8, 16, 576, 576]);  bmm_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_624: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_955, [0, 2, 3, 1]);  view_955 = None
        clone_656: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_624, memory_format = torch.contiguous_format);  permute_624 = None
        view_956: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_656, [2654208, 16]);  clone_656 = None
        permute_625: "f32[16, 16]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        mm_92: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_956, permute_625);  view_956 = permute_625 = None
        view_957: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_92, [8, 576, 576, 16]);  mm_92 = None
        add_434: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_957, arg190_1);  view_957 = arg190_1 = None
        permute_626: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_434, [0, 3, 1, 2]);  add_434 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_657: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_626, memory_format = torch.contiguous_format);  permute_626 = None
        amax_46: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_657, [-1], True)
        sub_144: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_657, amax_46);  clone_657 = amax_46 = None
        exp_46: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_144);  sub_144 = None
        sum_47: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_46, [-1], True)
        div_46: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_46, sum_47);  exp_46 = sum_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_627: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_46, [0, 2, 3, 1]);  div_46 = None
        clone_658: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_627, memory_format = torch.contiguous_format);  permute_627 = None
        view_958: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_658, [2654208, 16]);  clone_658 = None
        permute_628: "f32[16, 16]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        mm_93: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_958, permute_628);  view_958 = permute_628 = None
        view_959: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_93, [8, 576, 576, 16]);  mm_93 = None
        add_435: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_959, arg192_1);  view_959 = arg192_1 = None
        permute_629: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_435, [0, 3, 1, 2]);  add_435 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_187: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_629, [8, 16, 576, 576]);  permute_629 = None
        clone_660: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_187, memory_format = torch.contiguous_format);  expand_187 = None
        view_960: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_660, [128, 576, 576]);  clone_660 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_143: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_622, 0, 2);  permute_622 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_188: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_143, [8, 16, 576, 48]);  select_143 = None
        clone_661: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_188, memory_format = torch.contiguous_format);  expand_188 = None
        view_961: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_661, [128, 576, 48]);  clone_661 = None
        bmm_93: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_960, view_961);  view_960 = view_961 = None
        view_962: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_93, [8, 16, 576, 48]);  bmm_93 = None
        permute_630: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_962, [0, 2, 1, 3]);  view_962 = None
        clone_662: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format);  permute_630 = None
        view_963: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_662, [8, 576, 768]);  clone_662 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_964: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_963, [4608, 768]);  view_963 = None
        permute_631: "f32[768, 768]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        
        # No stacktrace found for following nodes
        mm_default_108: "f32[4608, 768]" = torch.ops.aten.mm.default(view_964, permute_631);  view_964 = permute_631 = None
        add_tensor_108: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_108, arg194_1);  mm_default_108 = arg194_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_965: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_108, [8, 576, 768]);  add_tensor_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_483: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg184_1, view_965);  arg184_1 = view_965 = None
        add_436: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_431, mul_483);  add_431 = mul_483 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_664: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_436, memory_format = torch.contiguous_format)
        var_mean_98 = torch.ops.aten.var_mean.correction(clone_664, [2], correction = 0, keepdim = True)
        getitem_204: "f32[8, 576, 1]" = var_mean_98[0]
        getitem_205: "f32[8, 576, 1]" = var_mean_98[1];  var_mean_98 = None
        sub_145: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_664, getitem_205);  clone_664 = getitem_205 = None
        add_437: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_204, 1e-06);  getitem_204 = None
        rsqrt_98: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_437);  add_437 = None
        mul_484: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_98);  sub_145 = rsqrt_98 = None
        mul_485: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_484, arg196_1);  mul_484 = arg196_1 = None
        add_438: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_485, arg197_1);  mul_485 = arg197_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_966: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_438, [4608, 768]);  add_438 = None
        permute_632: "f32[768, 3072]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        
        # No stacktrace found for following nodes
        mm_default_107: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_966, permute_632);  view_966 = permute_632 = None
        add_tensor_107: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_107, arg199_1);  mm_default_107 = arg199_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_967: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_107, [8, 576, 3072]);  add_tensor_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_486: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_967, 0.5)
        mul_487: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_967, 0.7071067811865476);  view_967 = None
        erf_48: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_487);  mul_487 = None
        add_439: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
        mul_488: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_486, add_439);  mul_486 = add_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_968: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_488, [4608, 3072]);  mul_488 = None
        permute_633: "f32[3072, 768]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        
        # No stacktrace found for following nodes
        mm_default_106: "f32[4608, 768]" = torch.ops.aten.mm.default(view_968, permute_633);  view_968 = permute_633 = None
        add_tensor_106: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_106, arg201_1);  mm_default_106 = arg201_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_969: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_106, [8, 576, 768]);  add_tensor_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_489: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg195_1, view_969);  arg195_1 = view_969 = None
        add_440: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_436, mul_489);  add_436 = mul_489 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_667: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_440, memory_format = torch.contiguous_format)
        var_mean_99 = torch.ops.aten.var_mean.correction(clone_667, [2], correction = 0, keepdim = True)
        getitem_206: "f32[8, 576, 1]" = var_mean_99[0]
        getitem_207: "f32[8, 576, 1]" = var_mean_99[1];  var_mean_99 = None
        sub_146: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_667, getitem_207);  clone_667 = getitem_207 = None
        add_441: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-06);  getitem_206 = None
        rsqrt_99: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_441);  add_441 = None
        mul_490: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_146, rsqrt_99);  sub_146 = rsqrt_99 = None
        mul_491: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_490, arg203_1);  mul_490 = arg203_1 = None
        add_442: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_491, arg204_1);  mul_491 = arg204_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_970: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_442, [4608, 768]);  add_442 = None
        permute_634: "f32[768, 2304]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        
        # No stacktrace found for following nodes
        mm_default_105: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_970, permute_634);  view_970 = permute_634 = None
        add_tensor_105: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_105, arg206_1);  mm_default_105 = arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_971: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_105, [8, 576, 2304]);  add_tensor_105 = None
        view_972: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_971, [8, 576, 3, 16, 48]);  view_971 = None
        permute_635: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_972, [2, 0, 3, 1, 4]);  view_972 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_144: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_635, 0, 0)
        mul_492: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_144, 0.14433756729740643);  select_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_189: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_492, [8, 16, 576, 48]);  mul_492 = None
        clone_668: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_189, memory_format = torch.contiguous_format);  expand_189 = None
        view_973: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_668, [128, 576, 48]);  clone_668 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_145: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_635, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_636: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_145, [0, 1, 3, 2]);  select_145 = None
        expand_190: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_636, [8, 16, 48, 576]);  permute_636 = None
        clone_669: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_190, memory_format = torch.contiguous_format);  expand_190 = None
        view_974: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_669, [128, 48, 576]);  clone_669 = None
        bmm_94: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_973, view_974);  view_973 = view_974 = None
        view_975: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_94, [8, 16, 576, 576]);  bmm_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_637: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_975, [0, 2, 3, 1]);  view_975 = None
        clone_670: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_637, memory_format = torch.contiguous_format);  permute_637 = None
        view_976: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_670, [2654208, 16]);  clone_670 = None
        permute_638: "f32[16, 16]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        mm_94: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_976, permute_638);  view_976 = permute_638 = None
        view_977: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_94, [8, 576, 576, 16]);  mm_94 = None
        add_443: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_977, arg208_1);  view_977 = arg208_1 = None
        permute_639: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_443, [0, 3, 1, 2]);  add_443 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_671: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_639, memory_format = torch.contiguous_format);  permute_639 = None
        amax_47: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_671, [-1], True)
        sub_147: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_671, amax_47);  clone_671 = amax_47 = None
        exp_47: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_147);  sub_147 = None
        sum_48: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_47, [-1], True)
        div_47: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_47, sum_48);  exp_47 = sum_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_640: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_47, [0, 2, 3, 1]);  div_47 = None
        clone_672: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_640, memory_format = torch.contiguous_format);  permute_640 = None
        view_978: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_672, [2654208, 16]);  clone_672 = None
        permute_641: "f32[16, 16]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        mm_95: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_978, permute_641);  view_978 = permute_641 = None
        view_979: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_95, [8, 576, 576, 16]);  mm_95 = None
        add_444: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_979, arg210_1);  view_979 = arg210_1 = None
        permute_642: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_444, [0, 3, 1, 2]);  add_444 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_191: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_642, [8, 16, 576, 576]);  permute_642 = None
        clone_674: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_191, memory_format = torch.contiguous_format);  expand_191 = None
        view_980: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_674, [128, 576, 576]);  clone_674 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_146: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_635, 0, 2);  permute_635 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_192: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_146, [8, 16, 576, 48]);  select_146 = None
        clone_675: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_192, memory_format = torch.contiguous_format);  expand_192 = None
        view_981: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_675, [128, 576, 48]);  clone_675 = None
        bmm_95: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_980, view_981);  view_980 = view_981 = None
        view_982: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_95, [8, 16, 576, 48]);  bmm_95 = None
        permute_643: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_982, [0, 2, 1, 3]);  view_982 = None
        clone_676: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_643, memory_format = torch.contiguous_format);  permute_643 = None
        view_983: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_676, [8, 576, 768]);  clone_676 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_984: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_983, [4608, 768]);  view_983 = None
        permute_644: "f32[768, 768]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        
        # No stacktrace found for following nodes
        mm_default_104: "f32[4608, 768]" = torch.ops.aten.mm.default(view_984, permute_644);  view_984 = permute_644 = None
        add_tensor_104: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_104, arg212_1);  mm_default_104 = arg212_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_985: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_104, [8, 576, 768]);  add_tensor_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_493: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg202_1, view_985);  arg202_1 = view_985 = None
        add_445: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_440, mul_493);  add_440 = mul_493 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_678: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_445, memory_format = torch.contiguous_format)
        var_mean_100 = torch.ops.aten.var_mean.correction(clone_678, [2], correction = 0, keepdim = True)
        getitem_208: "f32[8, 576, 1]" = var_mean_100[0]
        getitem_209: "f32[8, 576, 1]" = var_mean_100[1];  var_mean_100 = None
        sub_148: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_678, getitem_209);  clone_678 = getitem_209 = None
        add_446: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-06);  getitem_208 = None
        rsqrt_100: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_446);  add_446 = None
        mul_494: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_148, rsqrt_100);  sub_148 = rsqrt_100 = None
        mul_495: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_494, arg214_1);  mul_494 = arg214_1 = None
        add_447: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_495, arg215_1);  mul_495 = arg215_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_986: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_447, [4608, 768]);  add_447 = None
        permute_645: "f32[768, 3072]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        
        # No stacktrace found for following nodes
        mm_default_103: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_986, permute_645);  view_986 = permute_645 = None
        add_tensor_103: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_103, arg217_1);  mm_default_103 = arg217_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_987: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_103, [8, 576, 3072]);  add_tensor_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_496: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_987, 0.5)
        mul_497: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_987, 0.7071067811865476);  view_987 = None
        erf_49: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_497);  mul_497 = None
        add_448: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
        mul_498: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_496, add_448);  mul_496 = add_448 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_988: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_498, [4608, 3072]);  mul_498 = None
        permute_646: "f32[3072, 768]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        
        # No stacktrace found for following nodes
        mm_default_102: "f32[4608, 768]" = torch.ops.aten.mm.default(view_988, permute_646);  view_988 = permute_646 = None
        add_tensor_102: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_102, arg219_1);  mm_default_102 = arg219_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_989: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_102, [8, 576, 768]);  add_tensor_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_499: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg213_1, view_989);  arg213_1 = view_989 = None
        add_449: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_445, mul_499);  add_445 = mul_499 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_681: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_449, memory_format = torch.contiguous_format)
        var_mean_101 = torch.ops.aten.var_mean.correction(clone_681, [2], correction = 0, keepdim = True)
        getitem_210: "f32[8, 576, 1]" = var_mean_101[0]
        getitem_211: "f32[8, 576, 1]" = var_mean_101[1];  var_mean_101 = None
        sub_149: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_681, getitem_211);  clone_681 = getitem_211 = None
        add_450: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-06);  getitem_210 = None
        rsqrt_101: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_450);  add_450 = None
        mul_500: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_149, rsqrt_101);  sub_149 = rsqrt_101 = None
        mul_501: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_500, arg221_1);  mul_500 = arg221_1 = None
        add_451: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_501, arg222_1);  mul_501 = arg222_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_990: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_451, [4608, 768]);  add_451 = None
        permute_647: "f32[768, 2304]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        
        # No stacktrace found for following nodes
        mm_default_101: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_990, permute_647);  view_990 = permute_647 = None
        add_tensor_101: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_101, arg224_1);  mm_default_101 = arg224_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_991: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_101, [8, 576, 2304]);  add_tensor_101 = None
        view_992: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_991, [8, 576, 3, 16, 48]);  view_991 = None
        permute_648: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_992, [2, 0, 3, 1, 4]);  view_992 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_147: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_648, 0, 0)
        mul_502: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_147, 0.14433756729740643);  select_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_193: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_502, [8, 16, 576, 48]);  mul_502 = None
        clone_682: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_193, memory_format = torch.contiguous_format);  expand_193 = None
        view_993: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_682, [128, 576, 48]);  clone_682 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_148: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_648, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_649: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_148, [0, 1, 3, 2]);  select_148 = None
        expand_194: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_649, [8, 16, 48, 576]);  permute_649 = None
        clone_683: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_194, memory_format = torch.contiguous_format);  expand_194 = None
        view_994: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_683, [128, 48, 576]);  clone_683 = None
        bmm_96: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_993, view_994);  view_993 = view_994 = None
        view_995: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_96, [8, 16, 576, 576]);  bmm_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_650: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_995, [0, 2, 3, 1]);  view_995 = None
        clone_684: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
        view_996: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_684, [2654208, 16]);  clone_684 = None
        permute_651: "f32[16, 16]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        mm_96: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_996, permute_651);  view_996 = permute_651 = None
        view_997: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_96, [8, 576, 576, 16]);  mm_96 = None
        add_452: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_997, arg226_1);  view_997 = arg226_1 = None
        permute_652: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_452, [0, 3, 1, 2]);  add_452 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_685: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_652, memory_format = torch.contiguous_format);  permute_652 = None
        amax_48: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_685, [-1], True)
        sub_150: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_685, amax_48);  clone_685 = amax_48 = None
        exp_48: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_150);  sub_150 = None
        sum_49: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_48, [-1], True)
        div_48: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_48, sum_49);  exp_48 = sum_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_653: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_48, [0, 2, 3, 1]);  div_48 = None
        clone_686: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_653, memory_format = torch.contiguous_format);  permute_653 = None
        view_998: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_686, [2654208, 16]);  clone_686 = None
        permute_654: "f32[16, 16]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        mm_97: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_998, permute_654);  view_998 = permute_654 = None
        view_999: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_97, [8, 576, 576, 16]);  mm_97 = None
        add_453: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_999, arg228_1);  view_999 = arg228_1 = None
        permute_655: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_453, [0, 3, 1, 2]);  add_453 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_195: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_655, [8, 16, 576, 576]);  permute_655 = None
        clone_688: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_195, memory_format = torch.contiguous_format);  expand_195 = None
        view_1000: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_688, [128, 576, 576]);  clone_688 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_149: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_648, 0, 2);  permute_648 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_196: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_149, [8, 16, 576, 48]);  select_149 = None
        clone_689: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_196, memory_format = torch.contiguous_format);  expand_196 = None
        view_1001: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_689, [128, 576, 48]);  clone_689 = None
        bmm_97: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1000, view_1001);  view_1000 = view_1001 = None
        view_1002: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_97, [8, 16, 576, 48]);  bmm_97 = None
        permute_656: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1002, [0, 2, 1, 3]);  view_1002 = None
        clone_690: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_656, memory_format = torch.contiguous_format);  permute_656 = None
        view_1003: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_690, [8, 576, 768]);  clone_690 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1004: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1003, [4608, 768]);  view_1003 = None
        permute_657: "f32[768, 768]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        
        # No stacktrace found for following nodes
        mm_default_100: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1004, permute_657);  view_1004 = permute_657 = None
        add_tensor_100: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_100, arg230_1);  mm_default_100 = arg230_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1005: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_100, [8, 576, 768]);  add_tensor_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_503: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg220_1, view_1005);  arg220_1 = view_1005 = None
        add_454: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_449, mul_503);  add_449 = mul_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_692: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_454, memory_format = torch.contiguous_format)
        var_mean_102 = torch.ops.aten.var_mean.correction(clone_692, [2], correction = 0, keepdim = True)
        getitem_212: "f32[8, 576, 1]" = var_mean_102[0]
        getitem_213: "f32[8, 576, 1]" = var_mean_102[1];  var_mean_102 = None
        sub_151: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_692, getitem_213);  clone_692 = getitem_213 = None
        add_455: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-06);  getitem_212 = None
        rsqrt_102: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_455);  add_455 = None
        mul_504: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_151, rsqrt_102);  sub_151 = rsqrt_102 = None
        mul_505: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_504, arg232_1);  mul_504 = arg232_1 = None
        add_456: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_505, arg233_1);  mul_505 = arg233_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1006: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_456, [4608, 768]);  add_456 = None
        permute_658: "f32[768, 3072]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        
        # No stacktrace found for following nodes
        mm_default_99: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1006, permute_658);  view_1006 = permute_658 = None
        add_tensor_99: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_99, arg235_1);  mm_default_99 = arg235_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1007: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_99, [8, 576, 3072]);  add_tensor_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_506: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1007, 0.5)
        mul_507: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1007, 0.7071067811865476);  view_1007 = None
        erf_50: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_507);  mul_507 = None
        add_457: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
        mul_508: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_506, add_457);  mul_506 = add_457 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1008: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_508, [4608, 3072]);  mul_508 = None
        permute_659: "f32[3072, 768]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        
        # No stacktrace found for following nodes
        mm_default_98: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1008, permute_659);  view_1008 = permute_659 = None
        add_tensor_98: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_98, arg237_1);  mm_default_98 = arg237_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1009: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_98, [8, 576, 768]);  add_tensor_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_509: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg231_1, view_1009);  arg231_1 = view_1009 = None
        add_458: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_454, mul_509);  add_454 = mul_509 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_695: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_458, memory_format = torch.contiguous_format)
        var_mean_103 = torch.ops.aten.var_mean.correction(clone_695, [2], correction = 0, keepdim = True)
        getitem_214: "f32[8, 576, 1]" = var_mean_103[0]
        getitem_215: "f32[8, 576, 1]" = var_mean_103[1];  var_mean_103 = None
        sub_152: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_695, getitem_215);  clone_695 = getitem_215 = None
        add_459: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_214, 1e-06);  getitem_214 = None
        rsqrt_103: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_459);  add_459 = None
        mul_510: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_152, rsqrt_103);  sub_152 = rsqrt_103 = None
        mul_511: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_510, arg239_1);  mul_510 = arg239_1 = None
        add_460: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_511, arg240_1);  mul_511 = arg240_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1010: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_460, [4608, 768]);  add_460 = None
        permute_660: "f32[768, 2304]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        
        # No stacktrace found for following nodes
        mm_default_97: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1010, permute_660);  view_1010 = permute_660 = None
        add_tensor_97: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_97, arg242_1);  mm_default_97 = arg242_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1011: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_97, [8, 576, 2304]);  add_tensor_97 = None
        view_1012: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1011, [8, 576, 3, 16, 48]);  view_1011 = None
        permute_661: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1012, [2, 0, 3, 1, 4]);  view_1012 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_150: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_661, 0, 0)
        mul_512: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_150, 0.14433756729740643);  select_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_197: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_512, [8, 16, 576, 48]);  mul_512 = None
        clone_696: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_197, memory_format = torch.contiguous_format);  expand_197 = None
        view_1013: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_696, [128, 576, 48]);  clone_696 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_151: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_661, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_662: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_151, [0, 1, 3, 2]);  select_151 = None
        expand_198: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_662, [8, 16, 48, 576]);  permute_662 = None
        clone_697: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_198, memory_format = torch.contiguous_format);  expand_198 = None
        view_1014: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_697, [128, 48, 576]);  clone_697 = None
        bmm_98: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1013, view_1014);  view_1013 = view_1014 = None
        view_1015: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_98, [8, 16, 576, 576]);  bmm_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_663: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1015, [0, 2, 3, 1]);  view_1015 = None
        clone_698: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_663, memory_format = torch.contiguous_format);  permute_663 = None
        view_1016: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_698, [2654208, 16]);  clone_698 = None
        permute_664: "f32[16, 16]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        mm_98: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1016, permute_664);  view_1016 = permute_664 = None
        view_1017: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_98, [8, 576, 576, 16]);  mm_98 = None
        add_461: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1017, arg244_1);  view_1017 = arg244_1 = None
        permute_665: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_461, [0, 3, 1, 2]);  add_461 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_699: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_665, memory_format = torch.contiguous_format);  permute_665 = None
        amax_49: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_699, [-1], True)
        sub_153: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_699, amax_49);  clone_699 = amax_49 = None
        exp_49: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_153);  sub_153 = None
        sum_50: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_49, [-1], True)
        div_49: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_49, sum_50);  exp_49 = sum_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_666: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_49, [0, 2, 3, 1]);  div_49 = None
        clone_700: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_666, memory_format = torch.contiguous_format);  permute_666 = None
        view_1018: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_700, [2654208, 16]);  clone_700 = None
        permute_667: "f32[16, 16]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        mm_99: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1018, permute_667);  view_1018 = permute_667 = None
        view_1019: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_99, [8, 576, 576, 16]);  mm_99 = None
        add_462: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1019, arg246_1);  view_1019 = arg246_1 = None
        permute_668: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_462, [0, 3, 1, 2]);  add_462 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_199: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_668, [8, 16, 576, 576]);  permute_668 = None
        clone_702: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_199, memory_format = torch.contiguous_format);  expand_199 = None
        view_1020: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_702, [128, 576, 576]);  clone_702 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_152: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_661, 0, 2);  permute_661 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_200: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_152, [8, 16, 576, 48]);  select_152 = None
        clone_703: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_200, memory_format = torch.contiguous_format);  expand_200 = None
        view_1021: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_703, [128, 576, 48]);  clone_703 = None
        bmm_99: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1020, view_1021);  view_1020 = view_1021 = None
        view_1022: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_99, [8, 16, 576, 48]);  bmm_99 = None
        permute_669: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1022, [0, 2, 1, 3]);  view_1022 = None
        clone_704: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_669, memory_format = torch.contiguous_format);  permute_669 = None
        view_1023: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_704, [8, 576, 768]);  clone_704 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1024: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1023, [4608, 768]);  view_1023 = None
        permute_670: "f32[768, 768]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        
        # No stacktrace found for following nodes
        mm_default_96: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1024, permute_670);  view_1024 = permute_670 = None
        add_tensor_96: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_96, arg248_1);  mm_default_96 = arg248_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1025: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_96, [8, 576, 768]);  add_tensor_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_513: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg238_1, view_1025);  arg238_1 = view_1025 = None
        add_463: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_458, mul_513);  add_458 = mul_513 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_706: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_463, memory_format = torch.contiguous_format)
        var_mean_104 = torch.ops.aten.var_mean.correction(clone_706, [2], correction = 0, keepdim = True)
        getitem_216: "f32[8, 576, 1]" = var_mean_104[0]
        getitem_217: "f32[8, 576, 1]" = var_mean_104[1];  var_mean_104 = None
        sub_154: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_706, getitem_217);  clone_706 = getitem_217 = None
        add_464: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_216, 1e-06);  getitem_216 = None
        rsqrt_104: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_464);  add_464 = None
        mul_514: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_154, rsqrt_104);  sub_154 = rsqrt_104 = None
        mul_515: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_514, arg250_1);  mul_514 = arg250_1 = None
        add_465: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_515, arg251_1);  mul_515 = arg251_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1026: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_465, [4608, 768]);  add_465 = None
        permute_671: "f32[768, 3072]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
        
        # No stacktrace found for following nodes
        mm_default_95: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1026, permute_671);  view_1026 = permute_671 = None
        add_tensor_95: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_95, arg253_1);  mm_default_95 = arg253_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1027: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_95, [8, 576, 3072]);  add_tensor_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_516: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1027, 0.5)
        mul_517: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1027, 0.7071067811865476);  view_1027 = None
        erf_51: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_517);  mul_517 = None
        add_466: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
        mul_518: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_516, add_466);  mul_516 = add_466 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1028: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_518, [4608, 3072]);  mul_518 = None
        permute_672: "f32[3072, 768]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        
        # No stacktrace found for following nodes
        mm_default_94: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1028, permute_672);  view_1028 = permute_672 = None
        add_tensor_94: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_94, arg255_1);  mm_default_94 = arg255_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1029: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_94, [8, 576, 768]);  add_tensor_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_519: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg249_1, view_1029);  arg249_1 = view_1029 = None
        add_467: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_463, mul_519);  add_463 = mul_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_709: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_467, memory_format = torch.contiguous_format)
        var_mean_105 = torch.ops.aten.var_mean.correction(clone_709, [2], correction = 0, keepdim = True)
        getitem_218: "f32[8, 576, 1]" = var_mean_105[0]
        getitem_219: "f32[8, 576, 1]" = var_mean_105[1];  var_mean_105 = None
        sub_155: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_709, getitem_219);  clone_709 = getitem_219 = None
        add_468: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-06);  getitem_218 = None
        rsqrt_105: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_468);  add_468 = None
        mul_520: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_155, rsqrt_105);  sub_155 = rsqrt_105 = None
        mul_521: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_520, arg257_1);  mul_520 = arg257_1 = None
        add_469: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_521, arg258_1);  mul_521 = arg258_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1030: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_469, [4608, 768]);  add_469 = None
        permute_673: "f32[768, 2304]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        
        # No stacktrace found for following nodes
        mm_default_93: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1030, permute_673);  view_1030 = permute_673 = None
        add_tensor_93: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_93, arg260_1);  mm_default_93 = arg260_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1031: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_93, [8, 576, 2304]);  add_tensor_93 = None
        view_1032: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1031, [8, 576, 3, 16, 48]);  view_1031 = None
        permute_674: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1032, [2, 0, 3, 1, 4]);  view_1032 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_153: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_674, 0, 0)
        mul_522: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_153, 0.14433756729740643);  select_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_201: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_522, [8, 16, 576, 48]);  mul_522 = None
        clone_710: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_201, memory_format = torch.contiguous_format);  expand_201 = None
        view_1033: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_710, [128, 576, 48]);  clone_710 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_154: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_674, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_675: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_154, [0, 1, 3, 2]);  select_154 = None
        expand_202: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_675, [8, 16, 48, 576]);  permute_675 = None
        clone_711: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_202, memory_format = torch.contiguous_format);  expand_202 = None
        view_1034: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_711, [128, 48, 576]);  clone_711 = None
        bmm_100: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1033, view_1034);  view_1033 = view_1034 = None
        view_1035: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_100, [8, 16, 576, 576]);  bmm_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_676: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1035, [0, 2, 3, 1]);  view_1035 = None
        clone_712: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_676, memory_format = torch.contiguous_format);  permute_676 = None
        view_1036: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_712, [2654208, 16]);  clone_712 = None
        permute_677: "f32[16, 16]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        mm_100: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1036, permute_677);  view_1036 = permute_677 = None
        view_1037: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_100, [8, 576, 576, 16]);  mm_100 = None
        add_470: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1037, arg262_1);  view_1037 = arg262_1 = None
        permute_678: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_470, [0, 3, 1, 2]);  add_470 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_713: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_678, memory_format = torch.contiguous_format);  permute_678 = None
        amax_50: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_713, [-1], True)
        sub_156: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_713, amax_50);  clone_713 = amax_50 = None
        exp_50: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_156);  sub_156 = None
        sum_51: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_50, [-1], True)
        div_50: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_50, sum_51);  exp_50 = sum_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_679: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_50, [0, 2, 3, 1]);  div_50 = None
        clone_714: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_679, memory_format = torch.contiguous_format);  permute_679 = None
        view_1038: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_714, [2654208, 16]);  clone_714 = None
        permute_680: "f32[16, 16]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        mm_101: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1038, permute_680);  view_1038 = permute_680 = None
        view_1039: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_101, [8, 576, 576, 16]);  mm_101 = None
        add_471: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1039, arg264_1);  view_1039 = arg264_1 = None
        permute_681: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_471, [0, 3, 1, 2]);  add_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_203: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_681, [8, 16, 576, 576]);  permute_681 = None
        clone_716: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_203, memory_format = torch.contiguous_format);  expand_203 = None
        view_1040: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_716, [128, 576, 576]);  clone_716 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_155: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_674, 0, 2);  permute_674 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_204: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_155, [8, 16, 576, 48]);  select_155 = None
        clone_717: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_204, memory_format = torch.contiguous_format);  expand_204 = None
        view_1041: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_717, [128, 576, 48]);  clone_717 = None
        bmm_101: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1040, view_1041);  view_1040 = view_1041 = None
        view_1042: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_101, [8, 16, 576, 48]);  bmm_101 = None
        permute_682: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1042, [0, 2, 1, 3]);  view_1042 = None
        clone_718: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_682, memory_format = torch.contiguous_format);  permute_682 = None
        view_1043: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_718, [8, 576, 768]);  clone_718 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1044: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1043, [4608, 768]);  view_1043 = None
        permute_683: "f32[768, 768]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        
        # No stacktrace found for following nodes
        mm_default_92: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1044, permute_683);  view_1044 = permute_683 = None
        add_tensor_92: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_92, arg266_1);  mm_default_92 = arg266_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1045: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_92, [8, 576, 768]);  add_tensor_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_523: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg256_1, view_1045);  arg256_1 = view_1045 = None
        add_472: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_467, mul_523);  add_467 = mul_523 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_720: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_472, memory_format = torch.contiguous_format)
        var_mean_106 = torch.ops.aten.var_mean.correction(clone_720, [2], correction = 0, keepdim = True)
        getitem_220: "f32[8, 576, 1]" = var_mean_106[0]
        getitem_221: "f32[8, 576, 1]" = var_mean_106[1];  var_mean_106 = None
        sub_157: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_720, getitem_221);  clone_720 = getitem_221 = None
        add_473: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-06);  getitem_220 = None
        rsqrt_106: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_473);  add_473 = None
        mul_524: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_157, rsqrt_106);  sub_157 = rsqrt_106 = None
        mul_525: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_524, arg268_1);  mul_524 = arg268_1 = None
        add_474: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_525, arg269_1);  mul_525 = arg269_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1046: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_474, [4608, 768]);  add_474 = None
        permute_684: "f32[768, 3072]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        
        # No stacktrace found for following nodes
        mm_default_91: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1046, permute_684);  view_1046 = permute_684 = None
        add_tensor_91: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_91, arg271_1);  mm_default_91 = arg271_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1047: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_91, [8, 576, 3072]);  add_tensor_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_526: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1047, 0.5)
        mul_527: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1047, 0.7071067811865476);  view_1047 = None
        erf_52: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_527);  mul_527 = None
        add_475: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
        mul_528: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_526, add_475);  mul_526 = add_475 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1048: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_528, [4608, 3072]);  mul_528 = None
        permute_685: "f32[3072, 768]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        
        # No stacktrace found for following nodes
        mm_default_90: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1048, permute_685);  view_1048 = permute_685 = None
        add_tensor_90: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_90, arg273_1);  mm_default_90 = arg273_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1049: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_90, [8, 576, 768]);  add_tensor_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_529: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg267_1, view_1049);  arg267_1 = view_1049 = None
        add_476: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_472, mul_529);  add_472 = mul_529 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_723: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_476, memory_format = torch.contiguous_format)
        var_mean_107 = torch.ops.aten.var_mean.correction(clone_723, [2], correction = 0, keepdim = True)
        getitem_222: "f32[8, 576, 1]" = var_mean_107[0]
        getitem_223: "f32[8, 576, 1]" = var_mean_107[1];  var_mean_107 = None
        sub_158: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_723, getitem_223);  clone_723 = getitem_223 = None
        add_477: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-06);  getitem_222 = None
        rsqrt_107: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_477);  add_477 = None
        mul_530: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_158, rsqrt_107);  sub_158 = rsqrt_107 = None
        mul_531: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_530, arg275_1);  mul_530 = arg275_1 = None
        add_478: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_531, arg276_1);  mul_531 = arg276_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1050: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_478, [4608, 768]);  add_478 = None
        permute_686: "f32[768, 2304]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        
        # No stacktrace found for following nodes
        mm_default_89: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1050, permute_686);  view_1050 = permute_686 = None
        add_tensor_89: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_89, arg278_1);  mm_default_89 = arg278_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1051: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_89, [8, 576, 2304]);  add_tensor_89 = None
        view_1052: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1051, [8, 576, 3, 16, 48]);  view_1051 = None
        permute_687: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1052, [2, 0, 3, 1, 4]);  view_1052 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_156: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_687, 0, 0)
        mul_532: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_156, 0.14433756729740643);  select_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_205: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_532, [8, 16, 576, 48]);  mul_532 = None
        clone_724: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_205, memory_format = torch.contiguous_format);  expand_205 = None
        view_1053: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_724, [128, 576, 48]);  clone_724 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_157: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_687, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_688: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_157, [0, 1, 3, 2]);  select_157 = None
        expand_206: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_688, [8, 16, 48, 576]);  permute_688 = None
        clone_725: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_206, memory_format = torch.contiguous_format);  expand_206 = None
        view_1054: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_725, [128, 48, 576]);  clone_725 = None
        bmm_102: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1053, view_1054);  view_1053 = view_1054 = None
        view_1055: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_102, [8, 16, 576, 576]);  bmm_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_689: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1055, [0, 2, 3, 1]);  view_1055 = None
        clone_726: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_689, memory_format = torch.contiguous_format);  permute_689 = None
        view_1056: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_726, [2654208, 16]);  clone_726 = None
        permute_690: "f32[16, 16]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        mm_102: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1056, permute_690);  view_1056 = permute_690 = None
        view_1057: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_102, [8, 576, 576, 16]);  mm_102 = None
        add_479: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1057, arg280_1);  view_1057 = arg280_1 = None
        permute_691: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_479, [0, 3, 1, 2]);  add_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_727: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_691, memory_format = torch.contiguous_format);  permute_691 = None
        amax_51: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_727, [-1], True)
        sub_159: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_727, amax_51);  clone_727 = amax_51 = None
        exp_51: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_159);  sub_159 = None
        sum_52: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_51, [-1], True)
        div_51: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_51, sum_52);  exp_51 = sum_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_692: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_51, [0, 2, 3, 1]);  div_51 = None
        clone_728: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_692, memory_format = torch.contiguous_format);  permute_692 = None
        view_1058: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_728, [2654208, 16]);  clone_728 = None
        permute_693: "f32[16, 16]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        mm_103: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1058, permute_693);  view_1058 = permute_693 = None
        view_1059: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_103, [8, 576, 576, 16]);  mm_103 = None
        add_480: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1059, arg282_1);  view_1059 = arg282_1 = None
        permute_694: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_480, [0, 3, 1, 2]);  add_480 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_207: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_694, [8, 16, 576, 576]);  permute_694 = None
        clone_730: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_207, memory_format = torch.contiguous_format);  expand_207 = None
        view_1060: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_730, [128, 576, 576]);  clone_730 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_158: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_687, 0, 2);  permute_687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_208: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_158, [8, 16, 576, 48]);  select_158 = None
        clone_731: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_208, memory_format = torch.contiguous_format);  expand_208 = None
        view_1061: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_731, [128, 576, 48]);  clone_731 = None
        bmm_103: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1060, view_1061);  view_1060 = view_1061 = None
        view_1062: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_103, [8, 16, 576, 48]);  bmm_103 = None
        permute_695: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1062, [0, 2, 1, 3]);  view_1062 = None
        clone_732: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_695, memory_format = torch.contiguous_format);  permute_695 = None
        view_1063: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_732, [8, 576, 768]);  clone_732 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1064: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1063, [4608, 768]);  view_1063 = None
        permute_696: "f32[768, 768]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        
        # No stacktrace found for following nodes
        mm_default_88: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1064, permute_696);  view_1064 = permute_696 = None
        add_tensor_88: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_88, arg284_1);  mm_default_88 = arg284_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1065: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_88, [8, 576, 768]);  add_tensor_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_533: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg274_1, view_1065);  arg274_1 = view_1065 = None
        add_481: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_476, mul_533);  add_476 = mul_533 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_734: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_481, memory_format = torch.contiguous_format)
        var_mean_108 = torch.ops.aten.var_mean.correction(clone_734, [2], correction = 0, keepdim = True)
        getitem_224: "f32[8, 576, 1]" = var_mean_108[0]
        getitem_225: "f32[8, 576, 1]" = var_mean_108[1];  var_mean_108 = None
        sub_160: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_734, getitem_225);  clone_734 = getitem_225 = None
        add_482: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-06);  getitem_224 = None
        rsqrt_108: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_482);  add_482 = None
        mul_534: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_160, rsqrt_108);  sub_160 = rsqrt_108 = None
        mul_535: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_534, arg286_1);  mul_534 = arg286_1 = None
        add_483: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_535, arg287_1);  mul_535 = arg287_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1066: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_483, [4608, 768]);  add_483 = None
        permute_697: "f32[768, 3072]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        
        # No stacktrace found for following nodes
        mm_default_87: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1066, permute_697);  view_1066 = permute_697 = None
        add_tensor_87: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_87, arg289_1);  mm_default_87 = arg289_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1067: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_87, [8, 576, 3072]);  add_tensor_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_536: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1067, 0.5)
        mul_537: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1067, 0.7071067811865476);  view_1067 = None
        erf_53: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_537);  mul_537 = None
        add_484: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
        mul_538: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_536, add_484);  mul_536 = add_484 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1068: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_538, [4608, 3072]);  mul_538 = None
        permute_698: "f32[3072, 768]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        
        # No stacktrace found for following nodes
        mm_default_86: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1068, permute_698);  view_1068 = permute_698 = None
        add_tensor_86: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_86, arg291_1);  mm_default_86 = arg291_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1069: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_86, [8, 576, 768]);  add_tensor_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_539: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg285_1, view_1069);  arg285_1 = view_1069 = None
        add_485: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_481, mul_539);  add_481 = mul_539 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_737: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_485, memory_format = torch.contiguous_format)
        var_mean_109 = torch.ops.aten.var_mean.correction(clone_737, [2], correction = 0, keepdim = True)
        getitem_226: "f32[8, 576, 1]" = var_mean_109[0]
        getitem_227: "f32[8, 576, 1]" = var_mean_109[1];  var_mean_109 = None
        sub_161: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_737, getitem_227);  clone_737 = getitem_227 = None
        add_486: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-06);  getitem_226 = None
        rsqrt_109: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_486);  add_486 = None
        mul_540: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_161, rsqrt_109);  sub_161 = rsqrt_109 = None
        mul_541: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_540, arg293_1);  mul_540 = arg293_1 = None
        add_487: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_541, arg294_1);  mul_541 = arg294_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1070: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_487, [4608, 768]);  add_487 = None
        permute_699: "f32[768, 2304]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        
        # No stacktrace found for following nodes
        mm_default_85: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1070, permute_699);  view_1070 = permute_699 = None
        add_tensor_85: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_85, arg296_1);  mm_default_85 = arg296_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1071: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_85, [8, 576, 2304]);  add_tensor_85 = None
        view_1072: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1071, [8, 576, 3, 16, 48]);  view_1071 = None
        permute_700: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1072, [2, 0, 3, 1, 4]);  view_1072 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_159: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_700, 0, 0)
        mul_542: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_159, 0.14433756729740643);  select_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_209: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_542, [8, 16, 576, 48]);  mul_542 = None
        clone_738: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_209, memory_format = torch.contiguous_format);  expand_209 = None
        view_1073: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_738, [128, 576, 48]);  clone_738 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_160: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_700, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_701: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_160, [0, 1, 3, 2]);  select_160 = None
        expand_210: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_701, [8, 16, 48, 576]);  permute_701 = None
        clone_739: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_210, memory_format = torch.contiguous_format);  expand_210 = None
        view_1074: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_739, [128, 48, 576]);  clone_739 = None
        bmm_104: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1073, view_1074);  view_1073 = view_1074 = None
        view_1075: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_104, [8, 16, 576, 576]);  bmm_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_702: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1075, [0, 2, 3, 1]);  view_1075 = None
        clone_740: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_702, memory_format = torch.contiguous_format);  permute_702 = None
        view_1076: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_740, [2654208, 16]);  clone_740 = None
        permute_703: "f32[16, 16]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        mm_104: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1076, permute_703);  view_1076 = permute_703 = None
        view_1077: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_104, [8, 576, 576, 16]);  mm_104 = None
        add_488: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1077, arg298_1);  view_1077 = arg298_1 = None
        permute_704: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_488, [0, 3, 1, 2]);  add_488 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_741: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_704, memory_format = torch.contiguous_format);  permute_704 = None
        amax_52: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_741, [-1], True)
        sub_162: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_741, amax_52);  clone_741 = amax_52 = None
        exp_52: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_162);  sub_162 = None
        sum_53: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_52, [-1], True)
        div_52: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_52, sum_53);  exp_52 = sum_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_705: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_52, [0, 2, 3, 1]);  div_52 = None
        clone_742: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_705, memory_format = torch.contiguous_format);  permute_705 = None
        view_1078: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_742, [2654208, 16]);  clone_742 = None
        permute_706: "f32[16, 16]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        mm_105: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1078, permute_706);  view_1078 = permute_706 = None
        view_1079: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_105, [8, 576, 576, 16]);  mm_105 = None
        add_489: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1079, arg300_1);  view_1079 = arg300_1 = None
        permute_707: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_489, [0, 3, 1, 2]);  add_489 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_211: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_707, [8, 16, 576, 576]);  permute_707 = None
        clone_744: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_211, memory_format = torch.contiguous_format);  expand_211 = None
        view_1080: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_744, [128, 576, 576]);  clone_744 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_161: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_700, 0, 2);  permute_700 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_212: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_161, [8, 16, 576, 48]);  select_161 = None
        clone_745: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_212, memory_format = torch.contiguous_format);  expand_212 = None
        view_1081: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_745, [128, 576, 48]);  clone_745 = None
        bmm_105: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1080, view_1081);  view_1080 = view_1081 = None
        view_1082: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_105, [8, 16, 576, 48]);  bmm_105 = None
        permute_708: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1082, [0, 2, 1, 3]);  view_1082 = None
        clone_746: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_708, memory_format = torch.contiguous_format);  permute_708 = None
        view_1083: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_746, [8, 576, 768]);  clone_746 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1084: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1083, [4608, 768]);  view_1083 = None
        permute_709: "f32[768, 768]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        
        # No stacktrace found for following nodes
        mm_default_84: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1084, permute_709);  view_1084 = permute_709 = None
        add_tensor_84: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_84, arg302_1);  mm_default_84 = arg302_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1085: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_84, [8, 576, 768]);  add_tensor_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_543: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg292_1, view_1085);  arg292_1 = view_1085 = None
        add_490: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_485, mul_543);  add_485 = mul_543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_748: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_490, memory_format = torch.contiguous_format)
        var_mean_110 = torch.ops.aten.var_mean.correction(clone_748, [2], correction = 0, keepdim = True)
        getitem_228: "f32[8, 576, 1]" = var_mean_110[0]
        getitem_229: "f32[8, 576, 1]" = var_mean_110[1];  var_mean_110 = None
        sub_163: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_748, getitem_229);  clone_748 = getitem_229 = None
        add_491: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_228, 1e-06);  getitem_228 = None
        rsqrt_110: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_491);  add_491 = None
        mul_544: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_163, rsqrt_110);  sub_163 = rsqrt_110 = None
        mul_545: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_544, arg304_1);  mul_544 = arg304_1 = None
        add_492: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_545, arg305_1);  mul_545 = arg305_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1086: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_492, [4608, 768]);  add_492 = None
        permute_710: "f32[768, 3072]" = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
        
        # No stacktrace found for following nodes
        mm_default_83: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1086, permute_710);  view_1086 = permute_710 = None
        add_tensor_83: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_83, arg307_1);  mm_default_83 = arg307_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1087: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_83, [8, 576, 3072]);  add_tensor_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_546: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1087, 0.5)
        mul_547: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1087, 0.7071067811865476);  view_1087 = None
        erf_54: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_547);  mul_547 = None
        add_493: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
        mul_548: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_546, add_493);  mul_546 = add_493 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1088: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_548, [4608, 3072]);  mul_548 = None
        permute_711: "f32[3072, 768]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
        
        # No stacktrace found for following nodes
        mm_default_82: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1088, permute_711);  view_1088 = permute_711 = None
        add_tensor_82: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_82, arg309_1);  mm_default_82 = arg309_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1089: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_82, [8, 576, 768]);  add_tensor_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_549: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg303_1, view_1089);  arg303_1 = view_1089 = None
        add_494: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_490, mul_549);  add_490 = mul_549 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_751: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_494, memory_format = torch.contiguous_format)
        var_mean_111 = torch.ops.aten.var_mean.correction(clone_751, [2], correction = 0, keepdim = True)
        getitem_230: "f32[8, 576, 1]" = var_mean_111[0]
        getitem_231: "f32[8, 576, 1]" = var_mean_111[1];  var_mean_111 = None
        sub_164: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_751, getitem_231);  clone_751 = getitem_231 = None
        add_495: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_230, 1e-06);  getitem_230 = None
        rsqrt_111: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_495);  add_495 = None
        mul_550: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_164, rsqrt_111);  sub_164 = rsqrt_111 = None
        mul_551: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_550, arg311_1);  mul_550 = arg311_1 = None
        add_496: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_551, arg312_1);  mul_551 = arg312_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1090: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_496, [4608, 768]);  add_496 = None
        permute_712: "f32[768, 2304]" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        
        # No stacktrace found for following nodes
        mm_default_81: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1090, permute_712);  view_1090 = permute_712 = None
        add_tensor_81: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_81, arg314_1);  mm_default_81 = arg314_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1091: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_81, [8, 576, 2304]);  add_tensor_81 = None
        view_1092: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1091, [8, 576, 3, 16, 48]);  view_1091 = None
        permute_713: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1092, [2, 0, 3, 1, 4]);  view_1092 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_162: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_713, 0, 0)
        mul_552: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_162, 0.14433756729740643);  select_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_213: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_552, [8, 16, 576, 48]);  mul_552 = None
        clone_752: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_213, memory_format = torch.contiguous_format);  expand_213 = None
        view_1093: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_752, [128, 576, 48]);  clone_752 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_163: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_713, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_714: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_163, [0, 1, 3, 2]);  select_163 = None
        expand_214: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_714, [8, 16, 48, 576]);  permute_714 = None
        clone_753: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_214, memory_format = torch.contiguous_format);  expand_214 = None
        view_1094: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_753, [128, 48, 576]);  clone_753 = None
        bmm_106: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1093, view_1094);  view_1093 = view_1094 = None
        view_1095: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_106, [8, 16, 576, 576]);  bmm_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_715: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1095, [0, 2, 3, 1]);  view_1095 = None
        clone_754: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_715, memory_format = torch.contiguous_format);  permute_715 = None
        view_1096: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_754, [2654208, 16]);  clone_754 = None
        permute_716: "f32[16, 16]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        mm_106: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1096, permute_716);  view_1096 = permute_716 = None
        view_1097: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_106, [8, 576, 576, 16]);  mm_106 = None
        add_497: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1097, arg316_1);  view_1097 = arg316_1 = None
        permute_717: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_497, [0, 3, 1, 2]);  add_497 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_755: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_717, memory_format = torch.contiguous_format);  permute_717 = None
        amax_53: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_755, [-1], True)
        sub_165: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_755, amax_53);  clone_755 = amax_53 = None
        exp_53: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_165);  sub_165 = None
        sum_54: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_53, [-1], True)
        div_53: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_53, sum_54);  exp_53 = sum_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_718: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_53, [0, 2, 3, 1]);  div_53 = None
        clone_756: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_718, memory_format = torch.contiguous_format);  permute_718 = None
        view_1098: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_756, [2654208, 16]);  clone_756 = None
        permute_719: "f32[16, 16]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        mm_107: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1098, permute_719);  view_1098 = permute_719 = None
        view_1099: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_107, [8, 576, 576, 16]);  mm_107 = None
        add_498: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1099, arg318_1);  view_1099 = arg318_1 = None
        permute_720: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_498, [0, 3, 1, 2]);  add_498 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_215: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_720, [8, 16, 576, 576]);  permute_720 = None
        clone_758: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_215, memory_format = torch.contiguous_format);  expand_215 = None
        view_1100: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_758, [128, 576, 576]);  clone_758 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_164: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_713, 0, 2);  permute_713 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_216: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_164, [8, 16, 576, 48]);  select_164 = None
        clone_759: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_216, memory_format = torch.contiguous_format);  expand_216 = None
        view_1101: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_759, [128, 576, 48]);  clone_759 = None
        bmm_107: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1100, view_1101);  view_1100 = view_1101 = None
        view_1102: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_107, [8, 16, 576, 48]);  bmm_107 = None
        permute_721: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1102, [0, 2, 1, 3]);  view_1102 = None
        clone_760: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_721, memory_format = torch.contiguous_format);  permute_721 = None
        view_1103: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_760, [8, 576, 768]);  clone_760 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1104: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1103, [4608, 768]);  view_1103 = None
        permute_722: "f32[768, 768]" = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        
        # No stacktrace found for following nodes
        mm_default_80: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1104, permute_722);  view_1104 = permute_722 = None
        add_tensor_80: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_80, arg320_1);  mm_default_80 = arg320_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1105: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_80, [8, 576, 768]);  add_tensor_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_553: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg310_1, view_1105);  arg310_1 = view_1105 = None
        add_499: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_494, mul_553);  add_494 = mul_553 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_762: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_499, memory_format = torch.contiguous_format)
        var_mean_112 = torch.ops.aten.var_mean.correction(clone_762, [2], correction = 0, keepdim = True)
        getitem_232: "f32[8, 576, 1]" = var_mean_112[0]
        getitem_233: "f32[8, 576, 1]" = var_mean_112[1];  var_mean_112 = None
        sub_166: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_762, getitem_233);  clone_762 = getitem_233 = None
        add_500: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_232, 1e-06);  getitem_232 = None
        rsqrt_112: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_500);  add_500 = None
        mul_554: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_166, rsqrt_112);  sub_166 = rsqrt_112 = None
        mul_555: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_554, arg322_1);  mul_554 = arg322_1 = None
        add_501: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_555, arg323_1);  mul_555 = arg323_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1106: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_501, [4608, 768]);  add_501 = None
        permute_723: "f32[768, 3072]" = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
        
        # No stacktrace found for following nodes
        mm_default_79: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1106, permute_723);  view_1106 = permute_723 = None
        add_tensor_79: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_79, arg325_1);  mm_default_79 = arg325_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1107: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_79, [8, 576, 3072]);  add_tensor_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_556: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1107, 0.5)
        mul_557: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1107, 0.7071067811865476);  view_1107 = None
        erf_55: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_557);  mul_557 = None
        add_502: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
        mul_558: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_556, add_502);  mul_556 = add_502 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1108: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_558, [4608, 3072]);  mul_558 = None
        permute_724: "f32[3072, 768]" = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        
        # No stacktrace found for following nodes
        mm_default_78: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1108, permute_724);  view_1108 = permute_724 = None
        add_tensor_78: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_78, arg327_1);  mm_default_78 = arg327_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1109: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_78, [8, 576, 768]);  add_tensor_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_559: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg321_1, view_1109);  arg321_1 = view_1109 = None
        add_503: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_499, mul_559);  add_499 = mul_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_765: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_503, memory_format = torch.contiguous_format)
        var_mean_113 = torch.ops.aten.var_mean.correction(clone_765, [2], correction = 0, keepdim = True)
        getitem_234: "f32[8, 576, 1]" = var_mean_113[0]
        getitem_235: "f32[8, 576, 1]" = var_mean_113[1];  var_mean_113 = None
        sub_167: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_765, getitem_235);  clone_765 = getitem_235 = None
        add_504: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_234, 1e-06);  getitem_234 = None
        rsqrt_113: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_504);  add_504 = None
        mul_560: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_167, rsqrt_113);  sub_167 = rsqrt_113 = None
        mul_561: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_560, arg329_1);  mul_560 = arg329_1 = None
        add_505: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_561, arg330_1);  mul_561 = arg330_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1110: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_505, [4608, 768]);  add_505 = None
        permute_725: "f32[768, 2304]" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        
        # No stacktrace found for following nodes
        mm_default_77: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1110, permute_725);  view_1110 = permute_725 = None
        add_tensor_77: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_77, arg332_1);  mm_default_77 = arg332_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1111: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_77, [8, 576, 2304]);  add_tensor_77 = None
        view_1112: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1111, [8, 576, 3, 16, 48]);  view_1111 = None
        permute_726: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1112, [2, 0, 3, 1, 4]);  view_1112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_165: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_726, 0, 0)
        mul_562: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_165, 0.14433756729740643);  select_165 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_217: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_562, [8, 16, 576, 48]);  mul_562 = None
        clone_766: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_217, memory_format = torch.contiguous_format);  expand_217 = None
        view_1113: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_766, [128, 576, 48]);  clone_766 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_166: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_726, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_727: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_166, [0, 1, 3, 2]);  select_166 = None
        expand_218: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_727, [8, 16, 48, 576]);  permute_727 = None
        clone_767: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_218, memory_format = torch.contiguous_format);  expand_218 = None
        view_1114: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_767, [128, 48, 576]);  clone_767 = None
        bmm_108: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1113, view_1114);  view_1113 = view_1114 = None
        view_1115: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_108, [8, 16, 576, 576]);  bmm_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_728: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1115, [0, 2, 3, 1]);  view_1115 = None
        clone_768: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_728, memory_format = torch.contiguous_format);  permute_728 = None
        view_1116: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_768, [2654208, 16]);  clone_768 = None
        permute_729: "f32[16, 16]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        mm_108: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1116, permute_729);  view_1116 = permute_729 = None
        view_1117: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_108, [8, 576, 576, 16]);  mm_108 = None
        add_506: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1117, arg334_1);  view_1117 = arg334_1 = None
        permute_730: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_506, [0, 3, 1, 2]);  add_506 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_769: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_730, memory_format = torch.contiguous_format);  permute_730 = None
        amax_54: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_769, [-1], True)
        sub_168: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_769, amax_54);  clone_769 = amax_54 = None
        exp_54: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_168);  sub_168 = None
        sum_55: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_54, [-1], True)
        div_54: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_54, sum_55);  exp_54 = sum_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_731: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_54, [0, 2, 3, 1]);  div_54 = None
        clone_770: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_731, memory_format = torch.contiguous_format);  permute_731 = None
        view_1118: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_770, [2654208, 16]);  clone_770 = None
        permute_732: "f32[16, 16]" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        mm_109: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1118, permute_732);  view_1118 = permute_732 = None
        view_1119: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_109, [8, 576, 576, 16]);  mm_109 = None
        add_507: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1119, arg336_1);  view_1119 = arg336_1 = None
        permute_733: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_507, [0, 3, 1, 2]);  add_507 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_219: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_733, [8, 16, 576, 576]);  permute_733 = None
        clone_772: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_219, memory_format = torch.contiguous_format);  expand_219 = None
        view_1120: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_772, [128, 576, 576]);  clone_772 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_167: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_726, 0, 2);  permute_726 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_220: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_167, [8, 16, 576, 48]);  select_167 = None
        clone_773: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_220, memory_format = torch.contiguous_format);  expand_220 = None
        view_1121: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_773, [128, 576, 48]);  clone_773 = None
        bmm_109: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1120, view_1121);  view_1120 = view_1121 = None
        view_1122: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_109, [8, 16, 576, 48]);  bmm_109 = None
        permute_734: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1122, [0, 2, 1, 3]);  view_1122 = None
        clone_774: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_734, memory_format = torch.contiguous_format);  permute_734 = None
        view_1123: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_774, [8, 576, 768]);  clone_774 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1124: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1123, [4608, 768]);  view_1123 = None
        permute_735: "f32[768, 768]" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        
        # No stacktrace found for following nodes
        mm_default_76: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1124, permute_735);  view_1124 = permute_735 = None
        add_tensor_76: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_76, arg338_1);  mm_default_76 = arg338_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1125: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_76, [8, 576, 768]);  add_tensor_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_563: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg328_1, view_1125);  arg328_1 = view_1125 = None
        add_508: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_503, mul_563);  add_503 = mul_563 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_776: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_508, memory_format = torch.contiguous_format)
        var_mean_114 = torch.ops.aten.var_mean.correction(clone_776, [2], correction = 0, keepdim = True)
        getitem_236: "f32[8, 576, 1]" = var_mean_114[0]
        getitem_237: "f32[8, 576, 1]" = var_mean_114[1];  var_mean_114 = None
        sub_169: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_776, getitem_237);  clone_776 = getitem_237 = None
        add_509: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_236, 1e-06);  getitem_236 = None
        rsqrt_114: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_509);  add_509 = None
        mul_564: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_169, rsqrt_114);  sub_169 = rsqrt_114 = None
        mul_565: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_564, arg340_1);  mul_564 = arg340_1 = None
        add_510: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_565, arg341_1);  mul_565 = arg341_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1126: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_510, [4608, 768]);  add_510 = None
        permute_736: "f32[768, 3072]" = torch.ops.aten.permute.default(arg342_1, [1, 0]);  arg342_1 = None
        
        # No stacktrace found for following nodes
        mm_default_75: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1126, permute_736);  view_1126 = permute_736 = None
        add_tensor_75: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_75, arg343_1);  mm_default_75 = arg343_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1127: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_75, [8, 576, 3072]);  add_tensor_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_566: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1127, 0.5)
        mul_567: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1127, 0.7071067811865476);  view_1127 = None
        erf_56: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_567);  mul_567 = None
        add_511: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
        mul_568: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_566, add_511);  mul_566 = add_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1128: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_568, [4608, 3072]);  mul_568 = None
        permute_737: "f32[3072, 768]" = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
        
        # No stacktrace found for following nodes
        mm_default_74: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1128, permute_737);  view_1128 = permute_737 = None
        add_tensor_74: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_74, arg345_1);  mm_default_74 = arg345_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1129: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_74, [8, 576, 768]);  add_tensor_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_569: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg339_1, view_1129);  arg339_1 = view_1129 = None
        add_512: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_508, mul_569);  add_508 = mul_569 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_779: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_512, memory_format = torch.contiguous_format)
        var_mean_115 = torch.ops.aten.var_mean.correction(clone_779, [2], correction = 0, keepdim = True)
        getitem_238: "f32[8, 576, 1]" = var_mean_115[0]
        getitem_239: "f32[8, 576, 1]" = var_mean_115[1];  var_mean_115 = None
        sub_170: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_779, getitem_239);  clone_779 = getitem_239 = None
        add_513: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_238, 1e-06);  getitem_238 = None
        rsqrt_115: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_513);  add_513 = None
        mul_570: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_170, rsqrt_115);  sub_170 = rsqrt_115 = None
        mul_571: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_570, arg347_1);  mul_570 = arg347_1 = None
        add_514: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_571, arg348_1);  mul_571 = arg348_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1130: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_514, [4608, 768]);  add_514 = None
        permute_738: "f32[768, 2304]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        
        # No stacktrace found for following nodes
        mm_default_73: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1130, permute_738);  view_1130 = permute_738 = None
        add_tensor_73: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_73, arg350_1);  mm_default_73 = arg350_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1131: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_73, [8, 576, 2304]);  add_tensor_73 = None
        view_1132: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1131, [8, 576, 3, 16, 48]);  view_1131 = None
        permute_739: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1132, [2, 0, 3, 1, 4]);  view_1132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_168: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_739, 0, 0)
        mul_572: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_168, 0.14433756729740643);  select_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_221: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_572, [8, 16, 576, 48]);  mul_572 = None
        clone_780: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_221, memory_format = torch.contiguous_format);  expand_221 = None
        view_1133: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_780, [128, 576, 48]);  clone_780 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_169: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_739, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_740: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_169, [0, 1, 3, 2]);  select_169 = None
        expand_222: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_740, [8, 16, 48, 576]);  permute_740 = None
        clone_781: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_222, memory_format = torch.contiguous_format);  expand_222 = None
        view_1134: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_781, [128, 48, 576]);  clone_781 = None
        bmm_110: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1133, view_1134);  view_1133 = view_1134 = None
        view_1135: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_110, [8, 16, 576, 576]);  bmm_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_741: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1135, [0, 2, 3, 1]);  view_1135 = None
        clone_782: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_741, memory_format = torch.contiguous_format);  permute_741 = None
        view_1136: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_782, [2654208, 16]);  clone_782 = None
        permute_742: "f32[16, 16]" = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        mm_110: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1136, permute_742);  view_1136 = permute_742 = None
        view_1137: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_110, [8, 576, 576, 16]);  mm_110 = None
        add_515: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1137, arg352_1);  view_1137 = arg352_1 = None
        permute_743: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_515, [0, 3, 1, 2]);  add_515 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_783: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_743, memory_format = torch.contiguous_format);  permute_743 = None
        amax_55: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_783, [-1], True)
        sub_171: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_783, amax_55);  clone_783 = amax_55 = None
        exp_55: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_171);  sub_171 = None
        sum_56: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_55, [-1], True)
        div_55: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_55, sum_56);  exp_55 = sum_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_744: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_55, [0, 2, 3, 1]);  div_55 = None
        clone_784: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_744, memory_format = torch.contiguous_format);  permute_744 = None
        view_1138: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_784, [2654208, 16]);  clone_784 = None
        permute_745: "f32[16, 16]" = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        mm_111: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1138, permute_745);  view_1138 = permute_745 = None
        view_1139: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_111, [8, 576, 576, 16]);  mm_111 = None
        add_516: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1139, arg354_1);  view_1139 = arg354_1 = None
        permute_746: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_516, [0, 3, 1, 2]);  add_516 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_223: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_746, [8, 16, 576, 576]);  permute_746 = None
        clone_786: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_223, memory_format = torch.contiguous_format);  expand_223 = None
        view_1140: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_786, [128, 576, 576]);  clone_786 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_170: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_739, 0, 2);  permute_739 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_224: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_170, [8, 16, 576, 48]);  select_170 = None
        clone_787: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_224, memory_format = torch.contiguous_format);  expand_224 = None
        view_1141: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_787, [128, 576, 48]);  clone_787 = None
        bmm_111: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1140, view_1141);  view_1140 = view_1141 = None
        view_1142: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_111, [8, 16, 576, 48]);  bmm_111 = None
        permute_747: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1142, [0, 2, 1, 3]);  view_1142 = None
        clone_788: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_747, memory_format = torch.contiguous_format);  permute_747 = None
        view_1143: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_788, [8, 576, 768]);  clone_788 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1144: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1143, [4608, 768]);  view_1143 = None
        permute_748: "f32[768, 768]" = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
        
        # No stacktrace found for following nodes
        mm_default_72: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1144, permute_748);  view_1144 = permute_748 = None
        add_tensor_72: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_72, arg356_1);  mm_default_72 = arg356_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1145: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_72, [8, 576, 768]);  add_tensor_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_573: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg346_1, view_1145);  arg346_1 = view_1145 = None
        add_517: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_512, mul_573);  add_512 = mul_573 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_790: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_517, memory_format = torch.contiguous_format)
        var_mean_116 = torch.ops.aten.var_mean.correction(clone_790, [2], correction = 0, keepdim = True)
        getitem_240: "f32[8, 576, 1]" = var_mean_116[0]
        getitem_241: "f32[8, 576, 1]" = var_mean_116[1];  var_mean_116 = None
        sub_172: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_790, getitem_241);  clone_790 = getitem_241 = None
        add_518: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-06);  getitem_240 = None
        rsqrt_116: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_518);  add_518 = None
        mul_574: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_172, rsqrt_116);  sub_172 = rsqrt_116 = None
        mul_575: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_574, arg358_1);  mul_574 = arg358_1 = None
        add_519: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_575, arg359_1);  mul_575 = arg359_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1146: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_519, [4608, 768]);  add_519 = None
        permute_749: "f32[768, 3072]" = torch.ops.aten.permute.default(arg360_1, [1, 0]);  arg360_1 = None
        
        # No stacktrace found for following nodes
        mm_default_71: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1146, permute_749);  view_1146 = permute_749 = None
        add_tensor_71: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_71, arg361_1);  mm_default_71 = arg361_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1147: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_71, [8, 576, 3072]);  add_tensor_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_576: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1147, 0.5)
        mul_577: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1147, 0.7071067811865476);  view_1147 = None
        erf_57: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_577);  mul_577 = None
        add_520: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
        mul_578: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_576, add_520);  mul_576 = add_520 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1148: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_578, [4608, 3072]);  mul_578 = None
        permute_750: "f32[3072, 768]" = torch.ops.aten.permute.default(arg362_1, [1, 0]);  arg362_1 = None
        
        # No stacktrace found for following nodes
        mm_default_70: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1148, permute_750);  view_1148 = permute_750 = None
        add_tensor_70: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_70, arg363_1);  mm_default_70 = arg363_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1149: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_70, [8, 576, 768]);  add_tensor_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_579: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg357_1, view_1149);  arg357_1 = view_1149 = None
        add_521: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_517, mul_579);  add_517 = mul_579 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_793: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_521, memory_format = torch.contiguous_format)
        var_mean_117 = torch.ops.aten.var_mean.correction(clone_793, [2], correction = 0, keepdim = True)
        getitem_242: "f32[8, 576, 1]" = var_mean_117[0]
        getitem_243: "f32[8, 576, 1]" = var_mean_117[1];  var_mean_117 = None
        sub_173: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_793, getitem_243);  clone_793 = getitem_243 = None
        add_522: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
        rsqrt_117: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_522);  add_522 = None
        mul_580: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_173, rsqrt_117);  sub_173 = rsqrt_117 = None
        mul_581: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_580, arg365_1);  mul_580 = arg365_1 = None
        add_523: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_581, arg366_1);  mul_581 = arg366_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1150: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_523, [4608, 768]);  add_523 = None
        permute_751: "f32[768, 2304]" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        
        # No stacktrace found for following nodes
        mm_default_69: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1150, permute_751);  view_1150 = permute_751 = None
        add_tensor_69: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_69, arg368_1);  mm_default_69 = arg368_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1151: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_69, [8, 576, 2304]);  add_tensor_69 = None
        view_1152: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1151, [8, 576, 3, 16, 48]);  view_1151 = None
        permute_752: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1152, [2, 0, 3, 1, 4]);  view_1152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_171: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_752, 0, 0)
        mul_582: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_171, 0.14433756729740643);  select_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_225: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_582, [8, 16, 576, 48]);  mul_582 = None
        clone_794: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_225, memory_format = torch.contiguous_format);  expand_225 = None
        view_1153: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_794, [128, 576, 48]);  clone_794 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_172: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_752, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_753: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_172, [0, 1, 3, 2]);  select_172 = None
        expand_226: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_753, [8, 16, 48, 576]);  permute_753 = None
        clone_795: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_226, memory_format = torch.contiguous_format);  expand_226 = None
        view_1154: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_795, [128, 48, 576]);  clone_795 = None
        bmm_112: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1153, view_1154);  view_1153 = view_1154 = None
        view_1155: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_112, [8, 16, 576, 576]);  bmm_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_754: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1155, [0, 2, 3, 1]);  view_1155 = None
        clone_796: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
        view_1156: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_796, [2654208, 16]);  clone_796 = None
        permute_755: "f32[16, 16]" = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        mm_112: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1156, permute_755);  view_1156 = permute_755 = None
        view_1157: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_112, [8, 576, 576, 16]);  mm_112 = None
        add_524: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1157, arg370_1);  view_1157 = arg370_1 = None
        permute_756: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_524, [0, 3, 1, 2]);  add_524 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_797: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_756, memory_format = torch.contiguous_format);  permute_756 = None
        amax_56: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_797, [-1], True)
        sub_174: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_797, amax_56);  clone_797 = amax_56 = None
        exp_56: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_174);  sub_174 = None
        sum_57: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_56, [-1], True)
        div_56: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_56, sum_57);  exp_56 = sum_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_757: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_56, [0, 2, 3, 1]);  div_56 = None
        clone_798: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_757, memory_format = torch.contiguous_format);  permute_757 = None
        view_1158: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_798, [2654208, 16]);  clone_798 = None
        permute_758: "f32[16, 16]" = torch.ops.aten.permute.default(arg371_1, [1, 0]);  arg371_1 = None
        mm_113: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1158, permute_758);  view_1158 = permute_758 = None
        view_1159: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_113, [8, 576, 576, 16]);  mm_113 = None
        add_525: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1159, arg372_1);  view_1159 = arg372_1 = None
        permute_759: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_525, [0, 3, 1, 2]);  add_525 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_227: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_759, [8, 16, 576, 576]);  permute_759 = None
        clone_800: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_227, memory_format = torch.contiguous_format);  expand_227 = None
        view_1160: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_800, [128, 576, 576]);  clone_800 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_173: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_752, 0, 2);  permute_752 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_228: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_173, [8, 16, 576, 48]);  select_173 = None
        clone_801: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_228, memory_format = torch.contiguous_format);  expand_228 = None
        view_1161: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_801, [128, 576, 48]);  clone_801 = None
        bmm_113: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1160, view_1161);  view_1160 = view_1161 = None
        view_1162: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_113, [8, 16, 576, 48]);  bmm_113 = None
        permute_760: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1162, [0, 2, 1, 3]);  view_1162 = None
        clone_802: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_760, memory_format = torch.contiguous_format);  permute_760 = None
        view_1163: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_802, [8, 576, 768]);  clone_802 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1164: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1163, [4608, 768]);  view_1163 = None
        permute_761: "f32[768, 768]" = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        
        # No stacktrace found for following nodes
        mm_default_68: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1164, permute_761);  view_1164 = permute_761 = None
        add_tensor_68: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_68, arg374_1);  mm_default_68 = arg374_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1165: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_68, [8, 576, 768]);  add_tensor_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_583: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg364_1, view_1165);  arg364_1 = view_1165 = None
        add_526: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_521, mul_583);  add_521 = mul_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_804: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_526, memory_format = torch.contiguous_format)
        var_mean_118 = torch.ops.aten.var_mean.correction(clone_804, [2], correction = 0, keepdim = True)
        getitem_244: "f32[8, 576, 1]" = var_mean_118[0]
        getitem_245: "f32[8, 576, 1]" = var_mean_118[1];  var_mean_118 = None
        sub_175: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_804, getitem_245);  clone_804 = getitem_245 = None
        add_527: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_244, 1e-06);  getitem_244 = None
        rsqrt_118: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_527);  add_527 = None
        mul_584: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_175, rsqrt_118);  sub_175 = rsqrt_118 = None
        mul_585: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_584, arg376_1);  mul_584 = arg376_1 = None
        add_528: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_585, arg377_1);  mul_585 = arg377_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1166: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_528, [4608, 768]);  add_528 = None
        permute_762: "f32[768, 3072]" = torch.ops.aten.permute.default(arg378_1, [1, 0]);  arg378_1 = None
        
        # No stacktrace found for following nodes
        mm_default_67: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1166, permute_762);  view_1166 = permute_762 = None
        add_tensor_67: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_67, arg379_1);  mm_default_67 = arg379_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1167: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_67, [8, 576, 3072]);  add_tensor_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_586: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1167, 0.5)
        mul_587: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1167, 0.7071067811865476);  view_1167 = None
        erf_58: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_587);  mul_587 = None
        add_529: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
        mul_588: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_586, add_529);  mul_586 = add_529 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1168: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_588, [4608, 3072]);  mul_588 = None
        permute_763: "f32[3072, 768]" = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
        
        # No stacktrace found for following nodes
        mm_default_66: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1168, permute_763);  view_1168 = permute_763 = None
        add_tensor_66: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_66, arg381_1);  mm_default_66 = arg381_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1169: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_66, [8, 576, 768]);  add_tensor_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_589: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg375_1, view_1169);  arg375_1 = view_1169 = None
        add_530: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_526, mul_589);  add_526 = mul_589 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_807: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_530, memory_format = torch.contiguous_format)
        var_mean_119 = torch.ops.aten.var_mean.correction(clone_807, [2], correction = 0, keepdim = True)
        getitem_246: "f32[8, 576, 1]" = var_mean_119[0]
        getitem_247: "f32[8, 576, 1]" = var_mean_119[1];  var_mean_119 = None
        sub_176: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_807, getitem_247);  clone_807 = getitem_247 = None
        add_531: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-06);  getitem_246 = None
        rsqrt_119: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_531);  add_531 = None
        mul_590: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_176, rsqrt_119);  sub_176 = rsqrt_119 = None
        mul_591: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_590, arg383_1);  mul_590 = arg383_1 = None
        add_532: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_591, arg384_1);  mul_591 = arg384_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1170: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_532, [4608, 768]);  add_532 = None
        permute_764: "f32[768, 2304]" = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        
        # No stacktrace found for following nodes
        mm_default_65: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1170, permute_764);  view_1170 = permute_764 = None
        add_tensor_65: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_65, arg386_1);  mm_default_65 = arg386_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1171: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_65, [8, 576, 2304]);  add_tensor_65 = None
        view_1172: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1171, [8, 576, 3, 16, 48]);  view_1171 = None
        permute_765: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1172, [2, 0, 3, 1, 4]);  view_1172 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_174: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_765, 0, 0)
        mul_592: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_174, 0.14433756729740643);  select_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_229: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_592, [8, 16, 576, 48]);  mul_592 = None
        clone_808: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_229, memory_format = torch.contiguous_format);  expand_229 = None
        view_1173: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_808, [128, 576, 48]);  clone_808 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_175: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_765, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_766: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_175, [0, 1, 3, 2]);  select_175 = None
        expand_230: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_766, [8, 16, 48, 576]);  permute_766 = None
        clone_809: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_230, memory_format = torch.contiguous_format);  expand_230 = None
        view_1174: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_809, [128, 48, 576]);  clone_809 = None
        bmm_114: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1173, view_1174);  view_1173 = view_1174 = None
        view_1175: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_114, [8, 16, 576, 576]);  bmm_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_767: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1175, [0, 2, 3, 1]);  view_1175 = None
        clone_810: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_767, memory_format = torch.contiguous_format);  permute_767 = None
        view_1176: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_810, [2654208, 16]);  clone_810 = None
        permute_768: "f32[16, 16]" = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
        mm_114: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1176, permute_768);  view_1176 = permute_768 = None
        view_1177: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_114, [8, 576, 576, 16]);  mm_114 = None
        add_533: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1177, arg388_1);  view_1177 = arg388_1 = None
        permute_769: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_533, [0, 3, 1, 2]);  add_533 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_811: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_769, memory_format = torch.contiguous_format);  permute_769 = None
        amax_57: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_811, [-1], True)
        sub_177: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_811, amax_57);  clone_811 = amax_57 = None
        exp_57: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_177);  sub_177 = None
        sum_58: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_57, [-1], True)
        div_57: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_57, sum_58);  exp_57 = sum_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_770: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_57, [0, 2, 3, 1]);  div_57 = None
        clone_812: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_770, memory_format = torch.contiguous_format);  permute_770 = None
        view_1178: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_812, [2654208, 16]);  clone_812 = None
        permute_771: "f32[16, 16]" = torch.ops.aten.permute.default(arg389_1, [1, 0]);  arg389_1 = None
        mm_115: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1178, permute_771);  view_1178 = permute_771 = None
        view_1179: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_115, [8, 576, 576, 16]);  mm_115 = None
        add_534: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1179, arg390_1);  view_1179 = arg390_1 = None
        permute_772: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_534, [0, 3, 1, 2]);  add_534 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_231: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_772, [8, 16, 576, 576]);  permute_772 = None
        clone_814: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_231, memory_format = torch.contiguous_format);  expand_231 = None
        view_1180: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_814, [128, 576, 576]);  clone_814 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_176: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_765, 0, 2);  permute_765 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_232: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_176, [8, 16, 576, 48]);  select_176 = None
        clone_815: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_232, memory_format = torch.contiguous_format);  expand_232 = None
        view_1181: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_815, [128, 576, 48]);  clone_815 = None
        bmm_115: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1180, view_1181);  view_1180 = view_1181 = None
        view_1182: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_115, [8, 16, 576, 48]);  bmm_115 = None
        permute_773: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1182, [0, 2, 1, 3]);  view_1182 = None
        clone_816: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_773, memory_format = torch.contiguous_format);  permute_773 = None
        view_1183: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_816, [8, 576, 768]);  clone_816 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1184: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1183, [4608, 768]);  view_1183 = None
        permute_774: "f32[768, 768]" = torch.ops.aten.permute.default(arg391_1, [1, 0]);  arg391_1 = None
        
        # No stacktrace found for following nodes
        mm_default_64: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1184, permute_774);  view_1184 = permute_774 = None
        add_tensor_64: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_64, arg392_1);  mm_default_64 = arg392_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1185: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_64, [8, 576, 768]);  add_tensor_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_593: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg382_1, view_1185);  arg382_1 = view_1185 = None
        add_535: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_530, mul_593);  add_530 = mul_593 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_818: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_535, memory_format = torch.contiguous_format)
        var_mean_120 = torch.ops.aten.var_mean.correction(clone_818, [2], correction = 0, keepdim = True)
        getitem_248: "f32[8, 576, 1]" = var_mean_120[0]
        getitem_249: "f32[8, 576, 1]" = var_mean_120[1];  var_mean_120 = None
        sub_178: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_818, getitem_249);  clone_818 = getitem_249 = None
        add_536: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_248, 1e-06);  getitem_248 = None
        rsqrt_120: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_536);  add_536 = None
        mul_594: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_178, rsqrt_120);  sub_178 = rsqrt_120 = None
        mul_595: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_594, arg394_1);  mul_594 = arg394_1 = None
        add_537: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_595, arg395_1);  mul_595 = arg395_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1186: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_537, [4608, 768]);  add_537 = None
        permute_775: "f32[768, 3072]" = torch.ops.aten.permute.default(arg396_1, [1, 0]);  arg396_1 = None
        
        # No stacktrace found for following nodes
        mm_default_63: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1186, permute_775);  view_1186 = permute_775 = None
        add_tensor_63: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_63, arg397_1);  mm_default_63 = arg397_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1187: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_63, [8, 576, 3072]);  add_tensor_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_596: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1187, 0.5)
        mul_597: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1187, 0.7071067811865476);  view_1187 = None
        erf_59: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_597);  mul_597 = None
        add_538: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
        mul_598: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_596, add_538);  mul_596 = add_538 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1188: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_598, [4608, 3072]);  mul_598 = None
        permute_776: "f32[3072, 768]" = torch.ops.aten.permute.default(arg398_1, [1, 0]);  arg398_1 = None
        
        # No stacktrace found for following nodes
        mm_default_62: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1188, permute_776);  view_1188 = permute_776 = None
        add_tensor_62: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_62, arg399_1);  mm_default_62 = arg399_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1189: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_62, [8, 576, 768]);  add_tensor_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_599: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg393_1, view_1189);  arg393_1 = view_1189 = None
        add_539: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_535, mul_599);  add_535 = mul_599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_821: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_539, memory_format = torch.contiguous_format)
        var_mean_121 = torch.ops.aten.var_mean.correction(clone_821, [2], correction = 0, keepdim = True)
        getitem_250: "f32[8, 576, 1]" = var_mean_121[0]
        getitem_251: "f32[8, 576, 1]" = var_mean_121[1];  var_mean_121 = None
        sub_179: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_821, getitem_251);  clone_821 = getitem_251 = None
        add_540: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_250, 1e-06);  getitem_250 = None
        rsqrt_121: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_540);  add_540 = None
        mul_600: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_179, rsqrt_121);  sub_179 = rsqrt_121 = None
        mul_601: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_600, arg401_1);  mul_600 = arg401_1 = None
        add_541: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_601, arg402_1);  mul_601 = arg402_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1190: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_541, [4608, 768]);  add_541 = None
        permute_777: "f32[768, 2304]" = torch.ops.aten.permute.default(arg403_1, [1, 0]);  arg403_1 = None
        
        # No stacktrace found for following nodes
        mm_default_61: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1190, permute_777);  view_1190 = permute_777 = None
        add_tensor_61: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_61, arg404_1);  mm_default_61 = arg404_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1191: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_61, [8, 576, 2304]);  add_tensor_61 = None
        view_1192: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1191, [8, 576, 3, 16, 48]);  view_1191 = None
        permute_778: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1192, [2, 0, 3, 1, 4]);  view_1192 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_177: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_778, 0, 0)
        mul_602: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_177, 0.14433756729740643);  select_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_233: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_602, [8, 16, 576, 48]);  mul_602 = None
        clone_822: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_233, memory_format = torch.contiguous_format);  expand_233 = None
        view_1193: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_822, [128, 576, 48]);  clone_822 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_178: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_778, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_779: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_178, [0, 1, 3, 2]);  select_178 = None
        expand_234: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_779, [8, 16, 48, 576]);  permute_779 = None
        clone_823: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_234, memory_format = torch.contiguous_format);  expand_234 = None
        view_1194: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_823, [128, 48, 576]);  clone_823 = None
        bmm_116: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1193, view_1194);  view_1193 = view_1194 = None
        view_1195: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_116, [8, 16, 576, 576]);  bmm_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_780: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1195, [0, 2, 3, 1]);  view_1195 = None
        clone_824: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_780, memory_format = torch.contiguous_format);  permute_780 = None
        view_1196: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_824, [2654208, 16]);  clone_824 = None
        permute_781: "f32[16, 16]" = torch.ops.aten.permute.default(arg405_1, [1, 0]);  arg405_1 = None
        mm_116: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1196, permute_781);  view_1196 = permute_781 = None
        view_1197: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_116, [8, 576, 576, 16]);  mm_116 = None
        add_542: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1197, arg406_1);  view_1197 = arg406_1 = None
        permute_782: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_542, [0, 3, 1, 2]);  add_542 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_825: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_782, memory_format = torch.contiguous_format);  permute_782 = None
        amax_58: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_825, [-1], True)
        sub_180: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_825, amax_58);  clone_825 = amax_58 = None
        exp_58: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_180);  sub_180 = None
        sum_59: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_58, [-1], True)
        div_58: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_58, sum_59);  exp_58 = sum_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_783: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_58, [0, 2, 3, 1]);  div_58 = None
        clone_826: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_783, memory_format = torch.contiguous_format);  permute_783 = None
        view_1198: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_826, [2654208, 16]);  clone_826 = None
        permute_784: "f32[16, 16]" = torch.ops.aten.permute.default(arg407_1, [1, 0]);  arg407_1 = None
        mm_117: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1198, permute_784);  view_1198 = permute_784 = None
        view_1199: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_117, [8, 576, 576, 16]);  mm_117 = None
        add_543: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1199, arg408_1);  view_1199 = arg408_1 = None
        permute_785: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_543, [0, 3, 1, 2]);  add_543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_235: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_785, [8, 16, 576, 576]);  permute_785 = None
        clone_828: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_235, memory_format = torch.contiguous_format);  expand_235 = None
        view_1200: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_828, [128, 576, 576]);  clone_828 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_179: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_778, 0, 2);  permute_778 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_236: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_179, [8, 16, 576, 48]);  select_179 = None
        clone_829: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_236, memory_format = torch.contiguous_format);  expand_236 = None
        view_1201: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_829, [128, 576, 48]);  clone_829 = None
        bmm_117: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1200, view_1201);  view_1200 = view_1201 = None
        view_1202: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_117, [8, 16, 576, 48]);  bmm_117 = None
        permute_786: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1202, [0, 2, 1, 3]);  view_1202 = None
        clone_830: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_786, memory_format = torch.contiguous_format);  permute_786 = None
        view_1203: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_830, [8, 576, 768]);  clone_830 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1204: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1203, [4608, 768]);  view_1203 = None
        permute_787: "f32[768, 768]" = torch.ops.aten.permute.default(arg409_1, [1, 0]);  arg409_1 = None
        
        # No stacktrace found for following nodes
        mm_default_60: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1204, permute_787);  view_1204 = permute_787 = None
        add_tensor_60: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_60, arg410_1);  mm_default_60 = arg410_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1205: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_60, [8, 576, 768]);  add_tensor_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_603: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg400_1, view_1205);  arg400_1 = view_1205 = None
        add_544: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_539, mul_603);  add_539 = mul_603 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_832: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_544, memory_format = torch.contiguous_format)
        var_mean_122 = torch.ops.aten.var_mean.correction(clone_832, [2], correction = 0, keepdim = True)
        getitem_252: "f32[8, 576, 1]" = var_mean_122[0]
        getitem_253: "f32[8, 576, 1]" = var_mean_122[1];  var_mean_122 = None
        sub_181: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_832, getitem_253);  clone_832 = getitem_253 = None
        add_545: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_252, 1e-06);  getitem_252 = None
        rsqrt_122: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_545);  add_545 = None
        mul_604: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_181, rsqrt_122);  sub_181 = rsqrt_122 = None
        mul_605: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_604, arg412_1);  mul_604 = arg412_1 = None
        add_546: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_605, arg413_1);  mul_605 = arg413_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1206: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_546, [4608, 768]);  add_546 = None
        permute_788: "f32[768, 3072]" = torch.ops.aten.permute.default(arg414_1, [1, 0]);  arg414_1 = None
        
        # No stacktrace found for following nodes
        mm_default_59: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1206, permute_788);  view_1206 = permute_788 = None
        add_tensor_59: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_59, arg415_1);  mm_default_59 = arg415_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1207: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_59, [8, 576, 3072]);  add_tensor_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_606: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1207, 0.5)
        mul_607: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1207, 0.7071067811865476);  view_1207 = None
        erf_60: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_607);  mul_607 = None
        add_547: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_60, 1);  erf_60 = None
        mul_608: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_606, add_547);  mul_606 = add_547 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1208: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_608, [4608, 3072]);  mul_608 = None
        permute_789: "f32[3072, 768]" = torch.ops.aten.permute.default(arg416_1, [1, 0]);  arg416_1 = None
        
        # No stacktrace found for following nodes
        mm_default_58: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1208, permute_789);  view_1208 = permute_789 = None
        add_tensor_58: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_58, arg417_1);  mm_default_58 = arg417_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1209: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_58, [8, 576, 768]);  add_tensor_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_609: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg411_1, view_1209);  arg411_1 = view_1209 = None
        add_548: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_544, mul_609);  add_544 = mul_609 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_835: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_548, memory_format = torch.contiguous_format)
        var_mean_123 = torch.ops.aten.var_mean.correction(clone_835, [2], correction = 0, keepdim = True)
        getitem_254: "f32[8, 576, 1]" = var_mean_123[0]
        getitem_255: "f32[8, 576, 1]" = var_mean_123[1];  var_mean_123 = None
        sub_182: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_835, getitem_255);  clone_835 = getitem_255 = None
        add_549: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_254, 1e-06);  getitem_254 = None
        rsqrt_123: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_549);  add_549 = None
        mul_610: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_182, rsqrt_123);  sub_182 = rsqrt_123 = None
        mul_611: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_610, arg419_1);  mul_610 = arg419_1 = None
        add_550: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_611, arg420_1);  mul_611 = arg420_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1210: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_550, [4608, 768]);  add_550 = None
        permute_790: "f32[768, 2304]" = torch.ops.aten.permute.default(arg421_1, [1, 0]);  arg421_1 = None
        
        # No stacktrace found for following nodes
        mm_default_57: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1210, permute_790);  view_1210 = permute_790 = None
        add_tensor_57: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_57, arg422_1);  mm_default_57 = arg422_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1211: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_57, [8, 576, 2304]);  add_tensor_57 = None
        view_1212: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1211, [8, 576, 3, 16, 48]);  view_1211 = None
        permute_791: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1212, [2, 0, 3, 1, 4]);  view_1212 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_180: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_791, 0, 0)
        mul_612: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_180, 0.14433756729740643);  select_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_237: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_612, [8, 16, 576, 48]);  mul_612 = None
        clone_836: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_237, memory_format = torch.contiguous_format);  expand_237 = None
        view_1213: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_836, [128, 576, 48]);  clone_836 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_181: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_791, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_792: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_181, [0, 1, 3, 2]);  select_181 = None
        expand_238: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_792, [8, 16, 48, 576]);  permute_792 = None
        clone_837: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_238, memory_format = torch.contiguous_format);  expand_238 = None
        view_1214: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_837, [128, 48, 576]);  clone_837 = None
        bmm_118: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1213, view_1214);  view_1213 = view_1214 = None
        view_1215: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_118, [8, 16, 576, 576]);  bmm_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_793: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1215, [0, 2, 3, 1]);  view_1215 = None
        clone_838: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_793, memory_format = torch.contiguous_format);  permute_793 = None
        view_1216: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_838, [2654208, 16]);  clone_838 = None
        permute_794: "f32[16, 16]" = torch.ops.aten.permute.default(arg423_1, [1, 0]);  arg423_1 = None
        mm_118: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1216, permute_794);  view_1216 = permute_794 = None
        view_1217: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_118, [8, 576, 576, 16]);  mm_118 = None
        add_551: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1217, arg424_1);  view_1217 = arg424_1 = None
        permute_795: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_551, [0, 3, 1, 2]);  add_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_839: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_795, memory_format = torch.contiguous_format);  permute_795 = None
        amax_59: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_839, [-1], True)
        sub_183: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_839, amax_59);  clone_839 = amax_59 = None
        exp_59: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_183);  sub_183 = None
        sum_60: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_59, [-1], True)
        div_59: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_59, sum_60);  exp_59 = sum_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_796: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_59, [0, 2, 3, 1]);  div_59 = None
        clone_840: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_796, memory_format = torch.contiguous_format);  permute_796 = None
        view_1218: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_840, [2654208, 16]);  clone_840 = None
        permute_797: "f32[16, 16]" = torch.ops.aten.permute.default(arg425_1, [1, 0]);  arg425_1 = None
        mm_119: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1218, permute_797);  view_1218 = permute_797 = None
        view_1219: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_119, [8, 576, 576, 16]);  mm_119 = None
        add_552: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1219, arg426_1);  view_1219 = arg426_1 = None
        permute_798: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_552, [0, 3, 1, 2]);  add_552 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_239: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_798, [8, 16, 576, 576]);  permute_798 = None
        clone_842: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_239, memory_format = torch.contiguous_format);  expand_239 = None
        view_1220: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_842, [128, 576, 576]);  clone_842 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_182: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_791, 0, 2);  permute_791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_240: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_182, [8, 16, 576, 48]);  select_182 = None
        clone_843: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_240, memory_format = torch.contiguous_format);  expand_240 = None
        view_1221: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_843, [128, 576, 48]);  clone_843 = None
        bmm_119: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1220, view_1221);  view_1220 = view_1221 = None
        view_1222: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_119, [8, 16, 576, 48]);  bmm_119 = None
        permute_799: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1222, [0, 2, 1, 3]);  view_1222 = None
        clone_844: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_799, memory_format = torch.contiguous_format);  permute_799 = None
        view_1223: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_844, [8, 576, 768]);  clone_844 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1224: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1223, [4608, 768]);  view_1223 = None
        permute_800: "f32[768, 768]" = torch.ops.aten.permute.default(arg427_1, [1, 0]);  arg427_1 = None
        
        # No stacktrace found for following nodes
        mm_default_56: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1224, permute_800);  view_1224 = permute_800 = None
        add_tensor_56: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_56, arg428_1);  mm_default_56 = arg428_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1225: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_56, [8, 576, 768]);  add_tensor_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_613: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg418_1, view_1225);  arg418_1 = view_1225 = None
        add_553: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_548, mul_613);  add_548 = mul_613 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_846: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_553, memory_format = torch.contiguous_format)
        var_mean_124 = torch.ops.aten.var_mean.correction(clone_846, [2], correction = 0, keepdim = True)
        getitem_256: "f32[8, 576, 1]" = var_mean_124[0]
        getitem_257: "f32[8, 576, 1]" = var_mean_124[1];  var_mean_124 = None
        sub_184: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_846, getitem_257);  clone_846 = getitem_257 = None
        add_554: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_256, 1e-06);  getitem_256 = None
        rsqrt_124: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_554);  add_554 = None
        mul_614: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_184, rsqrt_124);  sub_184 = rsqrt_124 = None
        mul_615: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_614, arg430_1);  mul_614 = arg430_1 = None
        add_555: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_615, arg431_1);  mul_615 = arg431_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1226: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_555, [4608, 768]);  add_555 = None
        permute_801: "f32[768, 3072]" = torch.ops.aten.permute.default(arg432_1, [1, 0]);  arg432_1 = None
        
        # No stacktrace found for following nodes
        mm_default_55: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1226, permute_801);  view_1226 = permute_801 = None
        add_tensor_55: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_55, arg433_1);  mm_default_55 = arg433_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1227: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_55, [8, 576, 3072]);  add_tensor_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_616: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1227, 0.5)
        mul_617: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1227, 0.7071067811865476);  view_1227 = None
        erf_61: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_617);  mul_617 = None
        add_556: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_61, 1);  erf_61 = None
        mul_618: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_616, add_556);  mul_616 = add_556 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1228: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_618, [4608, 3072]);  mul_618 = None
        permute_802: "f32[3072, 768]" = torch.ops.aten.permute.default(arg434_1, [1, 0]);  arg434_1 = None
        
        # No stacktrace found for following nodes
        mm_default_54: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1228, permute_802);  view_1228 = permute_802 = None
        add_tensor_54: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_54, arg435_1);  mm_default_54 = arg435_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1229: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_54, [8, 576, 768]);  add_tensor_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_619: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg429_1, view_1229);  arg429_1 = view_1229 = None
        add_557: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_553, mul_619);  add_553 = mul_619 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_849: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_557, memory_format = torch.contiguous_format)
        var_mean_125 = torch.ops.aten.var_mean.correction(clone_849, [2], correction = 0, keepdim = True)
        getitem_258: "f32[8, 576, 1]" = var_mean_125[0]
        getitem_259: "f32[8, 576, 1]" = var_mean_125[1];  var_mean_125 = None
        sub_185: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_849, getitem_259);  clone_849 = getitem_259 = None
        add_558: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_258, 1e-06);  getitem_258 = None
        rsqrt_125: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_558);  add_558 = None
        mul_620: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_185, rsqrt_125);  sub_185 = rsqrt_125 = None
        mul_621: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_620, arg437_1);  mul_620 = arg437_1 = None
        add_559: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_621, arg438_1);  mul_621 = arg438_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1230: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_559, [4608, 768]);  add_559 = None
        permute_803: "f32[768, 2304]" = torch.ops.aten.permute.default(arg439_1, [1, 0]);  arg439_1 = None
        
        # No stacktrace found for following nodes
        mm_default_53: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1230, permute_803);  view_1230 = permute_803 = None
        add_tensor_53: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_53, arg440_1);  mm_default_53 = arg440_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1231: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_53, [8, 576, 2304]);  add_tensor_53 = None
        view_1232: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1231, [8, 576, 3, 16, 48]);  view_1231 = None
        permute_804: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1232, [2, 0, 3, 1, 4]);  view_1232 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_183: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_804, 0, 0)
        mul_622: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_183, 0.14433756729740643);  select_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_241: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_622, [8, 16, 576, 48]);  mul_622 = None
        clone_850: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_241, memory_format = torch.contiguous_format);  expand_241 = None
        view_1233: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_850, [128, 576, 48]);  clone_850 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_184: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_804, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_805: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_184, [0, 1, 3, 2]);  select_184 = None
        expand_242: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_805, [8, 16, 48, 576]);  permute_805 = None
        clone_851: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_242, memory_format = torch.contiguous_format);  expand_242 = None
        view_1234: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_851, [128, 48, 576]);  clone_851 = None
        bmm_120: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1233, view_1234);  view_1233 = view_1234 = None
        view_1235: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_120, [8, 16, 576, 576]);  bmm_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_806: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1235, [0, 2, 3, 1]);  view_1235 = None
        clone_852: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_806, memory_format = torch.contiguous_format);  permute_806 = None
        view_1236: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_852, [2654208, 16]);  clone_852 = None
        permute_807: "f32[16, 16]" = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
        mm_120: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1236, permute_807);  view_1236 = permute_807 = None
        view_1237: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_120, [8, 576, 576, 16]);  mm_120 = None
        add_560: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1237, arg442_1);  view_1237 = arg442_1 = None
        permute_808: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_560, [0, 3, 1, 2]);  add_560 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_853: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_808, memory_format = torch.contiguous_format);  permute_808 = None
        amax_60: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_853, [-1], True)
        sub_186: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_853, amax_60);  clone_853 = amax_60 = None
        exp_60: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_186);  sub_186 = None
        sum_61: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_60, [-1], True)
        div_60: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_60, sum_61);  exp_60 = sum_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_809: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_60, [0, 2, 3, 1]);  div_60 = None
        clone_854: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_809, memory_format = torch.contiguous_format);  permute_809 = None
        view_1238: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_854, [2654208, 16]);  clone_854 = None
        permute_810: "f32[16, 16]" = torch.ops.aten.permute.default(arg443_1, [1, 0]);  arg443_1 = None
        mm_121: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1238, permute_810);  view_1238 = permute_810 = None
        view_1239: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_121, [8, 576, 576, 16]);  mm_121 = None
        add_561: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1239, arg444_1);  view_1239 = arg444_1 = None
        permute_811: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_561, [0, 3, 1, 2]);  add_561 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_243: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_811, [8, 16, 576, 576]);  permute_811 = None
        clone_856: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_243, memory_format = torch.contiguous_format);  expand_243 = None
        view_1240: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_856, [128, 576, 576]);  clone_856 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_185: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_804, 0, 2);  permute_804 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_244: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_185, [8, 16, 576, 48]);  select_185 = None
        clone_857: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_244, memory_format = torch.contiguous_format);  expand_244 = None
        view_1241: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_857, [128, 576, 48]);  clone_857 = None
        bmm_121: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1240, view_1241);  view_1240 = view_1241 = None
        view_1242: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_121, [8, 16, 576, 48]);  bmm_121 = None
        permute_812: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1242, [0, 2, 1, 3]);  view_1242 = None
        clone_858: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_812, memory_format = torch.contiguous_format);  permute_812 = None
        view_1243: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_858, [8, 576, 768]);  clone_858 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1244: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1243, [4608, 768]);  view_1243 = None
        permute_813: "f32[768, 768]" = torch.ops.aten.permute.default(arg445_1, [1, 0]);  arg445_1 = None
        
        # No stacktrace found for following nodes
        mm_default_52: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1244, permute_813);  view_1244 = permute_813 = None
        add_tensor_52: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_52, arg446_1);  mm_default_52 = arg446_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1245: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_52, [8, 576, 768]);  add_tensor_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_623: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg436_1, view_1245);  arg436_1 = view_1245 = None
        add_562: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_557, mul_623);  add_557 = mul_623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_860: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_562, memory_format = torch.contiguous_format)
        var_mean_126 = torch.ops.aten.var_mean.correction(clone_860, [2], correction = 0, keepdim = True)
        getitem_260: "f32[8, 576, 1]" = var_mean_126[0]
        getitem_261: "f32[8, 576, 1]" = var_mean_126[1];  var_mean_126 = None
        sub_187: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_860, getitem_261);  clone_860 = getitem_261 = None
        add_563: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_260, 1e-06);  getitem_260 = None
        rsqrt_126: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_563);  add_563 = None
        mul_624: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_187, rsqrt_126);  sub_187 = rsqrt_126 = None
        mul_625: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_624, arg448_1);  mul_624 = arg448_1 = None
        add_564: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_625, arg449_1);  mul_625 = arg449_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1246: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_564, [4608, 768]);  add_564 = None
        permute_814: "f32[768, 3072]" = torch.ops.aten.permute.default(arg450_1, [1, 0]);  arg450_1 = None
        
        # No stacktrace found for following nodes
        mm_default_51: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1246, permute_814);  view_1246 = permute_814 = None
        add_tensor_51: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_51, arg451_1);  mm_default_51 = arg451_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1247: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_51, [8, 576, 3072]);  add_tensor_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_626: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1247, 0.5)
        mul_627: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1247, 0.7071067811865476);  view_1247 = None
        erf_62: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_627);  mul_627 = None
        add_565: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_62, 1);  erf_62 = None
        mul_628: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_626, add_565);  mul_626 = add_565 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1248: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_628, [4608, 3072]);  mul_628 = None
        permute_815: "f32[3072, 768]" = torch.ops.aten.permute.default(arg452_1, [1, 0]);  arg452_1 = None
        
        # No stacktrace found for following nodes
        mm_default_50: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1248, permute_815);  view_1248 = permute_815 = None
        add_tensor_50: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_50, arg453_1);  mm_default_50 = arg453_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1249: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_50, [8, 576, 768]);  add_tensor_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_629: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg447_1, view_1249);  arg447_1 = view_1249 = None
        add_566: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_562, mul_629);  add_562 = mul_629 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_863: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_566, memory_format = torch.contiguous_format)
        var_mean_127 = torch.ops.aten.var_mean.correction(clone_863, [2], correction = 0, keepdim = True)
        getitem_262: "f32[8, 576, 1]" = var_mean_127[0]
        getitem_263: "f32[8, 576, 1]" = var_mean_127[1];  var_mean_127 = None
        sub_188: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_863, getitem_263);  clone_863 = getitem_263 = None
        add_567: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_262, 1e-06);  getitem_262 = None
        rsqrt_127: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_567);  add_567 = None
        mul_630: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_188, rsqrt_127);  sub_188 = rsqrt_127 = None
        mul_631: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_630, arg455_1);  mul_630 = arg455_1 = None
        add_568: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_631, arg456_1);  mul_631 = arg456_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1250: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_568, [4608, 768]);  add_568 = None
        permute_816: "f32[768, 2304]" = torch.ops.aten.permute.default(arg457_1, [1, 0]);  arg457_1 = None
        
        # No stacktrace found for following nodes
        mm_default_49: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1250, permute_816);  view_1250 = permute_816 = None
        add_tensor_49: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_49, arg458_1);  mm_default_49 = arg458_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1251: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_49, [8, 576, 2304]);  add_tensor_49 = None
        view_1252: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1251, [8, 576, 3, 16, 48]);  view_1251 = None
        permute_817: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1252, [2, 0, 3, 1, 4]);  view_1252 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_186: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_817, 0, 0)
        mul_632: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_186, 0.14433756729740643);  select_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_245: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_632, [8, 16, 576, 48]);  mul_632 = None
        clone_864: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_245, memory_format = torch.contiguous_format);  expand_245 = None
        view_1253: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_864, [128, 576, 48]);  clone_864 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_187: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_817, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_818: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_187, [0, 1, 3, 2]);  select_187 = None
        expand_246: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_818, [8, 16, 48, 576]);  permute_818 = None
        clone_865: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_246, memory_format = torch.contiguous_format);  expand_246 = None
        view_1254: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_865, [128, 48, 576]);  clone_865 = None
        bmm_122: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1253, view_1254);  view_1253 = view_1254 = None
        view_1255: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_122, [8, 16, 576, 576]);  bmm_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_819: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1255, [0, 2, 3, 1]);  view_1255 = None
        clone_866: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_819, memory_format = torch.contiguous_format);  permute_819 = None
        view_1256: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_866, [2654208, 16]);  clone_866 = None
        permute_820: "f32[16, 16]" = torch.ops.aten.permute.default(arg459_1, [1, 0]);  arg459_1 = None
        mm_122: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1256, permute_820);  view_1256 = permute_820 = None
        view_1257: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_122, [8, 576, 576, 16]);  mm_122 = None
        add_569: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1257, arg460_1);  view_1257 = arg460_1 = None
        permute_821: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_569, [0, 3, 1, 2]);  add_569 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_867: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_821, memory_format = torch.contiguous_format);  permute_821 = None
        amax_61: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_867, [-1], True)
        sub_189: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_867, amax_61);  clone_867 = amax_61 = None
        exp_61: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_189);  sub_189 = None
        sum_62: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_61, [-1], True)
        div_61: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_61, sum_62);  exp_61 = sum_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_822: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_61, [0, 2, 3, 1]);  div_61 = None
        clone_868: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_822, memory_format = torch.contiguous_format);  permute_822 = None
        view_1258: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_868, [2654208, 16]);  clone_868 = None
        permute_823: "f32[16, 16]" = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
        mm_123: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1258, permute_823);  view_1258 = permute_823 = None
        view_1259: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_123, [8, 576, 576, 16]);  mm_123 = None
        add_570: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1259, arg462_1);  view_1259 = arg462_1 = None
        permute_824: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_570, [0, 3, 1, 2]);  add_570 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_247: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_824, [8, 16, 576, 576]);  permute_824 = None
        clone_870: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_247, memory_format = torch.contiguous_format);  expand_247 = None
        view_1260: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_870, [128, 576, 576]);  clone_870 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_188: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_817, 0, 2);  permute_817 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_248: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_188, [8, 16, 576, 48]);  select_188 = None
        clone_871: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_248, memory_format = torch.contiguous_format);  expand_248 = None
        view_1261: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_871, [128, 576, 48]);  clone_871 = None
        bmm_123: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1260, view_1261);  view_1260 = view_1261 = None
        view_1262: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_123, [8, 16, 576, 48]);  bmm_123 = None
        permute_825: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1262, [0, 2, 1, 3]);  view_1262 = None
        clone_872: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_825, memory_format = torch.contiguous_format);  permute_825 = None
        view_1263: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_872, [8, 576, 768]);  clone_872 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1264: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1263, [4608, 768]);  view_1263 = None
        permute_826: "f32[768, 768]" = torch.ops.aten.permute.default(arg463_1, [1, 0]);  arg463_1 = None
        
        # No stacktrace found for following nodes
        mm_default_48: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1264, permute_826);  view_1264 = permute_826 = None
        add_tensor_48: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_48, arg464_1);  mm_default_48 = arg464_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1265: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_48, [8, 576, 768]);  add_tensor_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_633: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg454_1, view_1265);  arg454_1 = view_1265 = None
        add_571: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_566, mul_633);  add_566 = mul_633 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_874: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_571, memory_format = torch.contiguous_format)
        var_mean_128 = torch.ops.aten.var_mean.correction(clone_874, [2], correction = 0, keepdim = True)
        getitem_264: "f32[8, 576, 1]" = var_mean_128[0]
        getitem_265: "f32[8, 576, 1]" = var_mean_128[1];  var_mean_128 = None
        sub_190: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_874, getitem_265);  clone_874 = getitem_265 = None
        add_572: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_264, 1e-06);  getitem_264 = None
        rsqrt_128: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_572);  add_572 = None
        mul_634: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_190, rsqrt_128);  sub_190 = rsqrt_128 = None
        mul_635: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_634, arg466_1);  mul_634 = arg466_1 = None
        add_573: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_635, arg467_1);  mul_635 = arg467_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1266: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_573, [4608, 768]);  add_573 = None
        permute_827: "f32[768, 3072]" = torch.ops.aten.permute.default(arg468_1, [1, 0]);  arg468_1 = None
        
        # No stacktrace found for following nodes
        mm_default_47: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1266, permute_827);  view_1266 = permute_827 = None
        add_tensor_47: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_47, arg469_1);  mm_default_47 = arg469_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1267: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_47, [8, 576, 3072]);  add_tensor_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_636: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1267, 0.5)
        mul_637: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1267, 0.7071067811865476);  view_1267 = None
        erf_63: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_637);  mul_637 = None
        add_574: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_63, 1);  erf_63 = None
        mul_638: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_636, add_574);  mul_636 = add_574 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1268: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_638, [4608, 3072]);  mul_638 = None
        permute_828: "f32[3072, 768]" = torch.ops.aten.permute.default(arg470_1, [1, 0]);  arg470_1 = None
        
        # No stacktrace found for following nodes
        mm_default_46: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1268, permute_828);  view_1268 = permute_828 = None
        add_tensor_46: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_46, arg471_1);  mm_default_46 = arg471_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1269: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_46, [8, 576, 768]);  add_tensor_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_639: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg465_1, view_1269);  arg465_1 = view_1269 = None
        add_575: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_571, mul_639);  add_571 = mul_639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_877: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_575, memory_format = torch.contiguous_format)
        var_mean_129 = torch.ops.aten.var_mean.correction(clone_877, [2], correction = 0, keepdim = True)
        getitem_266: "f32[8, 576, 1]" = var_mean_129[0]
        getitem_267: "f32[8, 576, 1]" = var_mean_129[1];  var_mean_129 = None
        sub_191: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_877, getitem_267);  clone_877 = getitem_267 = None
        add_576: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-06);  getitem_266 = None
        rsqrt_129: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_576);  add_576 = None
        mul_640: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_191, rsqrt_129);  sub_191 = rsqrt_129 = None
        mul_641: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_640, arg473_1);  mul_640 = arg473_1 = None
        add_577: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_641, arg474_1);  mul_641 = arg474_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1270: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_577, [4608, 768]);  add_577 = None
        permute_829: "f32[768, 2304]" = torch.ops.aten.permute.default(arg475_1, [1, 0]);  arg475_1 = None
        
        # No stacktrace found for following nodes
        mm_default_45: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1270, permute_829);  view_1270 = permute_829 = None
        add_tensor_45: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_45, arg476_1);  mm_default_45 = arg476_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1271: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_45, [8, 576, 2304]);  add_tensor_45 = None
        view_1272: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1271, [8, 576, 3, 16, 48]);  view_1271 = None
        permute_830: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1272, [2, 0, 3, 1, 4]);  view_1272 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_189: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_830, 0, 0)
        mul_642: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_189, 0.14433756729740643);  select_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_249: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_642, [8, 16, 576, 48]);  mul_642 = None
        clone_878: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_249, memory_format = torch.contiguous_format);  expand_249 = None
        view_1273: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_878, [128, 576, 48]);  clone_878 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_190: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_830, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_831: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_190, [0, 1, 3, 2]);  select_190 = None
        expand_250: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_831, [8, 16, 48, 576]);  permute_831 = None
        clone_879: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_250, memory_format = torch.contiguous_format);  expand_250 = None
        view_1274: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_879, [128, 48, 576]);  clone_879 = None
        bmm_124: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1273, view_1274);  view_1273 = view_1274 = None
        view_1275: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_124, [8, 16, 576, 576]);  bmm_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_832: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1275, [0, 2, 3, 1]);  view_1275 = None
        clone_880: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_832, memory_format = torch.contiguous_format);  permute_832 = None
        view_1276: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_880, [2654208, 16]);  clone_880 = None
        permute_833: "f32[16, 16]" = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
        mm_124: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1276, permute_833);  view_1276 = permute_833 = None
        view_1277: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_124, [8, 576, 576, 16]);  mm_124 = None
        add_578: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1277, arg478_1);  view_1277 = arg478_1 = None
        permute_834: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_578, [0, 3, 1, 2]);  add_578 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_881: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_834, memory_format = torch.contiguous_format);  permute_834 = None
        amax_62: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_881, [-1], True)
        sub_192: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_881, amax_62);  clone_881 = amax_62 = None
        exp_62: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_192);  sub_192 = None
        sum_63: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_62, [-1], True)
        div_62: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_62, sum_63);  exp_62 = sum_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_835: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_62, [0, 2, 3, 1]);  div_62 = None
        clone_882: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_835, memory_format = torch.contiguous_format);  permute_835 = None
        view_1278: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_882, [2654208, 16]);  clone_882 = None
        permute_836: "f32[16, 16]" = torch.ops.aten.permute.default(arg479_1, [1, 0]);  arg479_1 = None
        mm_125: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1278, permute_836);  view_1278 = permute_836 = None
        view_1279: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_125, [8, 576, 576, 16]);  mm_125 = None
        add_579: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1279, arg480_1);  view_1279 = arg480_1 = None
        permute_837: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_579, [0, 3, 1, 2]);  add_579 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_251: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_837, [8, 16, 576, 576]);  permute_837 = None
        clone_884: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_251, memory_format = torch.contiguous_format);  expand_251 = None
        view_1280: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_884, [128, 576, 576]);  clone_884 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_191: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_830, 0, 2);  permute_830 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_252: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_191, [8, 16, 576, 48]);  select_191 = None
        clone_885: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_252, memory_format = torch.contiguous_format);  expand_252 = None
        view_1281: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_885, [128, 576, 48]);  clone_885 = None
        bmm_125: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1280, view_1281);  view_1280 = view_1281 = None
        view_1282: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_125, [8, 16, 576, 48]);  bmm_125 = None
        permute_838: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1282, [0, 2, 1, 3]);  view_1282 = None
        clone_886: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_838, memory_format = torch.contiguous_format);  permute_838 = None
        view_1283: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_886, [8, 576, 768]);  clone_886 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1284: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1283, [4608, 768]);  view_1283 = None
        permute_839: "f32[768, 768]" = torch.ops.aten.permute.default(arg481_1, [1, 0]);  arg481_1 = None
        
        # No stacktrace found for following nodes
        mm_default_44: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1284, permute_839);  view_1284 = permute_839 = None
        add_tensor_44: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_44, arg482_1);  mm_default_44 = arg482_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1285: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_44, [8, 576, 768]);  add_tensor_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_643: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg472_1, view_1285);  arg472_1 = view_1285 = None
        add_580: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_575, mul_643);  add_575 = mul_643 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_888: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_580, memory_format = torch.contiguous_format)
        var_mean_130 = torch.ops.aten.var_mean.correction(clone_888, [2], correction = 0, keepdim = True)
        getitem_268: "f32[8, 576, 1]" = var_mean_130[0]
        getitem_269: "f32[8, 576, 1]" = var_mean_130[1];  var_mean_130 = None
        sub_193: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_888, getitem_269);  clone_888 = getitem_269 = None
        add_581: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_268, 1e-06);  getitem_268 = None
        rsqrt_130: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_581);  add_581 = None
        mul_644: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_193, rsqrt_130);  sub_193 = rsqrt_130 = None
        mul_645: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_644, arg484_1);  mul_644 = arg484_1 = None
        add_582: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_645, arg485_1);  mul_645 = arg485_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1286: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_582, [4608, 768]);  add_582 = None
        permute_840: "f32[768, 3072]" = torch.ops.aten.permute.default(arg486_1, [1, 0]);  arg486_1 = None
        
        # No stacktrace found for following nodes
        mm_default_43: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1286, permute_840);  view_1286 = permute_840 = None
        add_tensor_43: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_43, arg487_1);  mm_default_43 = arg487_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1287: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_43, [8, 576, 3072]);  add_tensor_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_646: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1287, 0.5)
        mul_647: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1287, 0.7071067811865476);  view_1287 = None
        erf_64: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_647);  mul_647 = None
        add_583: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_64, 1);  erf_64 = None
        mul_648: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_646, add_583);  mul_646 = add_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1288: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_648, [4608, 3072]);  mul_648 = None
        permute_841: "f32[3072, 768]" = torch.ops.aten.permute.default(arg488_1, [1, 0]);  arg488_1 = None
        
        # No stacktrace found for following nodes
        mm_default_42: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1288, permute_841);  view_1288 = permute_841 = None
        add_tensor_42: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_42, arg489_1);  mm_default_42 = arg489_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1289: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_42, [8, 576, 768]);  add_tensor_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_649: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg483_1, view_1289);  arg483_1 = view_1289 = None
        add_584: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_580, mul_649);  add_580 = mul_649 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_891: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_584, memory_format = torch.contiguous_format)
        var_mean_131 = torch.ops.aten.var_mean.correction(clone_891, [2], correction = 0, keepdim = True)
        getitem_270: "f32[8, 576, 1]" = var_mean_131[0]
        getitem_271: "f32[8, 576, 1]" = var_mean_131[1];  var_mean_131 = None
        sub_194: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_891, getitem_271);  clone_891 = getitem_271 = None
        add_585: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_270, 1e-06);  getitem_270 = None
        rsqrt_131: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_585);  add_585 = None
        mul_650: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_194, rsqrt_131);  sub_194 = rsqrt_131 = None
        mul_651: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_650, arg491_1);  mul_650 = arg491_1 = None
        add_586: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_651, arg492_1);  mul_651 = arg492_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1290: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_586, [4608, 768]);  add_586 = None
        permute_842: "f32[768, 2304]" = torch.ops.aten.permute.default(arg493_1, [1, 0]);  arg493_1 = None
        
        # No stacktrace found for following nodes
        mm_default_41: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1290, permute_842);  view_1290 = permute_842 = None
        add_tensor_41: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_41, arg494_1);  mm_default_41 = arg494_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1291: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_41, [8, 576, 2304]);  add_tensor_41 = None
        view_1292: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1291, [8, 576, 3, 16, 48]);  view_1291 = None
        permute_843: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1292, [2, 0, 3, 1, 4]);  view_1292 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_192: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_843, 0, 0)
        mul_652: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_192, 0.14433756729740643);  select_192 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_253: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_652, [8, 16, 576, 48]);  mul_652 = None
        clone_892: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_253, memory_format = torch.contiguous_format);  expand_253 = None
        view_1293: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_892, [128, 576, 48]);  clone_892 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_193: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_843, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_844: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_193, [0, 1, 3, 2]);  select_193 = None
        expand_254: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_844, [8, 16, 48, 576]);  permute_844 = None
        clone_893: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_254, memory_format = torch.contiguous_format);  expand_254 = None
        view_1294: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_893, [128, 48, 576]);  clone_893 = None
        bmm_126: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1293, view_1294);  view_1293 = view_1294 = None
        view_1295: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_126, [8, 16, 576, 576]);  bmm_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_845: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1295, [0, 2, 3, 1]);  view_1295 = None
        clone_894: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_845, memory_format = torch.contiguous_format);  permute_845 = None
        view_1296: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_894, [2654208, 16]);  clone_894 = None
        permute_846: "f32[16, 16]" = torch.ops.aten.permute.default(arg495_1, [1, 0]);  arg495_1 = None
        mm_126: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1296, permute_846);  view_1296 = permute_846 = None
        view_1297: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_126, [8, 576, 576, 16]);  mm_126 = None
        add_587: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1297, arg496_1);  view_1297 = arg496_1 = None
        permute_847: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_587, [0, 3, 1, 2]);  add_587 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_895: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_847, memory_format = torch.contiguous_format);  permute_847 = None
        amax_63: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_895, [-1], True)
        sub_195: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_895, amax_63);  clone_895 = amax_63 = None
        exp_63: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_195);  sub_195 = None
        sum_64: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_63, [-1], True)
        div_63: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_63, sum_64);  exp_63 = sum_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_848: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_63, [0, 2, 3, 1]);  div_63 = None
        clone_896: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_848, memory_format = torch.contiguous_format);  permute_848 = None
        view_1298: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_896, [2654208, 16]);  clone_896 = None
        permute_849: "f32[16, 16]" = torch.ops.aten.permute.default(arg497_1, [1, 0]);  arg497_1 = None
        mm_127: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1298, permute_849);  view_1298 = permute_849 = None
        view_1299: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_127, [8, 576, 576, 16]);  mm_127 = None
        add_588: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1299, arg498_1);  view_1299 = arg498_1 = None
        permute_850: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_588, [0, 3, 1, 2]);  add_588 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_255: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_850, [8, 16, 576, 576]);  permute_850 = None
        clone_898: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_255, memory_format = torch.contiguous_format);  expand_255 = None
        view_1300: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_898, [128, 576, 576]);  clone_898 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_194: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_843, 0, 2);  permute_843 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_256: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_194, [8, 16, 576, 48]);  select_194 = None
        clone_899: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_256, memory_format = torch.contiguous_format);  expand_256 = None
        view_1301: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_899, [128, 576, 48]);  clone_899 = None
        bmm_127: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1300, view_1301);  view_1300 = view_1301 = None
        view_1302: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_127, [8, 16, 576, 48]);  bmm_127 = None
        permute_851: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1302, [0, 2, 1, 3]);  view_1302 = None
        clone_900: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_851, memory_format = torch.contiguous_format);  permute_851 = None
        view_1303: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_900, [8, 576, 768]);  clone_900 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1304: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1303, [4608, 768]);  view_1303 = None
        permute_852: "f32[768, 768]" = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
        
        # No stacktrace found for following nodes
        mm_default_40: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1304, permute_852);  view_1304 = permute_852 = None
        add_tensor_40: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_40, arg500_1);  mm_default_40 = arg500_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1305: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_40, [8, 576, 768]);  add_tensor_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_653: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg490_1, view_1305);  arg490_1 = view_1305 = None
        add_589: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_584, mul_653);  add_584 = mul_653 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_902: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_589, memory_format = torch.contiguous_format)
        var_mean_132 = torch.ops.aten.var_mean.correction(clone_902, [2], correction = 0, keepdim = True)
        getitem_272: "f32[8, 576, 1]" = var_mean_132[0]
        getitem_273: "f32[8, 576, 1]" = var_mean_132[1];  var_mean_132 = None
        sub_196: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_902, getitem_273);  clone_902 = getitem_273 = None
        add_590: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_272, 1e-06);  getitem_272 = None
        rsqrt_132: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_590);  add_590 = None
        mul_654: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_196, rsqrt_132);  sub_196 = rsqrt_132 = None
        mul_655: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_654, arg502_1);  mul_654 = arg502_1 = None
        add_591: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_655, arg503_1);  mul_655 = arg503_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1306: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_591, [4608, 768]);  add_591 = None
        permute_853: "f32[768, 3072]" = torch.ops.aten.permute.default(arg504_1, [1, 0]);  arg504_1 = None
        
        # No stacktrace found for following nodes
        mm_default_39: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1306, permute_853);  view_1306 = permute_853 = None
        add_tensor_39: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_39, arg505_1);  mm_default_39 = arg505_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1307: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_39, [8, 576, 3072]);  add_tensor_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_656: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1307, 0.5)
        mul_657: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1307, 0.7071067811865476);  view_1307 = None
        erf_65: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_657);  mul_657 = None
        add_592: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_65, 1);  erf_65 = None
        mul_658: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_656, add_592);  mul_656 = add_592 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1308: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_658, [4608, 3072]);  mul_658 = None
        permute_854: "f32[3072, 768]" = torch.ops.aten.permute.default(arg506_1, [1, 0]);  arg506_1 = None
        
        # No stacktrace found for following nodes
        mm_default_38: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1308, permute_854);  view_1308 = permute_854 = None
        add_tensor_38: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_38, arg507_1);  mm_default_38 = arg507_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1309: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_38, [8, 576, 768]);  add_tensor_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_659: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg501_1, view_1309);  arg501_1 = view_1309 = None
        add_593: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_589, mul_659);  add_589 = mul_659 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_905: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_593, memory_format = torch.contiguous_format)
        var_mean_133 = torch.ops.aten.var_mean.correction(clone_905, [2], correction = 0, keepdim = True)
        getitem_274: "f32[8, 576, 1]" = var_mean_133[0]
        getitem_275: "f32[8, 576, 1]" = var_mean_133[1];  var_mean_133 = None
        sub_197: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_905, getitem_275);  clone_905 = getitem_275 = None
        add_594: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_274, 1e-06);  getitem_274 = None
        rsqrt_133: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_594);  add_594 = None
        mul_660: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_197, rsqrt_133);  sub_197 = rsqrt_133 = None
        mul_661: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_660, arg509_1);  mul_660 = arg509_1 = None
        add_595: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_661, arg510_1);  mul_661 = arg510_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1310: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_595, [4608, 768]);  add_595 = None
        permute_855: "f32[768, 2304]" = torch.ops.aten.permute.default(arg511_1, [1, 0]);  arg511_1 = None
        
        # No stacktrace found for following nodes
        mm_default_37: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1310, permute_855);  view_1310 = permute_855 = None
        add_tensor_37: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_37, arg512_1);  mm_default_37 = arg512_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1311: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_37, [8, 576, 2304]);  add_tensor_37 = None
        view_1312: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1311, [8, 576, 3, 16, 48]);  view_1311 = None
        permute_856: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1312, [2, 0, 3, 1, 4]);  view_1312 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_195: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_856, 0, 0)
        mul_662: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_195, 0.14433756729740643);  select_195 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_257: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_662, [8, 16, 576, 48]);  mul_662 = None
        clone_906: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_257, memory_format = torch.contiguous_format);  expand_257 = None
        view_1313: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_906, [128, 576, 48]);  clone_906 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_196: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_856, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_857: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_196, [0, 1, 3, 2]);  select_196 = None
        expand_258: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_857, [8, 16, 48, 576]);  permute_857 = None
        clone_907: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_258, memory_format = torch.contiguous_format);  expand_258 = None
        view_1314: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_907, [128, 48, 576]);  clone_907 = None
        bmm_128: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1313, view_1314);  view_1313 = view_1314 = None
        view_1315: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_128, [8, 16, 576, 576]);  bmm_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_858: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1315, [0, 2, 3, 1]);  view_1315 = None
        clone_908: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_858, memory_format = torch.contiguous_format);  permute_858 = None
        view_1316: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_908, [2654208, 16]);  clone_908 = None
        permute_859: "f32[16, 16]" = torch.ops.aten.permute.default(arg513_1, [1, 0]);  arg513_1 = None
        mm_128: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1316, permute_859);  view_1316 = permute_859 = None
        view_1317: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_128, [8, 576, 576, 16]);  mm_128 = None
        add_596: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1317, arg514_1);  view_1317 = arg514_1 = None
        permute_860: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_596, [0, 3, 1, 2]);  add_596 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_909: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_860, memory_format = torch.contiguous_format);  permute_860 = None
        amax_64: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_909, [-1], True)
        sub_198: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_909, amax_64);  clone_909 = amax_64 = None
        exp_64: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_198);  sub_198 = None
        sum_65: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_64, [-1], True)
        div_64: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_64, sum_65);  exp_64 = sum_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_861: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_64, [0, 2, 3, 1]);  div_64 = None
        clone_910: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_861, memory_format = torch.contiguous_format);  permute_861 = None
        view_1318: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_910, [2654208, 16]);  clone_910 = None
        permute_862: "f32[16, 16]" = torch.ops.aten.permute.default(arg515_1, [1, 0]);  arg515_1 = None
        mm_129: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1318, permute_862);  view_1318 = permute_862 = None
        view_1319: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_129, [8, 576, 576, 16]);  mm_129 = None
        add_597: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1319, arg516_1);  view_1319 = arg516_1 = None
        permute_863: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_597, [0, 3, 1, 2]);  add_597 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_259: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_863, [8, 16, 576, 576]);  permute_863 = None
        clone_912: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_259, memory_format = torch.contiguous_format);  expand_259 = None
        view_1320: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_912, [128, 576, 576]);  clone_912 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_197: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_856, 0, 2);  permute_856 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_260: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_197, [8, 16, 576, 48]);  select_197 = None
        clone_913: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_260, memory_format = torch.contiguous_format);  expand_260 = None
        view_1321: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_913, [128, 576, 48]);  clone_913 = None
        bmm_129: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1320, view_1321);  view_1320 = view_1321 = None
        view_1322: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_129, [8, 16, 576, 48]);  bmm_129 = None
        permute_864: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1322, [0, 2, 1, 3]);  view_1322 = None
        clone_914: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_864, memory_format = torch.contiguous_format);  permute_864 = None
        view_1323: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_914, [8, 576, 768]);  clone_914 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1324: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1323, [4608, 768]);  view_1323 = None
        permute_865: "f32[768, 768]" = torch.ops.aten.permute.default(arg517_1, [1, 0]);  arg517_1 = None
        
        # No stacktrace found for following nodes
        mm_default_36: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1324, permute_865);  view_1324 = permute_865 = None
        add_tensor_36: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_36, arg518_1);  mm_default_36 = arg518_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1325: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_36, [8, 576, 768]);  add_tensor_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_663: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg508_1, view_1325);  arg508_1 = view_1325 = None
        add_598: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_593, mul_663);  add_593 = mul_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_916: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_598, memory_format = torch.contiguous_format)
        var_mean_134 = torch.ops.aten.var_mean.correction(clone_916, [2], correction = 0, keepdim = True)
        getitem_276: "f32[8, 576, 1]" = var_mean_134[0]
        getitem_277: "f32[8, 576, 1]" = var_mean_134[1];  var_mean_134 = None
        sub_199: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_916, getitem_277);  clone_916 = getitem_277 = None
        add_599: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_276, 1e-06);  getitem_276 = None
        rsqrt_134: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_599);  add_599 = None
        mul_664: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_199, rsqrt_134);  sub_199 = rsqrt_134 = None
        mul_665: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_664, arg520_1);  mul_664 = arg520_1 = None
        add_600: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_665, arg521_1);  mul_665 = arg521_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1326: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_600, [4608, 768]);  add_600 = None
        permute_866: "f32[768, 3072]" = torch.ops.aten.permute.default(arg522_1, [1, 0]);  arg522_1 = None
        
        # No stacktrace found for following nodes
        mm_default_35: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1326, permute_866);  view_1326 = permute_866 = None
        add_tensor_35: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_35, arg523_1);  mm_default_35 = arg523_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1327: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_35, [8, 576, 3072]);  add_tensor_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_666: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1327, 0.5)
        mul_667: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1327, 0.7071067811865476);  view_1327 = None
        erf_66: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_667);  mul_667 = None
        add_601: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_66, 1);  erf_66 = None
        mul_668: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_666, add_601);  mul_666 = add_601 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1328: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_668, [4608, 3072]);  mul_668 = None
        permute_867: "f32[3072, 768]" = torch.ops.aten.permute.default(arg524_1, [1, 0]);  arg524_1 = None
        
        # No stacktrace found for following nodes
        mm_default_34: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1328, permute_867);  view_1328 = permute_867 = None
        add_tensor_34: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_34, arg525_1);  mm_default_34 = arg525_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1329: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_34, [8, 576, 768]);  add_tensor_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_669: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg519_1, view_1329);  arg519_1 = view_1329 = None
        add_602: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_598, mul_669);  add_598 = mul_669 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_919: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_602, memory_format = torch.contiguous_format)
        var_mean_135 = torch.ops.aten.var_mean.correction(clone_919, [2], correction = 0, keepdim = True)
        getitem_278: "f32[8, 576, 1]" = var_mean_135[0]
        getitem_279: "f32[8, 576, 1]" = var_mean_135[1];  var_mean_135 = None
        sub_200: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_919, getitem_279);  clone_919 = getitem_279 = None
        add_603: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_278, 1e-06);  getitem_278 = None
        rsqrt_135: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_603);  add_603 = None
        mul_670: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_200, rsqrt_135);  sub_200 = rsqrt_135 = None
        mul_671: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_670, arg527_1);  mul_670 = arg527_1 = None
        add_604: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_671, arg528_1);  mul_671 = arg528_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1330: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_604, [4608, 768]);  add_604 = None
        permute_868: "f32[768, 2304]" = torch.ops.aten.permute.default(arg529_1, [1, 0]);  arg529_1 = None
        
        # No stacktrace found for following nodes
        mm_default_33: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1330, permute_868);  view_1330 = permute_868 = None
        add_tensor_33: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_33, arg530_1);  mm_default_33 = arg530_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1331: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 576, 2304]);  add_tensor_33 = None
        view_1332: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1331, [8, 576, 3, 16, 48]);  view_1331 = None
        permute_869: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1332, [2, 0, 3, 1, 4]);  view_1332 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_198: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_869, 0, 0)
        mul_672: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_198, 0.14433756729740643);  select_198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_261: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_672, [8, 16, 576, 48]);  mul_672 = None
        clone_920: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_261, memory_format = torch.contiguous_format);  expand_261 = None
        view_1333: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_920, [128, 576, 48]);  clone_920 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_199: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_869, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_870: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_199, [0, 1, 3, 2]);  select_199 = None
        expand_262: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_870, [8, 16, 48, 576]);  permute_870 = None
        clone_921: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_262, memory_format = torch.contiguous_format);  expand_262 = None
        view_1334: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_921, [128, 48, 576]);  clone_921 = None
        bmm_130: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1333, view_1334);  view_1333 = view_1334 = None
        view_1335: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_130, [8, 16, 576, 576]);  bmm_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_871: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1335, [0, 2, 3, 1]);  view_1335 = None
        clone_922: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_871, memory_format = torch.contiguous_format);  permute_871 = None
        view_1336: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_922, [2654208, 16]);  clone_922 = None
        permute_872: "f32[16, 16]" = torch.ops.aten.permute.default(arg531_1, [1, 0]);  arg531_1 = None
        mm_130: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1336, permute_872);  view_1336 = permute_872 = None
        view_1337: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_130, [8, 576, 576, 16]);  mm_130 = None
        add_605: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1337, arg532_1);  view_1337 = arg532_1 = None
        permute_873: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_605, [0, 3, 1, 2]);  add_605 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_923: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_873, memory_format = torch.contiguous_format);  permute_873 = None
        amax_65: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_923, [-1], True)
        sub_201: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_923, amax_65);  clone_923 = amax_65 = None
        exp_65: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_201);  sub_201 = None
        sum_66: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_65, [-1], True)
        div_65: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_65, sum_66);  exp_65 = sum_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_874: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_65, [0, 2, 3, 1]);  div_65 = None
        clone_924: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_874, memory_format = torch.contiguous_format);  permute_874 = None
        view_1338: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_924, [2654208, 16]);  clone_924 = None
        permute_875: "f32[16, 16]" = torch.ops.aten.permute.default(arg533_1, [1, 0]);  arg533_1 = None
        mm_131: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1338, permute_875);  view_1338 = permute_875 = None
        view_1339: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_131, [8, 576, 576, 16]);  mm_131 = None
        add_606: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1339, arg534_1);  view_1339 = arg534_1 = None
        permute_876: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_606, [0, 3, 1, 2]);  add_606 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_263: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_876, [8, 16, 576, 576]);  permute_876 = None
        clone_926: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_263, memory_format = torch.contiguous_format);  expand_263 = None
        view_1340: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_926, [128, 576, 576]);  clone_926 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_200: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_869, 0, 2);  permute_869 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_264: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_200, [8, 16, 576, 48]);  select_200 = None
        clone_927: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_264, memory_format = torch.contiguous_format);  expand_264 = None
        view_1341: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_927, [128, 576, 48]);  clone_927 = None
        bmm_131: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1340, view_1341);  view_1340 = view_1341 = None
        view_1342: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_131, [8, 16, 576, 48]);  bmm_131 = None
        permute_877: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1342, [0, 2, 1, 3]);  view_1342 = None
        clone_928: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_877, memory_format = torch.contiguous_format);  permute_877 = None
        view_1343: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_928, [8, 576, 768]);  clone_928 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1344: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1343, [4608, 768]);  view_1343 = None
        permute_878: "f32[768, 768]" = torch.ops.aten.permute.default(arg535_1, [1, 0]);  arg535_1 = None
        
        # No stacktrace found for following nodes
        mm_default_32: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1344, permute_878);  view_1344 = permute_878 = None
        add_tensor_32: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_32, arg536_1);  mm_default_32 = arg536_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1345: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 576, 768]);  add_tensor_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_673: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg526_1, view_1345);  arg526_1 = view_1345 = None
        add_607: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_602, mul_673);  add_602 = mul_673 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_930: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_607, memory_format = torch.contiguous_format)
        var_mean_136 = torch.ops.aten.var_mean.correction(clone_930, [2], correction = 0, keepdim = True)
        getitem_280: "f32[8, 576, 1]" = var_mean_136[0]
        getitem_281: "f32[8, 576, 1]" = var_mean_136[1];  var_mean_136 = None
        sub_202: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_930, getitem_281);  clone_930 = getitem_281 = None
        add_608: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_280, 1e-06);  getitem_280 = None
        rsqrt_136: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_608);  add_608 = None
        mul_674: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_202, rsqrt_136);  sub_202 = rsqrt_136 = None
        mul_675: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_674, arg538_1);  mul_674 = arg538_1 = None
        add_609: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_675, arg539_1);  mul_675 = arg539_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1346: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_609, [4608, 768]);  add_609 = None
        permute_879: "f32[768, 3072]" = torch.ops.aten.permute.default(arg540_1, [1, 0]);  arg540_1 = None
        
        # No stacktrace found for following nodes
        mm_default_31: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1346, permute_879);  view_1346 = permute_879 = None
        add_tensor_31: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_31, arg541_1);  mm_default_31 = arg541_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1347: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 576, 3072]);  add_tensor_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_676: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1347, 0.5)
        mul_677: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1347, 0.7071067811865476);  view_1347 = None
        erf_67: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_677);  mul_677 = None
        add_610: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_67, 1);  erf_67 = None
        mul_678: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_676, add_610);  mul_676 = add_610 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1348: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_678, [4608, 3072]);  mul_678 = None
        permute_880: "f32[3072, 768]" = torch.ops.aten.permute.default(arg542_1, [1, 0]);  arg542_1 = None
        
        # No stacktrace found for following nodes
        mm_default_30: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1348, permute_880);  view_1348 = permute_880 = None
        add_tensor_30: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_30, arg543_1);  mm_default_30 = arg543_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1349: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 576, 768]);  add_tensor_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_679: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg537_1, view_1349);  arg537_1 = view_1349 = None
        add_611: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_607, mul_679);  add_607 = mul_679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_933: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_611, memory_format = torch.contiguous_format)
        var_mean_137 = torch.ops.aten.var_mean.correction(clone_933, [2], correction = 0, keepdim = True)
        getitem_282: "f32[8, 576, 1]" = var_mean_137[0]
        getitem_283: "f32[8, 576, 1]" = var_mean_137[1];  var_mean_137 = None
        sub_203: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_933, getitem_283);  clone_933 = getitem_283 = None
        add_612: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_282, 1e-06);  getitem_282 = None
        rsqrt_137: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_612);  add_612 = None
        mul_680: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_203, rsqrt_137);  sub_203 = rsqrt_137 = None
        mul_681: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_680, arg545_1);  mul_680 = arg545_1 = None
        add_613: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_681, arg546_1);  mul_681 = arg546_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1350: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_613, [4608, 768]);  add_613 = None
        permute_881: "f32[768, 2304]" = torch.ops.aten.permute.default(arg547_1, [1, 0]);  arg547_1 = None
        
        # No stacktrace found for following nodes
        mm_default_29: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1350, permute_881);  view_1350 = permute_881 = None
        add_tensor_29: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_29, arg548_1);  mm_default_29 = arg548_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1351: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_29, [8, 576, 2304]);  add_tensor_29 = None
        view_1352: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1351, [8, 576, 3, 16, 48]);  view_1351 = None
        permute_882: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1352, [2, 0, 3, 1, 4]);  view_1352 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_201: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_882, 0, 0)
        mul_682: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_201, 0.14433756729740643);  select_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_265: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_682, [8, 16, 576, 48]);  mul_682 = None
        clone_934: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_265, memory_format = torch.contiguous_format);  expand_265 = None
        view_1353: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_934, [128, 576, 48]);  clone_934 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_202: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_882, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_883: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_202, [0, 1, 3, 2]);  select_202 = None
        expand_266: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_883, [8, 16, 48, 576]);  permute_883 = None
        clone_935: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_266, memory_format = torch.contiguous_format);  expand_266 = None
        view_1354: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_935, [128, 48, 576]);  clone_935 = None
        bmm_132: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1353, view_1354);  view_1353 = view_1354 = None
        view_1355: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_132, [8, 16, 576, 576]);  bmm_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_884: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1355, [0, 2, 3, 1]);  view_1355 = None
        clone_936: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_884, memory_format = torch.contiguous_format);  permute_884 = None
        view_1356: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_936, [2654208, 16]);  clone_936 = None
        permute_885: "f32[16, 16]" = torch.ops.aten.permute.default(arg549_1, [1, 0]);  arg549_1 = None
        mm_132: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1356, permute_885);  view_1356 = permute_885 = None
        view_1357: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_132, [8, 576, 576, 16]);  mm_132 = None
        add_614: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1357, arg550_1);  view_1357 = arg550_1 = None
        permute_886: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_614, [0, 3, 1, 2]);  add_614 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_937: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_886, memory_format = torch.contiguous_format);  permute_886 = None
        amax_66: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_937, [-1], True)
        sub_204: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_937, amax_66);  clone_937 = amax_66 = None
        exp_66: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_204);  sub_204 = None
        sum_67: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_66, [-1], True)
        div_66: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_66, sum_67);  exp_66 = sum_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_887: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_66, [0, 2, 3, 1]);  div_66 = None
        clone_938: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_887, memory_format = torch.contiguous_format);  permute_887 = None
        view_1358: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_938, [2654208, 16]);  clone_938 = None
        permute_888: "f32[16, 16]" = torch.ops.aten.permute.default(arg551_1, [1, 0]);  arg551_1 = None
        mm_133: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1358, permute_888);  view_1358 = permute_888 = None
        view_1359: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_133, [8, 576, 576, 16]);  mm_133 = None
        add_615: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1359, arg552_1);  view_1359 = arg552_1 = None
        permute_889: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_615, [0, 3, 1, 2]);  add_615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_267: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_889, [8, 16, 576, 576]);  permute_889 = None
        clone_940: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_267, memory_format = torch.contiguous_format);  expand_267 = None
        view_1360: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_940, [128, 576, 576]);  clone_940 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_203: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_882, 0, 2);  permute_882 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_268: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_203, [8, 16, 576, 48]);  select_203 = None
        clone_941: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_268, memory_format = torch.contiguous_format);  expand_268 = None
        view_1361: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_941, [128, 576, 48]);  clone_941 = None
        bmm_133: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1360, view_1361);  view_1360 = view_1361 = None
        view_1362: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_133, [8, 16, 576, 48]);  bmm_133 = None
        permute_890: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1362, [0, 2, 1, 3]);  view_1362 = None
        clone_942: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_890, memory_format = torch.contiguous_format);  permute_890 = None
        view_1363: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_942, [8, 576, 768]);  clone_942 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1364: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1363, [4608, 768]);  view_1363 = None
        permute_891: "f32[768, 768]" = torch.ops.aten.permute.default(arg553_1, [1, 0]);  arg553_1 = None
        
        # No stacktrace found for following nodes
        mm_default_28: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1364, permute_891);  view_1364 = permute_891 = None
        add_tensor_28: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_28, arg554_1);  mm_default_28 = arg554_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1365: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_28, [8, 576, 768]);  add_tensor_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_683: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg544_1, view_1365);  arg544_1 = view_1365 = None
        add_616: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_611, mul_683);  add_611 = mul_683 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_944: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_616, memory_format = torch.contiguous_format)
        var_mean_138 = torch.ops.aten.var_mean.correction(clone_944, [2], correction = 0, keepdim = True)
        getitem_284: "f32[8, 576, 1]" = var_mean_138[0]
        getitem_285: "f32[8, 576, 1]" = var_mean_138[1];  var_mean_138 = None
        sub_205: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_944, getitem_285);  clone_944 = getitem_285 = None
        add_617: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_284, 1e-06);  getitem_284 = None
        rsqrt_138: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_617);  add_617 = None
        mul_684: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_205, rsqrt_138);  sub_205 = rsqrt_138 = None
        mul_685: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_684, arg556_1);  mul_684 = arg556_1 = None
        add_618: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_685, arg557_1);  mul_685 = arg557_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1366: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_618, [4608, 768]);  add_618 = None
        permute_892: "f32[768, 3072]" = torch.ops.aten.permute.default(arg558_1, [1, 0]);  arg558_1 = None
        
        # No stacktrace found for following nodes
        mm_default_27: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1366, permute_892);  view_1366 = permute_892 = None
        add_tensor_27: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_27, arg559_1);  mm_default_27 = arg559_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1367: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_27, [8, 576, 3072]);  add_tensor_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_686: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1367, 0.5)
        mul_687: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1367, 0.7071067811865476);  view_1367 = None
        erf_68: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_687);  mul_687 = None
        add_619: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_68, 1);  erf_68 = None
        mul_688: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_686, add_619);  mul_686 = add_619 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1368: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_688, [4608, 3072]);  mul_688 = None
        permute_893: "f32[3072, 768]" = torch.ops.aten.permute.default(arg560_1, [1, 0]);  arg560_1 = None
        
        # No stacktrace found for following nodes
        mm_default_26: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1368, permute_893);  view_1368 = permute_893 = None
        add_tensor_26: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_26, arg561_1);  mm_default_26 = arg561_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1369: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 576, 768]);  add_tensor_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_689: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg555_1, view_1369);  arg555_1 = view_1369 = None
        add_620: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_616, mul_689);  add_616 = mul_689 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_947: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_620, memory_format = torch.contiguous_format)
        var_mean_139 = torch.ops.aten.var_mean.correction(clone_947, [2], correction = 0, keepdim = True)
        getitem_286: "f32[8, 576, 1]" = var_mean_139[0]
        getitem_287: "f32[8, 576, 1]" = var_mean_139[1];  var_mean_139 = None
        sub_206: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_947, getitem_287);  clone_947 = getitem_287 = None
        add_621: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_286, 1e-06);  getitem_286 = None
        rsqrt_139: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_621);  add_621 = None
        mul_690: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_206, rsqrt_139);  sub_206 = rsqrt_139 = None
        mul_691: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_690, arg563_1);  mul_690 = arg563_1 = None
        add_622: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_691, arg564_1);  mul_691 = arg564_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1370: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_622, [4608, 768]);  add_622 = None
        permute_894: "f32[768, 2304]" = torch.ops.aten.permute.default(arg565_1, [1, 0]);  arg565_1 = None
        
        # No stacktrace found for following nodes
        mm_default_25: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1370, permute_894);  view_1370 = permute_894 = None
        add_tensor_25: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_25, arg566_1);  mm_default_25 = arg566_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1371: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 576, 2304]);  add_tensor_25 = None
        view_1372: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1371, [8, 576, 3, 16, 48]);  view_1371 = None
        permute_895: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1372, [2, 0, 3, 1, 4]);  view_1372 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_204: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_895, 0, 0)
        mul_692: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_204, 0.14433756729740643);  select_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_269: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_692, [8, 16, 576, 48]);  mul_692 = None
        clone_948: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_269, memory_format = torch.contiguous_format);  expand_269 = None
        view_1373: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_948, [128, 576, 48]);  clone_948 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_205: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_895, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_896: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_205, [0, 1, 3, 2]);  select_205 = None
        expand_270: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_896, [8, 16, 48, 576]);  permute_896 = None
        clone_949: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_270, memory_format = torch.contiguous_format);  expand_270 = None
        view_1374: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_949, [128, 48, 576]);  clone_949 = None
        bmm_134: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1373, view_1374);  view_1373 = view_1374 = None
        view_1375: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_134, [8, 16, 576, 576]);  bmm_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_897: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1375, [0, 2, 3, 1]);  view_1375 = None
        clone_950: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_897, memory_format = torch.contiguous_format);  permute_897 = None
        view_1376: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_950, [2654208, 16]);  clone_950 = None
        permute_898: "f32[16, 16]" = torch.ops.aten.permute.default(arg567_1, [1, 0]);  arg567_1 = None
        mm_134: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1376, permute_898);  view_1376 = permute_898 = None
        view_1377: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_134, [8, 576, 576, 16]);  mm_134 = None
        add_623: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1377, arg568_1);  view_1377 = arg568_1 = None
        permute_899: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_623, [0, 3, 1, 2]);  add_623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_951: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_899, memory_format = torch.contiguous_format);  permute_899 = None
        amax_67: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_951, [-1], True)
        sub_207: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_951, amax_67);  clone_951 = amax_67 = None
        exp_67: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_207);  sub_207 = None
        sum_68: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_67, [-1], True)
        div_67: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_67, sum_68);  exp_67 = sum_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_900: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_67, [0, 2, 3, 1]);  div_67 = None
        clone_952: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_900, memory_format = torch.contiguous_format);  permute_900 = None
        view_1378: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_952, [2654208, 16]);  clone_952 = None
        permute_901: "f32[16, 16]" = torch.ops.aten.permute.default(arg569_1, [1, 0]);  arg569_1 = None
        mm_135: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1378, permute_901);  view_1378 = permute_901 = None
        view_1379: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_135, [8, 576, 576, 16]);  mm_135 = None
        add_624: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1379, arg570_1);  view_1379 = arg570_1 = None
        permute_902: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_624, [0, 3, 1, 2]);  add_624 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_271: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_902, [8, 16, 576, 576]);  permute_902 = None
        clone_954: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_271, memory_format = torch.contiguous_format);  expand_271 = None
        view_1380: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_954, [128, 576, 576]);  clone_954 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_206: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_895, 0, 2);  permute_895 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_272: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_206, [8, 16, 576, 48]);  select_206 = None
        clone_955: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_272, memory_format = torch.contiguous_format);  expand_272 = None
        view_1381: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_955, [128, 576, 48]);  clone_955 = None
        bmm_135: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1380, view_1381);  view_1380 = view_1381 = None
        view_1382: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_135, [8, 16, 576, 48]);  bmm_135 = None
        permute_903: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1382, [0, 2, 1, 3]);  view_1382 = None
        clone_956: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_903, memory_format = torch.contiguous_format);  permute_903 = None
        view_1383: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_956, [8, 576, 768]);  clone_956 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1384: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1383, [4608, 768]);  view_1383 = None
        permute_904: "f32[768, 768]" = torch.ops.aten.permute.default(arg571_1, [1, 0]);  arg571_1 = None
        
        # No stacktrace found for following nodes
        mm_default_24: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1384, permute_904);  view_1384 = permute_904 = None
        add_tensor_24: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_24, arg572_1);  mm_default_24 = arg572_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1385: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 576, 768]);  add_tensor_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_693: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg562_1, view_1385);  arg562_1 = view_1385 = None
        add_625: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_620, mul_693);  add_620 = mul_693 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_958: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_625, memory_format = torch.contiguous_format)
        var_mean_140 = torch.ops.aten.var_mean.correction(clone_958, [2], correction = 0, keepdim = True)
        getitem_288: "f32[8, 576, 1]" = var_mean_140[0]
        getitem_289: "f32[8, 576, 1]" = var_mean_140[1];  var_mean_140 = None
        sub_208: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_958, getitem_289);  clone_958 = getitem_289 = None
        add_626: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_288, 1e-06);  getitem_288 = None
        rsqrt_140: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_626);  add_626 = None
        mul_694: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_208, rsqrt_140);  sub_208 = rsqrt_140 = None
        mul_695: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_694, arg574_1);  mul_694 = arg574_1 = None
        add_627: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_695, arg575_1);  mul_695 = arg575_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1386: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_627, [4608, 768]);  add_627 = None
        permute_905: "f32[768, 3072]" = torch.ops.aten.permute.default(arg576_1, [1, 0]);  arg576_1 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1386, permute_905);  view_1386 = permute_905 = None
        add_tensor_23: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_23, arg577_1);  mm_default_23 = arg577_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1387: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 576, 3072]);  add_tensor_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_696: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1387, 0.5)
        mul_697: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1387, 0.7071067811865476);  view_1387 = None
        erf_69: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_697);  mul_697 = None
        add_628: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_69, 1);  erf_69 = None
        mul_698: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_696, add_628);  mul_696 = add_628 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1388: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_698, [4608, 3072]);  mul_698 = None
        permute_906: "f32[3072, 768]" = torch.ops.aten.permute.default(arg578_1, [1, 0]);  arg578_1 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1388, permute_906);  view_1388 = permute_906 = None
        add_tensor_22: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_22, arg579_1);  mm_default_22 = arg579_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1389: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 576, 768]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_699: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg573_1, view_1389);  arg573_1 = view_1389 = None
        add_629: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_625, mul_699);  add_625 = mul_699 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_961: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_629, memory_format = torch.contiguous_format)
        var_mean_141 = torch.ops.aten.var_mean.correction(clone_961, [2], correction = 0, keepdim = True)
        getitem_290: "f32[8, 576, 1]" = var_mean_141[0]
        getitem_291: "f32[8, 576, 1]" = var_mean_141[1];  var_mean_141 = None
        sub_209: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_961, getitem_291);  clone_961 = getitem_291 = None
        add_630: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_290, 1e-06);  getitem_290 = None
        rsqrt_141: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_630);  add_630 = None
        mul_700: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_209, rsqrt_141);  sub_209 = rsqrt_141 = None
        mul_701: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_700, arg581_1);  mul_700 = arg581_1 = None
        add_631: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_701, arg582_1);  mul_701 = arg582_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1390: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_631, [4608, 768]);  add_631 = None
        permute_907: "f32[768, 2304]" = torch.ops.aten.permute.default(arg583_1, [1, 0]);  arg583_1 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1390, permute_907);  view_1390 = permute_907 = None
        add_tensor_21: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_21, arg584_1);  mm_default_21 = arg584_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1391: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 576, 2304]);  add_tensor_21 = None
        view_1392: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1391, [8, 576, 3, 16, 48]);  view_1391 = None
        permute_908: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1392, [2, 0, 3, 1, 4]);  view_1392 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_207: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_908, 0, 0)
        mul_702: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_207, 0.14433756729740643);  select_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_273: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_702, [8, 16, 576, 48]);  mul_702 = None
        clone_962: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_273, memory_format = torch.contiguous_format);  expand_273 = None
        view_1393: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_962, [128, 576, 48]);  clone_962 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_208: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_908, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_909: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_208, [0, 1, 3, 2]);  select_208 = None
        expand_274: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_909, [8, 16, 48, 576]);  permute_909 = None
        clone_963: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_274, memory_format = torch.contiguous_format);  expand_274 = None
        view_1394: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_963, [128, 48, 576]);  clone_963 = None
        bmm_136: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1393, view_1394);  view_1393 = view_1394 = None
        view_1395: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_136, [8, 16, 576, 576]);  bmm_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_910: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1395, [0, 2, 3, 1]);  view_1395 = None
        clone_964: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_910, memory_format = torch.contiguous_format);  permute_910 = None
        view_1396: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_964, [2654208, 16]);  clone_964 = None
        permute_911: "f32[16, 16]" = torch.ops.aten.permute.default(arg585_1, [1, 0]);  arg585_1 = None
        mm_136: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1396, permute_911);  view_1396 = permute_911 = None
        view_1397: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_136, [8, 576, 576, 16]);  mm_136 = None
        add_632: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1397, arg586_1);  view_1397 = arg586_1 = None
        permute_912: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_632, [0, 3, 1, 2]);  add_632 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_965: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_912, memory_format = torch.contiguous_format);  permute_912 = None
        amax_68: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_965, [-1], True)
        sub_210: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_965, amax_68);  clone_965 = amax_68 = None
        exp_68: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_210);  sub_210 = None
        sum_69: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_68, [-1], True)
        div_68: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_68, sum_69);  exp_68 = sum_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_913: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_68, [0, 2, 3, 1]);  div_68 = None
        clone_966: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_913, memory_format = torch.contiguous_format);  permute_913 = None
        view_1398: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_966, [2654208, 16]);  clone_966 = None
        permute_914: "f32[16, 16]" = torch.ops.aten.permute.default(arg587_1, [1, 0]);  arg587_1 = None
        mm_137: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1398, permute_914);  view_1398 = permute_914 = None
        view_1399: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_137, [8, 576, 576, 16]);  mm_137 = None
        add_633: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1399, arg588_1);  view_1399 = arg588_1 = None
        permute_915: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_633, [0, 3, 1, 2]);  add_633 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_275: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_915, [8, 16, 576, 576]);  permute_915 = None
        clone_968: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_275, memory_format = torch.contiguous_format);  expand_275 = None
        view_1400: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_968, [128, 576, 576]);  clone_968 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_209: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_908, 0, 2);  permute_908 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_276: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_209, [8, 16, 576, 48]);  select_209 = None
        clone_969: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_276, memory_format = torch.contiguous_format);  expand_276 = None
        view_1401: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_969, [128, 576, 48]);  clone_969 = None
        bmm_137: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1400, view_1401);  view_1400 = view_1401 = None
        view_1402: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_137, [8, 16, 576, 48]);  bmm_137 = None
        permute_916: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1402, [0, 2, 1, 3]);  view_1402 = None
        clone_970: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_916, memory_format = torch.contiguous_format);  permute_916 = None
        view_1403: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_970, [8, 576, 768]);  clone_970 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1404: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1403, [4608, 768]);  view_1403 = None
        permute_917: "f32[768, 768]" = torch.ops.aten.permute.default(arg589_1, [1, 0]);  arg589_1 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1404, permute_917);  view_1404 = permute_917 = None
        add_tensor_20: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_20, arg590_1);  mm_default_20 = arg590_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1405: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 576, 768]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_703: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg580_1, view_1405);  arg580_1 = view_1405 = None
        add_634: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_629, mul_703);  add_629 = mul_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_972: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_634, memory_format = torch.contiguous_format)
        var_mean_142 = torch.ops.aten.var_mean.correction(clone_972, [2], correction = 0, keepdim = True)
        getitem_292: "f32[8, 576, 1]" = var_mean_142[0]
        getitem_293: "f32[8, 576, 1]" = var_mean_142[1];  var_mean_142 = None
        sub_211: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_972, getitem_293);  clone_972 = getitem_293 = None
        add_635: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_292, 1e-06);  getitem_292 = None
        rsqrt_142: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_635);  add_635 = None
        mul_704: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_211, rsqrt_142);  sub_211 = rsqrt_142 = None
        mul_705: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_704, arg592_1);  mul_704 = arg592_1 = None
        add_636: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_705, arg593_1);  mul_705 = arg593_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1406: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_636, [4608, 768]);  add_636 = None
        permute_918: "f32[768, 3072]" = torch.ops.aten.permute.default(arg594_1, [1, 0]);  arg594_1 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1406, permute_918);  view_1406 = permute_918 = None
        add_tensor_19: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_19, arg595_1);  mm_default_19 = arg595_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1407: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 576, 3072]);  add_tensor_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_706: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1407, 0.5)
        mul_707: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1407, 0.7071067811865476);  view_1407 = None
        erf_70: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_707);  mul_707 = None
        add_637: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_70, 1);  erf_70 = None
        mul_708: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_706, add_637);  mul_706 = add_637 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1408: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_708, [4608, 3072]);  mul_708 = None
        permute_919: "f32[3072, 768]" = torch.ops.aten.permute.default(arg596_1, [1, 0]);  arg596_1 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1408, permute_919);  view_1408 = permute_919 = None
        add_tensor_18: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_18, arg597_1);  mm_default_18 = arg597_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1409: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 576, 768]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_709: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg591_1, view_1409);  arg591_1 = view_1409 = None
        add_638: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_634, mul_709);  add_634 = mul_709 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_975: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_638, memory_format = torch.contiguous_format)
        var_mean_143 = torch.ops.aten.var_mean.correction(clone_975, [2], correction = 0, keepdim = True)
        getitem_294: "f32[8, 576, 1]" = var_mean_143[0]
        getitem_295: "f32[8, 576, 1]" = var_mean_143[1];  var_mean_143 = None
        sub_212: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_975, getitem_295);  clone_975 = getitem_295 = None
        add_639: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_294, 1e-06);  getitem_294 = None
        rsqrt_143: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_639);  add_639 = None
        mul_710: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_212, rsqrt_143);  sub_212 = rsqrt_143 = None
        mul_711: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_710, arg599_1);  mul_710 = arg599_1 = None
        add_640: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_711, arg600_1);  mul_711 = arg600_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1410: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_640, [4608, 768]);  add_640 = None
        permute_920: "f32[768, 2304]" = torch.ops.aten.permute.default(arg601_1, [1, 0]);  arg601_1 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1410, permute_920);  view_1410 = permute_920 = None
        add_tensor_17: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_17, arg602_1);  mm_default_17 = arg602_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1411: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 576, 2304]);  add_tensor_17 = None
        view_1412: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1411, [8, 576, 3, 16, 48]);  view_1411 = None
        permute_921: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1412, [2, 0, 3, 1, 4]);  view_1412 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_210: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_921, 0, 0)
        mul_712: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_210, 0.14433756729740643);  select_210 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_277: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_712, [8, 16, 576, 48]);  mul_712 = None
        clone_976: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_277, memory_format = torch.contiguous_format);  expand_277 = None
        view_1413: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_976, [128, 576, 48]);  clone_976 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_211: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_921, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_922: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_211, [0, 1, 3, 2]);  select_211 = None
        expand_278: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_922, [8, 16, 48, 576]);  permute_922 = None
        clone_977: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_278, memory_format = torch.contiguous_format);  expand_278 = None
        view_1414: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_977, [128, 48, 576]);  clone_977 = None
        bmm_138: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1413, view_1414);  view_1413 = view_1414 = None
        view_1415: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_138, [8, 16, 576, 576]);  bmm_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_923: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1415, [0, 2, 3, 1]);  view_1415 = None
        clone_978: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_923, memory_format = torch.contiguous_format);  permute_923 = None
        view_1416: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_978, [2654208, 16]);  clone_978 = None
        permute_924: "f32[16, 16]" = torch.ops.aten.permute.default(arg603_1, [1, 0]);  arg603_1 = None
        mm_138: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1416, permute_924);  view_1416 = permute_924 = None
        view_1417: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_138, [8, 576, 576, 16]);  mm_138 = None
        add_641: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1417, arg604_1);  view_1417 = arg604_1 = None
        permute_925: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_641, [0, 3, 1, 2]);  add_641 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_979: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_925, memory_format = torch.contiguous_format);  permute_925 = None
        amax_69: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_979, [-1], True)
        sub_213: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_979, amax_69);  clone_979 = amax_69 = None
        exp_69: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_213);  sub_213 = None
        sum_70: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_69, [-1], True)
        div_69: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_69, sum_70);  exp_69 = sum_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_926: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_69, [0, 2, 3, 1]);  div_69 = None
        clone_980: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_926, memory_format = torch.contiguous_format);  permute_926 = None
        view_1418: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_980, [2654208, 16]);  clone_980 = None
        permute_927: "f32[16, 16]" = torch.ops.aten.permute.default(arg605_1, [1, 0]);  arg605_1 = None
        mm_139: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1418, permute_927);  view_1418 = permute_927 = None
        view_1419: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_139, [8, 576, 576, 16]);  mm_139 = None
        add_642: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1419, arg606_1);  view_1419 = arg606_1 = None
        permute_928: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_642, [0, 3, 1, 2]);  add_642 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_279: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_928, [8, 16, 576, 576]);  permute_928 = None
        clone_982: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_279, memory_format = torch.contiguous_format);  expand_279 = None
        view_1420: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_982, [128, 576, 576]);  clone_982 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_212: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_921, 0, 2);  permute_921 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_280: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_212, [8, 16, 576, 48]);  select_212 = None
        clone_983: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_280, memory_format = torch.contiguous_format);  expand_280 = None
        view_1421: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_983, [128, 576, 48]);  clone_983 = None
        bmm_139: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1420, view_1421);  view_1420 = view_1421 = None
        view_1422: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_139, [8, 16, 576, 48]);  bmm_139 = None
        permute_929: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1422, [0, 2, 1, 3]);  view_1422 = None
        clone_984: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_929, memory_format = torch.contiguous_format);  permute_929 = None
        view_1423: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_984, [8, 576, 768]);  clone_984 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1424: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1423, [4608, 768]);  view_1423 = None
        permute_930: "f32[768, 768]" = torch.ops.aten.permute.default(arg607_1, [1, 0]);  arg607_1 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1424, permute_930);  view_1424 = permute_930 = None
        add_tensor_16: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_16, arg608_1);  mm_default_16 = arg608_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1425: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 576, 768]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_713: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg598_1, view_1425);  arg598_1 = view_1425 = None
        add_643: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_638, mul_713);  add_638 = mul_713 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_986: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_643, memory_format = torch.contiguous_format)
        var_mean_144 = torch.ops.aten.var_mean.correction(clone_986, [2], correction = 0, keepdim = True)
        getitem_296: "f32[8, 576, 1]" = var_mean_144[0]
        getitem_297: "f32[8, 576, 1]" = var_mean_144[1];  var_mean_144 = None
        sub_214: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_986, getitem_297);  clone_986 = getitem_297 = None
        add_644: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_296, 1e-06);  getitem_296 = None
        rsqrt_144: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_644);  add_644 = None
        mul_714: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_214, rsqrt_144);  sub_214 = rsqrt_144 = None
        mul_715: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_714, arg610_1);  mul_714 = arg610_1 = None
        add_645: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_715, arg611_1);  mul_715 = arg611_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1426: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_645, [4608, 768]);  add_645 = None
        permute_931: "f32[768, 3072]" = torch.ops.aten.permute.default(arg612_1, [1, 0]);  arg612_1 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1426, permute_931);  view_1426 = permute_931 = None
        add_tensor_15: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_15, arg613_1);  mm_default_15 = arg613_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1427: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 576, 3072]);  add_tensor_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_716: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1427, 0.5)
        mul_717: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1427, 0.7071067811865476);  view_1427 = None
        erf_71: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_717);  mul_717 = None
        add_646: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_71, 1);  erf_71 = None
        mul_718: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_716, add_646);  mul_716 = add_646 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1428: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_718, [4608, 3072]);  mul_718 = None
        permute_932: "f32[3072, 768]" = torch.ops.aten.permute.default(arg614_1, [1, 0]);  arg614_1 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1428, permute_932);  view_1428 = permute_932 = None
        add_tensor_14: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_14, arg615_1);  mm_default_14 = arg615_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1429: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 576, 768]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_719: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg609_1, view_1429);  arg609_1 = view_1429 = None
        add_647: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_643, mul_719);  add_643 = mul_719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_989: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_647, memory_format = torch.contiguous_format)
        var_mean_145 = torch.ops.aten.var_mean.correction(clone_989, [2], correction = 0, keepdim = True)
        getitem_298: "f32[8, 576, 1]" = var_mean_145[0]
        getitem_299: "f32[8, 576, 1]" = var_mean_145[1];  var_mean_145 = None
        sub_215: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_989, getitem_299);  clone_989 = getitem_299 = None
        add_648: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_298, 1e-06);  getitem_298 = None
        rsqrt_145: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_648);  add_648 = None
        mul_720: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_215, rsqrt_145);  sub_215 = rsqrt_145 = None
        mul_721: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_720, arg617_1);  mul_720 = arg617_1 = None
        add_649: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_721, arg618_1);  mul_721 = arg618_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1430: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_649, [4608, 768]);  add_649 = None
        permute_933: "f32[768, 2304]" = torch.ops.aten.permute.default(arg619_1, [1, 0]);  arg619_1 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1430, permute_933);  view_1430 = permute_933 = None
        add_tensor_13: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_13, arg620_1);  mm_default_13 = arg620_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1431: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 576, 2304]);  add_tensor_13 = None
        view_1432: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1431, [8, 576, 3, 16, 48]);  view_1431 = None
        permute_934: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1432, [2, 0, 3, 1, 4]);  view_1432 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_213: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_934, 0, 0)
        mul_722: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_213, 0.14433756729740643);  select_213 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_281: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_722, [8, 16, 576, 48]);  mul_722 = None
        clone_990: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_281, memory_format = torch.contiguous_format);  expand_281 = None
        view_1433: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_990, [128, 576, 48]);  clone_990 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_214: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_934, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_935: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_214, [0, 1, 3, 2]);  select_214 = None
        expand_282: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_935, [8, 16, 48, 576]);  permute_935 = None
        clone_991: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_282, memory_format = torch.contiguous_format);  expand_282 = None
        view_1434: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_991, [128, 48, 576]);  clone_991 = None
        bmm_140: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1433, view_1434);  view_1433 = view_1434 = None
        view_1435: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_140, [8, 16, 576, 576]);  bmm_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_936: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1435, [0, 2, 3, 1]);  view_1435 = None
        clone_992: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_936, memory_format = torch.contiguous_format);  permute_936 = None
        view_1436: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_992, [2654208, 16]);  clone_992 = None
        permute_937: "f32[16, 16]" = torch.ops.aten.permute.default(arg621_1, [1, 0]);  arg621_1 = None
        mm_140: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1436, permute_937);  view_1436 = permute_937 = None
        view_1437: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_140, [8, 576, 576, 16]);  mm_140 = None
        add_650: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1437, arg622_1);  view_1437 = arg622_1 = None
        permute_938: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_650, [0, 3, 1, 2]);  add_650 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_993: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_938, memory_format = torch.contiguous_format);  permute_938 = None
        amax_70: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_993, [-1], True)
        sub_216: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_993, amax_70);  clone_993 = amax_70 = None
        exp_70: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_216);  sub_216 = None
        sum_71: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_70, [-1], True)
        div_70: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_70, sum_71);  exp_70 = sum_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_939: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_70, [0, 2, 3, 1]);  div_70 = None
        clone_994: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_939, memory_format = torch.contiguous_format);  permute_939 = None
        view_1438: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_994, [2654208, 16]);  clone_994 = None
        permute_940: "f32[16, 16]" = torch.ops.aten.permute.default(arg623_1, [1, 0]);  arg623_1 = None
        mm_141: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1438, permute_940);  view_1438 = permute_940 = None
        view_1439: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_141, [8, 576, 576, 16]);  mm_141 = None
        add_651: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1439, arg624_1);  view_1439 = arg624_1 = None
        permute_941: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_651, [0, 3, 1, 2]);  add_651 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_283: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_941, [8, 16, 576, 576]);  permute_941 = None
        clone_996: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_283, memory_format = torch.contiguous_format);  expand_283 = None
        view_1440: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_996, [128, 576, 576]);  clone_996 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_215: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_934, 0, 2);  permute_934 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_284: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_215, [8, 16, 576, 48]);  select_215 = None
        clone_997: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_284, memory_format = torch.contiguous_format);  expand_284 = None
        view_1441: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_997, [128, 576, 48]);  clone_997 = None
        bmm_141: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1440, view_1441);  view_1440 = view_1441 = None
        view_1442: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_141, [8, 16, 576, 48]);  bmm_141 = None
        permute_942: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1442, [0, 2, 1, 3]);  view_1442 = None
        clone_998: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_942, memory_format = torch.contiguous_format);  permute_942 = None
        view_1443: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_998, [8, 576, 768]);  clone_998 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1444: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1443, [4608, 768]);  view_1443 = None
        permute_943: "f32[768, 768]" = torch.ops.aten.permute.default(arg625_1, [1, 0]);  arg625_1 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1444, permute_943);  view_1444 = permute_943 = None
        add_tensor_12: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_12, arg626_1);  mm_default_12 = arg626_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1445: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 576, 768]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_723: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg616_1, view_1445);  arg616_1 = view_1445 = None
        add_652: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_647, mul_723);  add_647 = mul_723 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_1000: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_652, memory_format = torch.contiguous_format)
        var_mean_146 = torch.ops.aten.var_mean.correction(clone_1000, [2], correction = 0, keepdim = True)
        getitem_300: "f32[8, 576, 1]" = var_mean_146[0]
        getitem_301: "f32[8, 576, 1]" = var_mean_146[1];  var_mean_146 = None
        sub_217: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_1000, getitem_301);  clone_1000 = getitem_301 = None
        add_653: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_300, 1e-06);  getitem_300 = None
        rsqrt_146: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_653);  add_653 = None
        mul_724: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_217, rsqrt_146);  sub_217 = rsqrt_146 = None
        mul_725: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_724, arg628_1);  mul_724 = arg628_1 = None
        add_654: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_725, arg629_1);  mul_725 = arg629_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1446: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_654, [4608, 768]);  add_654 = None
        permute_944: "f32[768, 3072]" = torch.ops.aten.permute.default(arg630_1, [1, 0]);  arg630_1 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1446, permute_944);  view_1446 = permute_944 = None
        add_tensor_11: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_11, arg631_1);  mm_default_11 = arg631_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1447: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 576, 3072]);  add_tensor_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_726: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1447, 0.5)
        mul_727: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1447, 0.7071067811865476);  view_1447 = None
        erf_72: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_727);  mul_727 = None
        add_655: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_72, 1);  erf_72 = None
        mul_728: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_726, add_655);  mul_726 = add_655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1448: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_728, [4608, 3072]);  mul_728 = None
        permute_945: "f32[3072, 768]" = torch.ops.aten.permute.default(arg632_1, [1, 0]);  arg632_1 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1448, permute_945);  view_1448 = permute_945 = None
        add_tensor_10: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_10, arg633_1);  mm_default_10 = arg633_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1449: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 576, 768]);  add_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_729: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg627_1, view_1449);  arg627_1 = view_1449 = None
        add_656: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_652, mul_729);  add_652 = mul_729 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        clone_1003: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_656, memory_format = torch.contiguous_format)
        var_mean_147 = torch.ops.aten.var_mean.correction(clone_1003, [2], correction = 0, keepdim = True)
        getitem_302: "f32[8, 576, 1]" = var_mean_147[0]
        getitem_303: "f32[8, 576, 1]" = var_mean_147[1];  var_mean_147 = None
        sub_218: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_1003, getitem_303);  clone_1003 = getitem_303 = None
        add_657: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_302, 1e-06);  getitem_302 = None
        rsqrt_147: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_657);  add_657 = None
        mul_730: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_218, rsqrt_147);  sub_218 = rsqrt_147 = None
        mul_731: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_730, arg635_1);  mul_730 = arg635_1 = None
        add_658: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_731, arg636_1);  mul_731 = arg636_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1450: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_658, [4608, 768]);  add_658 = None
        permute_946: "f32[768, 2304]" = torch.ops.aten.permute.default(arg637_1, [1, 0]);  arg637_1 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[4608, 2304]" = torch.ops.aten.mm.default(view_1450, permute_946);  view_1450 = permute_946 = None
        add_tensor_9: "f32[4608, 2304]" = torch.ops.aten.add.Tensor(mm_default_9, arg638_1);  mm_default_9 = arg638_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:141 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1451: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 576, 2304]);  add_tensor_9 = None
        view_1452: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.reshape.default(view_1451, [8, 576, 3, 16, 48]);  view_1451 = None
        permute_947: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1452, [2, 0, 3, 1, 4]);  view_1452 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_216: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_947, 0, 0)
        mul_732: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_216, 0.14433756729740643);  select_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_285: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_732, [8, 16, 576, 48]);  mul_732 = None
        clone_1004: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_285, memory_format = torch.contiguous_format);  expand_285 = None
        view_1453: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_1004, [128, 576, 48]);  clone_1004 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_217: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_947, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:144 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_948: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_217, [0, 1, 3, 2]);  select_217 = None
        expand_286: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_948, [8, 16, 48, 576]);  permute_948 = None
        clone_1005: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_286, memory_format = torch.contiguous_format);  expand_286 = None
        view_1454: "f32[128, 48, 576]" = torch.ops.aten.reshape.default(clone_1005, [128, 48, 576]);  clone_1005 = None
        bmm_142: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1453, view_1454);  view_1453 = view_1454 = None
        view_1455: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_142, [8, 16, 576, 576]);  bmm_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:146 in forward, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_949: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1455, [0, 2, 3, 1]);  view_1455 = None
        clone_1006: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_949, memory_format = torch.contiguous_format);  permute_949 = None
        view_1456: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_1006, [2654208, 16]);  clone_1006 = None
        permute_950: "f32[16, 16]" = torch.ops.aten.permute.default(arg639_1, [1, 0]);  arg639_1 = None
        mm_142: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1456, permute_950);  view_1456 = permute_950 = None
        view_1457: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_142, [8, 576, 576, 16]);  mm_142 = None
        add_659: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1457, arg640_1);  view_1457 = arg640_1 = None
        permute_951: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_659, [0, 3, 1, 2]);  add_659 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:148 in forward, code: attn = attn.softmax(dim=-1)
        clone_1007: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_951, memory_format = torch.contiguous_format);  permute_951 = None
        amax_71: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_1007, [-1], True)
        sub_219: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_1007, amax_71);  clone_1007 = amax_71 = None
        exp_71: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_219);  sub_219 = None
        sum_72: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_71, [-1], True)
        div_71: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_71, sum_72);  exp_71 = sum_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:150 in forward, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_952: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_71, [0, 2, 3, 1]);  div_71 = None
        clone_1008: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_952, memory_format = torch.contiguous_format);  permute_952 = None
        view_1458: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_1008, [2654208, 16]);  clone_1008 = None
        permute_953: "f32[16, 16]" = torch.ops.aten.permute.default(arg641_1, [1, 0]);  arg641_1 = None
        mm_143: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1458, permute_953);  view_1458 = permute_953 = None
        view_1459: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_143, [8, 576, 576, 16]);  mm_143 = None
        add_660: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_1459, arg642_1);  view_1459 = arg642_1 = None
        permute_954: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_660, [0, 3, 1, 2]);  add_660 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_287: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(permute_954, [8, 16, 576, 576]);  permute_954 = None
        clone_1010: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_287, memory_format = torch.contiguous_format);  expand_287 = None
        view_1460: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_1010, [128, 576, 576]);  clone_1010 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:142 in forward, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        select_218: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_947, 0, 2);  permute_947 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:153 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        expand_288: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_218, [8, 16, 576, 48]);  select_218 = None
        clone_1011: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_288, memory_format = torch.contiguous_format);  expand_288 = None
        view_1461: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_1011, [128, 576, 48]);  clone_1011 = None
        bmm_143: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1460, view_1461);  view_1460 = view_1461 = None
        view_1462: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_143, [8, 16, 576, 48]);  bmm_143 = None
        permute_955: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_1462, [0, 2, 1, 3]);  view_1462 = None
        clone_1012: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_955, memory_format = torch.contiguous_format);  permute_955 = None
        view_1463: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(clone_1012, [8, 576, 768]);  clone_1012 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1464: "f32[4608, 768]" = torch.ops.aten.reshape.default(view_1463, [4608, 768]);  view_1463 = None
        permute_956: "f32[768, 768]" = torch.ops.aten.permute.default(arg643_1, [1, 0]);  arg643_1 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1464, permute_956);  view_1464 = permute_956 = None
        add_tensor_8: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_8, arg644_1);  mm_default_8 = arg644_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:154 in forward, code: x = self.proj(x)
        view_1465: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 576, 768]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:199 in forward, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        mul_733: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg634_1, view_1465);  arg634_1 = view_1465 = None
        add_661: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_656, mul_733);  add_656 = mul_733 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        clone_1014: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_661, memory_format = torch.contiguous_format)
        var_mean_148 = torch.ops.aten.var_mean.correction(clone_1014, [2], correction = 0, keepdim = True)
        getitem_304: "f32[8, 576, 1]" = var_mean_148[0]
        getitem_305: "f32[8, 576, 1]" = var_mean_148[1];  var_mean_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:424 in forward_features, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        expand_289: "f32[8, 1, 768]" = torch.ops.aten.expand.default(arg652_1, [8, -1, -1]);  arg652_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        sub_220: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_1014, getitem_305);  clone_1014 = getitem_305 = None
        add_662: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_304, 1e-06);  getitem_304 = None
        rsqrt_148: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_662);  add_662 = None
        mul_734: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_220, rsqrt_148);  sub_220 = rsqrt_148 = None
        mul_735: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_734, arg646_1);  mul_734 = arg646_1 = None
        add_663: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_735, arg647_1);  mul_735 = arg647_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1466: "f32[4608, 768]" = torch.ops.aten.reshape.default(add_663, [4608, 768]);  add_663 = None
        permute_957: "f32[768, 3072]" = torch.ops.aten.permute.default(arg648_1, [1, 0]);  arg648_1 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1466, permute_957);  view_1466 = permute_957 = None
        add_tensor_7: "f32[4608, 3072]" = torch.ops.aten.add.Tensor(mm_default_7, arg649_1);  mm_default_7 = arg649_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1467: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 576, 3072]);  add_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_736: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1467, 0.5)
        mul_737: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1467, 0.7071067811865476);  view_1467 = None
        erf_73: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_737);  mul_737 = None
        add_664: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_73, 1);  erf_73 = None
        mul_738: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_736, add_664);  mul_736 = add_664 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1468: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_738, [4608, 3072]);  mul_738 = None
        permute_958: "f32[3072, 768]" = torch.ops.aten.permute.default(arg650_1, [1, 0]);  arg650_1 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1468, permute_958);  view_1468 = permute_958 = None
        add_tensor_6: "f32[4608, 768]" = torch.ops.aten.add.Tensor(mm_default_6, arg651_1);  mm_default_6 = arg651_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1469: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 576, 768]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:200 in forward, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_739: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(arg645_1, view_1469);  arg645_1 = view_1469 = None
        add_665: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_661, mul_739);  add_661 = mul_739 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:111 in forward, code: u = torch.cat((x_cls, x), dim=1)
        cat_3: "f32[8, 577, 768]" = torch.ops.aten.cat.default([expand_289, add_665], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:112 in forward, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        var_mean_149 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
        getitem_306: "f32[8, 577, 1]" = var_mean_149[0]
        getitem_307: "f32[8, 577, 1]" = var_mean_149[1];  var_mean_149 = None
        sub_221: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(cat_3, getitem_307);  cat_3 = getitem_307 = None
        add_666: "f32[8, 577, 1]" = torch.ops.aten.add.Tensor(getitem_306, 1e-06);  getitem_306 = None
        rsqrt_149: "f32[8, 577, 1]" = torch.ops.aten.rsqrt.default(add_666);  add_666 = None
        mul_740: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(sub_221, rsqrt_149);  sub_221 = rsqrt_149 = None
        mul_741: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_740, arg654_1);  mul_740 = arg654_1 = None
        add_667: "f32[8, 577, 768]" = torch.ops.aten.add.Tensor(mul_741, arg655_1);  mul_741 = arg655_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:48 in forward, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        select_219: "f32[8, 768]" = torch.ops.aten.select.int(add_667, 1, 0)
        permute_959: "f32[768, 768]" = torch.ops.aten.permute.default(arg656_1, [1, 0]);  arg656_1 = None
        addmm_301: "f32[8, 768]" = torch.ops.aten.addmm.default(arg657_1, select_219, permute_959);  arg657_1 = select_219 = permute_959 = None
        unsqueeze_2: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_301, 1);  addmm_301 = None
        view_1470: "f32[8, 1, 16, 48]" = torch.ops.aten.reshape.default(unsqueeze_2, [8, 1, 16, 48]);  unsqueeze_2 = None
        permute_960: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_1470, [0, 2, 1, 3]);  view_1470 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:49 in forward, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_1471: "f32[4616, 768]" = torch.ops.aten.reshape.default(add_667, [4616, 768])
        permute_961: "f32[768, 768]" = torch.ops.aten.permute.default(arg658_1, [1, 0]);  arg658_1 = None
        addmm_302: "f32[4616, 768]" = torch.ops.aten.addmm.default(arg659_1, view_1471, permute_961);  arg659_1 = view_1471 = permute_961 = None
        view_1472: "f32[8, 577, 768]" = torch.ops.aten.reshape.default(addmm_302, [8, 577, 768]);  addmm_302 = None
        view_1473: "f32[8, 577, 16, 48]" = torch.ops.aten.reshape.default(view_1472, [8, 577, 16, 48]);  view_1472 = None
        permute_962: "f32[8, 16, 577, 48]" = torch.ops.aten.permute.default(view_1473, [0, 2, 1, 3]);  view_1473 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:50 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_1474: "f32[4616, 768]" = torch.ops.aten.reshape.default(add_667, [4616, 768]);  add_667 = None
        permute_963: "f32[768, 768]" = torch.ops.aten.permute.default(arg660_1, [1, 0]);  arg660_1 = None
        addmm_303: "f32[4616, 768]" = torch.ops.aten.addmm.default(arg661_1, view_1474, permute_963);  arg661_1 = view_1474 = permute_963 = None
        view_1475: "f32[8, 577, 768]" = torch.ops.aten.reshape.default(addmm_303, [8, 577, 768]);  addmm_303 = None
        view_1476: "f32[8, 577, 16, 48]" = torch.ops.aten.reshape.default(view_1475, [8, 577, 16, 48]);  view_1475 = None
        permute_964: "f32[8, 16, 577, 48]" = torch.ops.aten.permute.default(view_1476, [0, 2, 1, 3]);  view_1476 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:53 in forward, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_960, permute_962, permute_964, None, False);  permute_960 = permute_962 = permute_964 = None
        getitem_308: "f32[8, 16, 1, 48]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:64 in forward, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
        permute_965: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_308, [0, 2, 1, 3]);  getitem_308 = None
        view_1477: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(permute_965, [8, 1, 768]);  permute_965 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:65 in forward, code: x_cls = self.proj(x_cls)
        view_1478: "f32[8, 768]" = torch.ops.aten.reshape.default(view_1477, [8, 768]);  view_1477 = None
        permute_966: "f32[768, 768]" = torch.ops.aten.permute.default(arg662_1, [1, 0]);  arg662_1 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[8, 768]" = torch.ops.aten.mm.default(view_1478, permute_966);  view_1478 = permute_966 = None
        add_tensor_5: "f32[8, 768]" = torch.ops.aten.add.Tensor(mm_default_5, arg663_1);  mm_default_5 = arg663_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:65 in forward, code: x_cls = self.proj(x_cls)
        view_1479: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 1, 768]);  add_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:112 in forward, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        mul_742: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(arg653_1, view_1479);  arg653_1 = view_1479 = None
        add_668: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(expand_289, mul_742);  expand_289 = mul_742 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:113 in forward, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        var_mean_150 = torch.ops.aten.var_mean.correction(add_668, [2], correction = 0, keepdim = True)
        getitem_312: "f32[8, 1, 1]" = var_mean_150[0]
        getitem_313: "f32[8, 1, 1]" = var_mean_150[1];  var_mean_150 = None
        sub_222: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(add_668, getitem_313);  getitem_313 = None
        add_669: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_312, 1e-06);  getitem_312 = None
        rsqrt_150: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_669);  add_669 = None
        mul_743: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_222, rsqrt_150);  sub_222 = rsqrt_150 = None
        mul_744: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(mul_743, arg665_1);  mul_743 = arg665_1 = None
        add_670: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(mul_744, arg666_1);  mul_744 = arg666_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1480: "f32[8, 768]" = torch.ops.aten.reshape.default(add_670, [8, 768]);  add_670 = None
        permute_967: "f32[768, 3072]" = torch.ops.aten.permute.default(arg667_1, [1, 0]);  arg667_1 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[8, 3072]" = torch.ops.aten.mm.default(view_1480, permute_967);  view_1480 = permute_967 = None
        add_tensor_4: "f32[8, 3072]" = torch.ops.aten.add.Tensor(mm_default_4, arg668_1);  mm_default_4 = arg668_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1481: "f32[8, 1, 3072]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 1, 3072]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_745: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_1481, 0.5)
        mul_746: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_1481, 0.7071067811865476);  view_1481 = None
        erf_74: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_746);  mul_746 = None
        add_671: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_74, 1);  erf_74 = None
        mul_747: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_745, add_671);  mul_745 = add_671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1482: "f32[8, 3072]" = torch.ops.aten.reshape.default(mul_747, [8, 3072]);  mul_747 = None
        permute_968: "f32[3072, 768]" = torch.ops.aten.permute.default(arg669_1, [1, 0]);  arg669_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[8, 768]" = torch.ops.aten.mm.default(view_1482, permute_968);  view_1482 = permute_968 = None
        add_tensor_3: "f32[8, 768]" = torch.ops.aten.add.Tensor(mm_default_3, arg670_1);  mm_default_3 = arg670_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1483: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 1, 768]);  add_tensor_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:113 in forward, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        mul_748: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(arg664_1, view_1483);  arg664_1 = view_1483 = None
        add_672: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(add_668, mul_748);  add_668 = mul_748 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:111 in forward, code: u = torch.cat((x_cls, x), dim=1)
        cat_4: "f32[8, 577, 768]" = torch.ops.aten.cat.default([add_672, add_665], 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:112 in forward, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        var_mean_151 = torch.ops.aten.var_mean.correction(cat_4, [2], correction = 0, keepdim = True)
        getitem_314: "f32[8, 577, 1]" = var_mean_151[0]
        getitem_315: "f32[8, 577, 1]" = var_mean_151[1];  var_mean_151 = None
        sub_223: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(cat_4, getitem_315);  cat_4 = getitem_315 = None
        add_673: "f32[8, 577, 1]" = torch.ops.aten.add.Tensor(getitem_314, 1e-06);  getitem_314 = None
        rsqrt_151: "f32[8, 577, 1]" = torch.ops.aten.rsqrt.default(add_673);  add_673 = None
        mul_749: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(sub_223, rsqrt_151);  sub_223 = rsqrt_151 = None
        mul_750: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_749, arg672_1);  mul_749 = arg672_1 = None
        add_674: "f32[8, 577, 768]" = torch.ops.aten.add.Tensor(mul_750, arg673_1);  mul_750 = arg673_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:48 in forward, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        select_220: "f32[8, 768]" = torch.ops.aten.select.int(add_674, 1, 0)
        permute_969: "f32[768, 768]" = torch.ops.aten.permute.default(arg674_1, [1, 0]);  arg674_1 = None
        addmm_307: "f32[8, 768]" = torch.ops.aten.addmm.default(arg675_1, select_220, permute_969);  arg675_1 = select_220 = permute_969 = None
        unsqueeze_3: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_307, 1);  addmm_307 = None
        view_1484: "f32[8, 1, 16, 48]" = torch.ops.aten.reshape.default(unsqueeze_3, [8, 1, 16, 48]);  unsqueeze_3 = None
        permute_970: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_1484, [0, 2, 1, 3]);  view_1484 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:49 in forward, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_1485: "f32[4616, 768]" = torch.ops.aten.reshape.default(add_674, [4616, 768])
        permute_971: "f32[768, 768]" = torch.ops.aten.permute.default(arg676_1, [1, 0]);  arg676_1 = None
        addmm_308: "f32[4616, 768]" = torch.ops.aten.addmm.default(arg677_1, view_1485, permute_971);  arg677_1 = view_1485 = permute_971 = None
        view_1486: "f32[8, 577, 768]" = torch.ops.aten.reshape.default(addmm_308, [8, 577, 768]);  addmm_308 = None
        view_1487: "f32[8, 577, 16, 48]" = torch.ops.aten.reshape.default(view_1486, [8, 577, 16, 48]);  view_1486 = None
        permute_972: "f32[8, 16, 577, 48]" = torch.ops.aten.permute.default(view_1487, [0, 2, 1, 3]);  view_1487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:50 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        view_1488: "f32[4616, 768]" = torch.ops.aten.reshape.default(add_674, [4616, 768]);  add_674 = None
        permute_973: "f32[768, 768]" = torch.ops.aten.permute.default(arg678_1, [1, 0]);  arg678_1 = None
        addmm_309: "f32[4616, 768]" = torch.ops.aten.addmm.default(arg679_1, view_1488, permute_973);  arg679_1 = view_1488 = permute_973 = None
        view_1489: "f32[8, 577, 768]" = torch.ops.aten.reshape.default(addmm_309, [8, 577, 768]);  addmm_309 = None
        view_1490: "f32[8, 577, 16, 48]" = torch.ops.aten.reshape.default(view_1489, [8, 577, 16, 48]);  view_1489 = None
        permute_974: "f32[8, 16, 577, 48]" = torch.ops.aten.permute.default(view_1490, [0, 2, 1, 3]);  view_1490 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:53 in forward, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_970, permute_972, permute_974, None, False);  permute_970 = permute_972 = permute_974 = None
        getitem_316: "f32[8, 16, 1, 48]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:64 in forward, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
        permute_975: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_316, [0, 2, 1, 3]);  getitem_316 = None
        view_1491: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(permute_975, [8, 1, 768]);  permute_975 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:65 in forward, code: x_cls = self.proj(x_cls)
        view_1492: "f32[8, 768]" = torch.ops.aten.reshape.default(view_1491, [8, 768]);  view_1491 = None
        permute_976: "f32[768, 768]" = torch.ops.aten.permute.default(arg680_1, [1, 0]);  arg680_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[8, 768]" = torch.ops.aten.mm.default(view_1492, permute_976);  view_1492 = permute_976 = None
        add_tensor_2: "f32[8, 768]" = torch.ops.aten.add.Tensor(mm_default_2, arg681_1);  mm_default_2 = arg681_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:65 in forward, code: x_cls = self.proj(x_cls)
        view_1493: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 1, 768]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:112 in forward, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        mul_751: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(arg671_1, view_1493);  arg671_1 = view_1493 = None
        add_675: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(add_672, mul_751);  add_672 = mul_751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:113 in forward, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        var_mean_152 = torch.ops.aten.var_mean.correction(add_675, [2], correction = 0, keepdim = True)
        getitem_320: "f32[8, 1, 1]" = var_mean_152[0]
        getitem_321: "f32[8, 1, 1]" = var_mean_152[1];  var_mean_152 = None
        sub_224: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(add_675, getitem_321);  getitem_321 = None
        add_676: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_320, 1e-06);  getitem_320 = None
        rsqrt_152: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_676);  add_676 = None
        mul_752: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_224, rsqrt_152);  sub_224 = rsqrt_152 = None
        mul_753: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(mul_752, arg683_1);  mul_752 = arg683_1 = None
        add_677: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(mul_753, arg684_1);  mul_753 = arg684_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1494: "f32[8, 768]" = torch.ops.aten.reshape.default(add_677, [8, 768]);  add_677 = None
        permute_977: "f32[768, 3072]" = torch.ops.aten.permute.default(arg685_1, [1, 0]);  arg685_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[8, 3072]" = torch.ops.aten.mm.default(view_1494, permute_977);  view_1494 = permute_977 = None
        add_tensor_1: "f32[8, 3072]" = torch.ops.aten.add.Tensor(mm_default_1, arg686_1);  mm_default_1 = arg686_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1495: "f32[8, 1, 3072]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 1, 3072]);  add_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_754: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_1495, 0.5)
        mul_755: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_1495, 0.7071067811865476);  view_1495 = None
        erf_75: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_755);  mul_755 = None
        add_678: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_75, 1);  erf_75 = None
        mul_756: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_754, add_678);  mul_754 = add_678 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1496: "f32[8, 3072]" = torch.ops.aten.reshape.default(mul_756, [8, 3072]);  mul_756 = None
        permute_978: "f32[3072, 768]" = torch.ops.aten.permute.default(arg687_1, [1, 0]);  arg687_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[8, 768]" = torch.ops.aten.mm.default(view_1496, permute_978);  view_1496 = permute_978 = None
        add_tensor: "f32[8, 768]" = torch.ops.aten.add.Tensor(mm_default, arg688_1);  mm_default = arg688_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1497: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(add_tensor, [8, 1, 768]);  add_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:113 in forward, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        mul_757: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(arg682_1, view_1497);  arg682_1 = view_1497 = None
        add_679: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(add_675, mul_757);  add_675 = mul_757 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:427 in forward_features, code: x = torch.cat((cls_tokens, x), dim=1)
        cat_5: "f32[8, 577, 768]" = torch.ops.aten.cat.default([add_679, add_665], 1);  add_679 = add_665 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:428 in forward_features, code: x = self.norm(x)
        var_mean_153 = torch.ops.aten.var_mean.correction(cat_5, [2], correction = 0, keepdim = True)
        getitem_322: "f32[8, 577, 1]" = var_mean_153[0]
        getitem_323: "f32[8, 577, 1]" = var_mean_153[1];  var_mean_153 = None
        sub_225: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(cat_5, getitem_323);  cat_5 = getitem_323 = None
        add_680: "f32[8, 577, 1]" = torch.ops.aten.add.Tensor(getitem_322, 1e-06);  getitem_322 = None
        rsqrt_153: "f32[8, 577, 1]" = torch.ops.aten.rsqrt.default(add_680);  add_680 = None
        mul_758: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(sub_225, rsqrt_153);  sub_225 = rsqrt_153 = None
        mul_759: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_758, arg689_1);  mul_758 = arg689_1 = None
        add_681: "f32[8, 577, 768]" = torch.ops.aten.add.Tensor(mul_759, arg690_1);  mul_759 = arg690_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:433 in forward_head, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        select_221: "f32[8, 768]" = torch.ops.aten.select.int(add_681, 1, 0);  add_681 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:434 in forward_head, code: x = self.head_drop(x)
        clone_1023: "f32[8, 768]" = torch.ops.aten.clone.default(select_221);  select_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cait.py:435 in forward_head, code: return x if pre_logits else self.head(x)
        permute_979: "f32[768, 1000]" = torch.ops.aten.permute.default(arg691_1, [1, 0]);  arg691_1 = None
        addmm_313: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg692_1, clone_1023, permute_979);  arg692_1 = clone_1023 = permute_979 = None
        return (addmm_313,)
        