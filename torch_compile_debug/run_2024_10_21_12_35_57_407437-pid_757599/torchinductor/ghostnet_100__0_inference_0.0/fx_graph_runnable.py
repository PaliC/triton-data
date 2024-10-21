
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.fallback_random = True
torch._inductor.config.triton.cudagraphs = True
torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.6.0.dev20241021+cu118
# torch cuda version: 11.8
# torch git version: 5553778a0095e7234b2cd0874c2ff4dcc0216323


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Fri_Jan__6_16:45:21_PST_2023 
# Cuda compilation tools, release 12.0, V12.0.140 
# Build cuda_12.0.r12.0/compiler.32267302_0 

# GPU Hardware Info: 
# NVIDIA H100 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1):
        convolution_95 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_183 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_80 = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_80 = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_247 = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_641 = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642 = torch.ops.aten.unsqueeze.default(mul_247, -1);  mul_247 = None
        unsqueeze_643 = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80 = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_641);  convolution_95 = unsqueeze_641 = None
        mul_248 = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_645 = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_645);  mul_248 = unsqueeze_645 = None
        unsqueeze_646 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_647 = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_184 = torch.ops.aten.add.Tensor(mul_249, unsqueeze_647);  mul_249 = unsqueeze_647 = None
        relu_42 = torch.ops.aten.relu.default(add_184);  add_184 = None
        convolution_96 = torch.ops.aten.convolution.default(relu_42, arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg6_1 = None
        add_185 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_81 = torch.ops.aten.sqrt.default(add_185);  add_185 = None
        reciprocal_81 = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_250 = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_649 = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650 = torch.ops.aten.unsqueeze.default(mul_250, -1);  mul_250 = None
        unsqueeze_651 = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81 = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_649);  convolution_96 = unsqueeze_649 = None
        mul_251 = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_653 = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_252 = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_653);  mul_251 = unsqueeze_653 = None
        unsqueeze_654 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_655 = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_186 = torch.ops.aten.add.Tensor(mul_252, unsqueeze_655);  mul_252 = unsqueeze_655 = None
        relu_43 = torch.ops.aten.relu.default(add_186);  add_186 = None
        convolution_97 = torch.ops.aten.convolution.default(relu_43, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  arg11_1 = None
        add_187 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_82 = torch.ops.aten.sqrt.default(add_187);  add_187 = None
        reciprocal_82 = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_253 = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_656 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_657 = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        unsqueeze_658 = torch.ops.aten.unsqueeze.default(mul_253, -1);  mul_253 = None
        unsqueeze_659 = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        sub_82 = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_657);  convolution_97 = unsqueeze_657 = None
        mul_254 = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_661 = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_255 = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_661);  mul_254 = unsqueeze_661 = None
        unsqueeze_662 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_663 = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_188 = torch.ops.aten.add.Tensor(mul_255, unsqueeze_663);  mul_255 = unsqueeze_663 = None
        relu_44 = torch.ops.aten.relu.default(add_188);  add_188 = None
        cat_32 = torch.ops.aten.cat.default([relu_43, relu_44], 1);  relu_43 = relu_44 = None
        convolution_98 = torch.ops.aten.convolution.default(cat_32, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_32 = arg16_1 = None
        add_189 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_83 = torch.ops.aten.sqrt.default(add_189);  add_189 = None
        reciprocal_83 = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_256 = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_664 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_665 = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        unsqueeze_666 = torch.ops.aten.unsqueeze.default(mul_256, -1);  mul_256 = None
        unsqueeze_667 = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        sub_83 = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_665);  convolution_98 = unsqueeze_665 = None
        mul_257 = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_669 = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_258 = torch.ops.aten.mul.Tensor(mul_257, unsqueeze_669);  mul_257 = unsqueeze_669 = None
        unsqueeze_670 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_671 = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_190 = torch.ops.aten.add.Tensor(mul_258, unsqueeze_671);  mul_258 = unsqueeze_671 = None
        convolution_99 = torch.ops.aten.convolution.default(add_190, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  arg21_1 = None
        add_191 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_84 = torch.ops.aten.sqrt.default(add_191);  add_191 = None
        reciprocal_84 = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_259 = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_672 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_673 = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        unsqueeze_674 = torch.ops.aten.unsqueeze.default(mul_259, -1);  mul_259 = None
        unsqueeze_675 = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        sub_84 = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_673);  convolution_99 = unsqueeze_673 = None
        mul_260 = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_677 = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_261 = torch.ops.aten.mul.Tensor(mul_260, unsqueeze_677);  mul_260 = unsqueeze_677 = None
        unsqueeze_678 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_679 = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_192 = torch.ops.aten.add.Tensor(mul_261, unsqueeze_679);  mul_261 = unsqueeze_679 = None
        cat_33 = torch.ops.aten.cat.default([add_190, add_192], 1);  add_190 = add_192 = None
        add_193 = torch.ops.aten.add.Tensor(cat_33, relu_42);  cat_33 = relu_42 = None
        convolution_100 = torch.ops.aten.convolution.default(add_193, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg26_1 = None
        add_194 = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_85 = torch.ops.aten.sqrt.default(add_194);  add_194 = None
        reciprocal_85 = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_262 = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_681 = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682 = torch.ops.aten.unsqueeze.default(mul_262, -1);  mul_262 = None
        unsqueeze_683 = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85 = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_681);  convolution_100 = unsqueeze_681 = None
        mul_263 = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_685 = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_264 = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_685);  mul_263 = unsqueeze_685 = None
        unsqueeze_686 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_687 = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_195 = torch.ops.aten.add.Tensor(mul_264, unsqueeze_687);  mul_264 = unsqueeze_687 = None
        relu_45 = torch.ops.aten.relu.default(add_195);  add_195 = None
        convolution_101 = torch.ops.aten.convolution.default(relu_45, arg31_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 24);  arg31_1 = None
        add_196 = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_86 = torch.ops.aten.sqrt.default(add_196);  add_196 = None
        reciprocal_86 = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_265 = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688 = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_689 = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690 = torch.ops.aten.unsqueeze.default(mul_265, -1);  mul_265 = None
        unsqueeze_691 = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86 = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_689);  convolution_101 = unsqueeze_689 = None
        mul_266 = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_693 = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_267 = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_693);  mul_266 = unsqueeze_693 = None
        unsqueeze_694 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_695 = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_197 = torch.ops.aten.add.Tensor(mul_267, unsqueeze_695);  mul_267 = unsqueeze_695 = None
        relu_46 = torch.ops.aten.relu.default(add_197);  add_197 = None
        cat_34 = torch.ops.aten.cat.default([relu_45, relu_46], 1);  relu_45 = relu_46 = None
        convolution_102 = torch.ops.aten.convolution.default(cat_34, arg36_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 48);  cat_34 = arg36_1 = None
        add_198 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_87 = torch.ops.aten.sqrt.default(add_198);  add_198 = None
        reciprocal_87 = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_268 = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_697 = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698 = torch.ops.aten.unsqueeze.default(mul_268, -1);  mul_268 = None
        unsqueeze_699 = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87 = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_697);  convolution_102 = unsqueeze_697 = None
        mul_269 = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_701 = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_270 = torch.ops.aten.mul.Tensor(mul_269, unsqueeze_701);  mul_269 = unsqueeze_701 = None
        unsqueeze_702 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_703 = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_199 = torch.ops.aten.add.Tensor(mul_270, unsqueeze_703);  mul_270 = unsqueeze_703 = None
        convolution_103 = torch.ops.aten.convolution.default(add_199, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_199 = arg41_1 = None
        add_200 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_88 = torch.ops.aten.sqrt.default(add_200);  add_200 = None
        reciprocal_88 = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_271 = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_705 = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706 = torch.ops.aten.unsqueeze.default(mul_271, -1);  mul_271 = None
        unsqueeze_707 = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88 = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_705);  convolution_103 = unsqueeze_705 = None
        mul_272 = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_709 = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_273 = torch.ops.aten.mul.Tensor(mul_272, unsqueeze_709);  mul_272 = unsqueeze_709 = None
        unsqueeze_710 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_711 = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_201 = torch.ops.aten.add.Tensor(mul_273, unsqueeze_711);  mul_273 = unsqueeze_711 = None
        convolution_104 = torch.ops.aten.convolution.default(add_201, arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 12);  arg46_1 = None
        add_202 = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_89 = torch.ops.aten.sqrt.default(add_202);  add_202 = None
        reciprocal_89 = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_274 = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_713 = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714 = torch.ops.aten.unsqueeze.default(mul_274, -1);  mul_274 = None
        unsqueeze_715 = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89 = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_713);  convolution_104 = unsqueeze_713 = None
        mul_275 = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_717 = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_276 = torch.ops.aten.mul.Tensor(mul_275, unsqueeze_717);  mul_275 = unsqueeze_717 = None
        unsqueeze_718 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_719 = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_203 = torch.ops.aten.add.Tensor(mul_276, unsqueeze_719);  mul_276 = unsqueeze_719 = None
        cat_35 = torch.ops.aten.cat.default([add_201, add_203], 1);  add_201 = add_203 = None
        convolution_105 = torch.ops.aten.convolution.default(add_193, arg51_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16);  add_193 = arg51_1 = None
        add_204 = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_90 = torch.ops.aten.sqrt.default(add_204);  add_204 = None
        reciprocal_90 = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_277 = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_721 = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722 = torch.ops.aten.unsqueeze.default(mul_277, -1);  mul_277 = None
        unsqueeze_723 = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90 = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_721);  convolution_105 = unsqueeze_721 = None
        mul_278 = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_725 = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_725);  mul_278 = unsqueeze_725 = None
        unsqueeze_726 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_727 = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_205 = torch.ops.aten.add.Tensor(mul_279, unsqueeze_727);  mul_279 = unsqueeze_727 = None
        convolution_106 = torch.ops.aten.convolution.default(add_205, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_205 = arg56_1 = None
        add_206 = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_91 = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_91 = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_280 = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_729 = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730 = torch.ops.aten.unsqueeze.default(mul_280, -1);  mul_280 = None
        unsqueeze_731 = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91 = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_729);  convolution_106 = unsqueeze_729 = None
        mul_281 = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_733 = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_282 = torch.ops.aten.mul.Tensor(mul_281, unsqueeze_733);  mul_281 = unsqueeze_733 = None
        unsqueeze_734 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_735 = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_207 = torch.ops.aten.add.Tensor(mul_282, unsqueeze_735);  mul_282 = unsqueeze_735 = None
        add_208 = torch.ops.aten.add.Tensor(cat_35, add_207);  cat_35 = add_207 = None
        convolution_107 = torch.ops.aten.convolution.default(add_208, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg61_1 = None
        add_209 = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_92 = torch.ops.aten.sqrt.default(add_209);  add_209 = None
        reciprocal_92 = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_283 = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_736 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_737 = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        unsqueeze_738 = torch.ops.aten.unsqueeze.default(mul_283, -1);  mul_283 = None
        unsqueeze_739 = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        sub_92 = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_737);  convolution_107 = unsqueeze_737 = None
        mul_284 = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_741 = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_285 = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_741);  mul_284 = unsqueeze_741 = None
        unsqueeze_742 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_743 = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_210 = torch.ops.aten.add.Tensor(mul_285, unsqueeze_743);  mul_285 = unsqueeze_743 = None
        relu_47 = torch.ops.aten.relu.default(add_210);  add_210 = None
        convolution_108 = torch.ops.aten.convolution.default(relu_47, arg66_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 36);  arg66_1 = None
        add_211 = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_93 = torch.ops.aten.sqrt.default(add_211);  add_211 = None
        reciprocal_93 = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_286 = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_744 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_745 = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        unsqueeze_746 = torch.ops.aten.unsqueeze.default(mul_286, -1);  mul_286 = None
        unsqueeze_747 = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        sub_93 = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_745);  convolution_108 = unsqueeze_745 = None
        mul_287 = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_749 = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_288 = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_749);  mul_287 = unsqueeze_749 = None
        unsqueeze_750 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_751 = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_212 = torch.ops.aten.add.Tensor(mul_288, unsqueeze_751);  mul_288 = unsqueeze_751 = None
        relu_48 = torch.ops.aten.relu.default(add_212);  add_212 = None
        cat_36 = torch.ops.aten.cat.default([relu_47, relu_48], 1);  relu_47 = relu_48 = None
        convolution_109 = torch.ops.aten.convolution.default(cat_36, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_36 = arg71_1 = None
        add_213 = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_94 = torch.ops.aten.sqrt.default(add_213);  add_213 = None
        reciprocal_94 = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_289 = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_752 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_753 = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        unsqueeze_754 = torch.ops.aten.unsqueeze.default(mul_289, -1);  mul_289 = None
        unsqueeze_755 = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        sub_94 = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_753);  convolution_109 = unsqueeze_753 = None
        mul_290 = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_757 = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, unsqueeze_757);  mul_290 = unsqueeze_757 = None
        unsqueeze_758 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_759 = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_214 = torch.ops.aten.add.Tensor(mul_291, unsqueeze_759);  mul_291 = unsqueeze_759 = None
        convolution_110 = torch.ops.aten.convolution.default(add_214, arg76_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 12);  arg76_1 = None
        add_215 = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_95 = torch.ops.aten.sqrt.default(add_215);  add_215 = None
        reciprocal_95 = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_292 = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_760 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_761 = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        unsqueeze_762 = torch.ops.aten.unsqueeze.default(mul_292, -1);  mul_292 = None
        unsqueeze_763 = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        sub_95 = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_761);  convolution_110 = unsqueeze_761 = None
        mul_293 = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_765 = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_294 = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_765);  mul_293 = unsqueeze_765 = None
        unsqueeze_766 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_767 = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_216 = torch.ops.aten.add.Tensor(mul_294, unsqueeze_767);  mul_294 = unsqueeze_767 = None
        cat_37 = torch.ops.aten.cat.default([add_214, add_216], 1);  add_214 = add_216 = None
        add_217 = torch.ops.aten.add.Tensor(cat_37, add_208);  cat_37 = add_208 = None
        convolution_111 = torch.ops.aten.convolution.default(add_217, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg81_1 = None
        add_218 = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_96 = torch.ops.aten.sqrt.default(add_218);  add_218 = None
        reciprocal_96 = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_295 = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_768 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_769 = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        unsqueeze_770 = torch.ops.aten.unsqueeze.default(mul_295, -1);  mul_295 = None
        unsqueeze_771 = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        sub_96 = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_769);  convolution_111 = unsqueeze_769 = None
        mul_296 = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_773 = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_773);  mul_296 = unsqueeze_773 = None
        unsqueeze_774 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_775 = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_219 = torch.ops.aten.add.Tensor(mul_297, unsqueeze_775);  mul_297 = unsqueeze_775 = None
        relu_49 = torch.ops.aten.relu.default(add_219);  add_219 = None
        convolution_112 = torch.ops.aten.convolution.default(relu_49, arg86_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 36);  arg86_1 = None
        add_220 = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_97 = torch.ops.aten.sqrt.default(add_220);  add_220 = None
        reciprocal_97 = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_298 = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_776 = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_777 = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        unsqueeze_778 = torch.ops.aten.unsqueeze.default(mul_298, -1);  mul_298 = None
        unsqueeze_779 = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        sub_97 = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_777);  convolution_112 = unsqueeze_777 = None
        mul_299 = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_781 = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_300 = torch.ops.aten.mul.Tensor(mul_299, unsqueeze_781);  mul_299 = unsqueeze_781 = None
        unsqueeze_782 = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_783 = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_221 = torch.ops.aten.add.Tensor(mul_300, unsqueeze_783);  mul_300 = unsqueeze_783 = None
        relu_50 = torch.ops.aten.relu.default(add_221);  add_221 = None
        cat_38 = torch.ops.aten.cat.default([relu_49, relu_50], 1);  relu_49 = relu_50 = None
        convolution_113 = torch.ops.aten.convolution.default(cat_38, arg91_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 72);  cat_38 = arg91_1 = None
        add_222 = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_98 = torch.ops.aten.sqrt.default(add_222);  add_222 = None
        reciprocal_98 = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_301 = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_784 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_785 = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        unsqueeze_786 = torch.ops.aten.unsqueeze.default(mul_301, -1);  mul_301 = None
        unsqueeze_787 = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        sub_98 = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_785);  convolution_113 = unsqueeze_785 = None
        mul_302 = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_789 = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_303 = torch.ops.aten.mul.Tensor(mul_302, unsqueeze_789);  mul_302 = unsqueeze_789 = None
        unsqueeze_790 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_791 = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_223 = torch.ops.aten.add.Tensor(mul_303, unsqueeze_791);  mul_303 = unsqueeze_791 = None
        mean_8 = torch.ops.aten.mean.dim(add_223, [2, 3], True)
        convolution_114 = torch.ops.aten.convolution.default(mean_8, arg96_1, arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_8 = arg96_1 = arg97_1 = None
        relu_51 = torch.ops.aten.relu.default(convolution_114);  convolution_114 = None
        convolution_115 = torch.ops.aten.convolution.default(relu_51, arg98_1, arg99_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_51 = arg98_1 = arg99_1 = None
        add_224 = torch.ops.aten.add.Tensor(convolution_115, 3);  convolution_115 = None
        clamp_min_7 = torch.ops.aten.clamp_min.default(add_224, 0);  add_224 = None
        clamp_max_7 = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
        div_7 = torch.ops.aten.div.Tensor(clamp_max_7, 6);  clamp_max_7 = None
        mul_304 = torch.ops.aten.mul.Tensor(add_223, div_7);  add_223 = div_7 = None
        convolution_116 = torch.ops.aten.convolution.default(mul_304, arg100_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_304 = arg100_1 = None
        add_225 = torch.ops.aten.add.Tensor(arg102_1, 1e-05);  arg102_1 = None
        sqrt_99 = torch.ops.aten.sqrt.default(add_225);  add_225 = None
        reciprocal_99 = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_305 = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_792 = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_793 = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        unsqueeze_794 = torch.ops.aten.unsqueeze.default(mul_305, -1);  mul_305 = None
        unsqueeze_795 = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        sub_99 = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_793);  convolution_116 = unsqueeze_793 = None
        mul_306 = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796 = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_797 = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_307 = torch.ops.aten.mul.Tensor(mul_306, unsqueeze_797);  mul_306 = unsqueeze_797 = None
        unsqueeze_798 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_799 = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_226 = torch.ops.aten.add.Tensor(mul_307, unsqueeze_799);  mul_307 = unsqueeze_799 = None
        convolution_117 = torch.ops.aten.convolution.default(add_226, arg105_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 20);  arg105_1 = None
        add_227 = torch.ops.aten.add.Tensor(arg107_1, 1e-05);  arg107_1 = None
        sqrt_100 = torch.ops.aten.sqrt.default(add_227);  add_227 = None
        reciprocal_100 = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_308 = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_800 = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_801 = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        unsqueeze_802 = torch.ops.aten.unsqueeze.default(mul_308, -1);  mul_308 = None
        unsqueeze_803 = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        sub_100 = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_801);  convolution_117 = unsqueeze_801 = None
        mul_309 = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804 = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_805 = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_310 = torch.ops.aten.mul.Tensor(mul_309, unsqueeze_805);  mul_309 = unsqueeze_805 = None
        unsqueeze_806 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_807 = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_228 = torch.ops.aten.add.Tensor(mul_310, unsqueeze_807);  mul_310 = unsqueeze_807 = None
        cat_39 = torch.ops.aten.cat.default([add_226, add_228], 1);  add_226 = add_228 = None
        convolution_118 = torch.ops.aten.convolution.default(add_217, arg110_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 24);  add_217 = arg110_1 = None
        add_229 = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
        sqrt_101 = torch.ops.aten.sqrt.default(add_229);  add_229 = None
        reciprocal_101 = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_311 = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_808 = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_809 = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        unsqueeze_810 = torch.ops.aten.unsqueeze.default(mul_311, -1);  mul_311 = None
        unsqueeze_811 = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        sub_101 = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_809);  convolution_118 = unsqueeze_809 = None
        mul_312 = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812 = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_813 = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_313 = torch.ops.aten.mul.Tensor(mul_312, unsqueeze_813);  mul_312 = unsqueeze_813 = None
        unsqueeze_814 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_815 = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_230 = torch.ops.aten.add.Tensor(mul_313, unsqueeze_815);  mul_313 = unsqueeze_815 = None
        convolution_119 = torch.ops.aten.convolution.default(add_230, arg115_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_230 = arg115_1 = None
        add_231 = torch.ops.aten.add.Tensor(arg117_1, 1e-05);  arg117_1 = None
        sqrt_102 = torch.ops.aten.sqrt.default(add_231);  add_231 = None
        reciprocal_102 = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_314 = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_816 = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_817 = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        unsqueeze_818 = torch.ops.aten.unsqueeze.default(mul_314, -1);  mul_314 = None
        unsqueeze_819 = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        sub_102 = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_817);  convolution_119 = unsqueeze_817 = None
        mul_315 = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820 = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_821 = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_316 = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_821);  mul_315 = unsqueeze_821 = None
        unsqueeze_822 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_823 = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_232 = torch.ops.aten.add.Tensor(mul_316, unsqueeze_823);  mul_316 = unsqueeze_823 = None
        add_233 = torch.ops.aten.add.Tensor(cat_39, add_232);  cat_39 = add_232 = None
        convolution_120 = torch.ops.aten.convolution.default(add_233, arg120_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg120_1 = None
        add_234 = torch.ops.aten.add.Tensor(arg122_1, 1e-05);  arg122_1 = None
        sqrt_103 = torch.ops.aten.sqrt.default(add_234);  add_234 = None
        reciprocal_103 = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_317 = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_824 = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_825 = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        unsqueeze_826 = torch.ops.aten.unsqueeze.default(mul_317, -1);  mul_317 = None
        unsqueeze_827 = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        sub_103 = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_825);  convolution_120 = unsqueeze_825 = None
        mul_318 = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828 = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_829 = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_319 = torch.ops.aten.mul.Tensor(mul_318, unsqueeze_829);  mul_318 = unsqueeze_829 = None
        unsqueeze_830 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_831 = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_235 = torch.ops.aten.add.Tensor(mul_319, unsqueeze_831);  mul_319 = unsqueeze_831 = None
        relu_52 = torch.ops.aten.relu.default(add_235);  add_235 = None
        convolution_121 = torch.ops.aten.convolution.default(relu_52, arg125_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 60);  arg125_1 = None
        add_236 = torch.ops.aten.add.Tensor(arg127_1, 1e-05);  arg127_1 = None
        sqrt_104 = torch.ops.aten.sqrt.default(add_236);  add_236 = None
        reciprocal_104 = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_320 = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_832 = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_833 = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        unsqueeze_834 = torch.ops.aten.unsqueeze.default(mul_320, -1);  mul_320 = None
        unsqueeze_835 = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        sub_104 = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_833);  convolution_121 = unsqueeze_833 = None
        mul_321 = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836 = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_837 = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_322 = torch.ops.aten.mul.Tensor(mul_321, unsqueeze_837);  mul_321 = unsqueeze_837 = None
        unsqueeze_838 = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_839 = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_237 = torch.ops.aten.add.Tensor(mul_322, unsqueeze_839);  mul_322 = unsqueeze_839 = None
        relu_53 = torch.ops.aten.relu.default(add_237);  add_237 = None
        cat_40 = torch.ops.aten.cat.default([relu_52, relu_53], 1);  relu_52 = relu_53 = None
        mean_9 = torch.ops.aten.mean.dim(cat_40, [2, 3], True)
        convolution_122 = torch.ops.aten.convolution.default(mean_9, arg130_1, arg131_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_9 = arg130_1 = arg131_1 = None
        relu_54 = torch.ops.aten.relu.default(convolution_122);  convolution_122 = None
        convolution_123 = torch.ops.aten.convolution.default(relu_54, arg132_1, arg133_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_54 = arg132_1 = arg133_1 = None
        add_238 = torch.ops.aten.add.Tensor(convolution_123, 3);  convolution_123 = None
        clamp_min_8 = torch.ops.aten.clamp_min.default(add_238, 0);  add_238 = None
        clamp_max_8 = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
        div_8 = torch.ops.aten.div.Tensor(clamp_max_8, 6);  clamp_max_8 = None
        mul_323 = torch.ops.aten.mul.Tensor(cat_40, div_8);  cat_40 = div_8 = None
        convolution_124 = torch.ops.aten.convolution.default(mul_323, arg134_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_323 = arg134_1 = None
        add_239 = torch.ops.aten.add.Tensor(arg136_1, 1e-05);  arg136_1 = None
        sqrt_105 = torch.ops.aten.sqrt.default(add_239);  add_239 = None
        reciprocal_105 = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_324 = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_840 = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_841 = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        unsqueeze_842 = torch.ops.aten.unsqueeze.default(mul_324, -1);  mul_324 = None
        unsqueeze_843 = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        sub_105 = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_841);  convolution_124 = unsqueeze_841 = None
        mul_325 = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844 = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_845 = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_845);  mul_325 = unsqueeze_845 = None
        unsqueeze_846 = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_847 = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_240 = torch.ops.aten.add.Tensor(mul_326, unsqueeze_847);  mul_326 = unsqueeze_847 = None
        convolution_125 = torch.ops.aten.convolution.default(add_240, arg139_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 20);  arg139_1 = None
        add_241 = torch.ops.aten.add.Tensor(arg141_1, 1e-05);  arg141_1 = None
        sqrt_106 = torch.ops.aten.sqrt.default(add_241);  add_241 = None
        reciprocal_106 = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_327 = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_848 = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_849 = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        unsqueeze_850 = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
        unsqueeze_851 = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        sub_106 = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_849);  convolution_125 = unsqueeze_849 = None
        mul_328 = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_853 = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_329 = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_853);  mul_328 = unsqueeze_853 = None
        unsqueeze_854 = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_855 = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_242 = torch.ops.aten.add.Tensor(mul_329, unsqueeze_855);  mul_329 = unsqueeze_855 = None
        cat_41 = torch.ops.aten.cat.default([add_240, add_242], 1);  add_240 = add_242 = None
        add_243 = torch.ops.aten.add.Tensor(cat_41, add_233);  cat_41 = add_233 = None
        convolution_126 = torch.ops.aten.convolution.default(add_243, arg144_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg144_1 = None
        add_244 = torch.ops.aten.add.Tensor(arg146_1, 1e-05);  arg146_1 = None
        sqrt_107 = torch.ops.aten.sqrt.default(add_244);  add_244 = None
        reciprocal_107 = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_330 = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_856 = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_857 = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        unsqueeze_858 = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
        unsqueeze_859 = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        sub_107 = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_857);  convolution_126 = unsqueeze_857 = None
        mul_331 = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860 = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_861 = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_861);  mul_331 = unsqueeze_861 = None
        unsqueeze_862 = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_863 = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_245 = torch.ops.aten.add.Tensor(mul_332, unsqueeze_863);  mul_332 = unsqueeze_863 = None
        relu_55 = torch.ops.aten.relu.default(add_245);  add_245 = None
        convolution_127 = torch.ops.aten.convolution.default(relu_55, arg149_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  arg149_1 = None
        add_246 = torch.ops.aten.add.Tensor(arg151_1, 1e-05);  arg151_1 = None
        sqrt_108 = torch.ops.aten.sqrt.default(add_246);  add_246 = None
        reciprocal_108 = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_333 = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_864 = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_865 = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        unsqueeze_866 = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
        unsqueeze_867 = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        sub_108 = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_865);  convolution_127 = unsqueeze_865 = None
        mul_334 = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868 = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_869 = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_869);  mul_334 = unsqueeze_869 = None
        unsqueeze_870 = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_871 = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_247 = torch.ops.aten.add.Tensor(mul_335, unsqueeze_871);  mul_335 = unsqueeze_871 = None
        relu_56 = torch.ops.aten.relu.default(add_247);  add_247 = None
        cat_42 = torch.ops.aten.cat.default([relu_55, relu_56], 1);  relu_55 = relu_56 = None
        convolution_128 = torch.ops.aten.convolution.default(cat_42, arg154_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240);  cat_42 = arg154_1 = None
        add_248 = torch.ops.aten.add.Tensor(arg156_1, 1e-05);  arg156_1 = None
        sqrt_109 = torch.ops.aten.sqrt.default(add_248);  add_248 = None
        reciprocal_109 = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_336 = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_872 = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_873 = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        unsqueeze_874 = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
        unsqueeze_875 = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        sub_109 = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_873);  convolution_128 = unsqueeze_873 = None
        mul_337 = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876 = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_877 = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_338 = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_877);  mul_337 = unsqueeze_877 = None
        unsqueeze_878 = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_879 = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_249 = torch.ops.aten.add.Tensor(mul_338, unsqueeze_879);  mul_338 = unsqueeze_879 = None
        convolution_129 = torch.ops.aten.convolution.default(add_249, arg159_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_249 = arg159_1 = None
        add_250 = torch.ops.aten.add.Tensor(arg161_1, 1e-05);  arg161_1 = None
        sqrt_110 = torch.ops.aten.sqrt.default(add_250);  add_250 = None
        reciprocal_110 = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_339 = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_880 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_881 = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        unsqueeze_882 = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
        unsqueeze_883 = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        sub_110 = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_881);  convolution_129 = unsqueeze_881 = None
        mul_340 = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884 = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_885 = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_885);  mul_340 = unsqueeze_885 = None
        unsqueeze_886 = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_887 = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_251 = torch.ops.aten.add.Tensor(mul_341, unsqueeze_887);  mul_341 = unsqueeze_887 = None
        convolution_130 = torch.ops.aten.convolution.default(add_251, arg164_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40);  arg164_1 = None
        add_252 = torch.ops.aten.add.Tensor(arg166_1, 1e-05);  arg166_1 = None
        sqrt_111 = torch.ops.aten.sqrt.default(add_252);  add_252 = None
        reciprocal_111 = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_342 = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_888 = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_889 = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        unsqueeze_890 = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
        unsqueeze_891 = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        sub_111 = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_889);  convolution_130 = unsqueeze_889 = None
        mul_343 = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892 = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_893 = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_893);  mul_343 = unsqueeze_893 = None
        unsqueeze_894 = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_895 = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_253 = torch.ops.aten.add.Tensor(mul_344, unsqueeze_895);  mul_344 = unsqueeze_895 = None
        cat_43 = torch.ops.aten.cat.default([add_251, add_253], 1);  add_251 = add_253 = None
        convolution_131 = torch.ops.aten.convolution.default(add_243, arg169_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 40);  add_243 = arg169_1 = None
        add_254 = torch.ops.aten.add.Tensor(arg171_1, 1e-05);  arg171_1 = None
        sqrt_112 = torch.ops.aten.sqrt.default(add_254);  add_254 = None
        reciprocal_112 = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_345 = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_896 = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_897 = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        unsqueeze_898 = torch.ops.aten.unsqueeze.default(mul_345, -1);  mul_345 = None
        unsqueeze_899 = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        sub_112 = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_897);  convolution_131 = unsqueeze_897 = None
        mul_346 = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900 = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_901 = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_901);  mul_346 = unsqueeze_901 = None
        unsqueeze_902 = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_903 = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_255 = torch.ops.aten.add.Tensor(mul_347, unsqueeze_903);  mul_347 = unsqueeze_903 = None
        convolution_132 = torch.ops.aten.convolution.default(add_255, arg174_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_255 = arg174_1 = None
        add_256 = torch.ops.aten.add.Tensor(arg176_1, 1e-05);  arg176_1 = None
        sqrt_113 = torch.ops.aten.sqrt.default(add_256);  add_256 = None
        reciprocal_113 = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_348 = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_904 = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_905 = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        unsqueeze_906 = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
        unsqueeze_907 = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        sub_113 = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_905);  convolution_132 = unsqueeze_905 = None
        mul_349 = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908 = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_909 = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_350 = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_909);  mul_349 = unsqueeze_909 = None
        unsqueeze_910 = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_911 = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_257 = torch.ops.aten.add.Tensor(mul_350, unsqueeze_911);  mul_350 = unsqueeze_911 = None
        add_258 = torch.ops.aten.add.Tensor(cat_43, add_257);  cat_43 = add_257 = None
        convolution_133 = torch.ops.aten.convolution.default(add_258, arg179_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg179_1 = None
        add_259 = torch.ops.aten.add.Tensor(arg181_1, 1e-05);  arg181_1 = None
        sqrt_114 = torch.ops.aten.sqrt.default(add_259);  add_259 = None
        reciprocal_114 = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_351 = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_912 = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_913 = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        unsqueeze_914 = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
        unsqueeze_915 = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        sub_114 = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_913);  convolution_133 = unsqueeze_913 = None
        mul_352 = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916 = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_917 = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_917);  mul_352 = unsqueeze_917 = None
        unsqueeze_918 = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_919 = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_260 = torch.ops.aten.add.Tensor(mul_353, unsqueeze_919);  mul_353 = unsqueeze_919 = None
        relu_57 = torch.ops.aten.relu.default(add_260);  add_260 = None
        convolution_134 = torch.ops.aten.convolution.default(relu_57, arg184_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 100);  arg184_1 = None
        add_261 = torch.ops.aten.add.Tensor(arg186_1, 1e-05);  arg186_1 = None
        sqrt_115 = torch.ops.aten.sqrt.default(add_261);  add_261 = None
        reciprocal_115 = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_354 = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_920 = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_921 = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        unsqueeze_922 = torch.ops.aten.unsqueeze.default(mul_354, -1);  mul_354 = None
        unsqueeze_923 = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        sub_115 = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_921);  convolution_134 = unsqueeze_921 = None
        mul_355 = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924 = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_925 = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_356 = torch.ops.aten.mul.Tensor(mul_355, unsqueeze_925);  mul_355 = unsqueeze_925 = None
        unsqueeze_926 = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_927 = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_262 = torch.ops.aten.add.Tensor(mul_356, unsqueeze_927);  mul_356 = unsqueeze_927 = None
        relu_58 = torch.ops.aten.relu.default(add_262);  add_262 = None
        cat_44 = torch.ops.aten.cat.default([relu_57, relu_58], 1);  relu_57 = relu_58 = None
        convolution_135 = torch.ops.aten.convolution.default(cat_44, arg189_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_44 = arg189_1 = None
        add_263 = torch.ops.aten.add.Tensor(arg191_1, 1e-05);  arg191_1 = None
        sqrt_116 = torch.ops.aten.sqrt.default(add_263);  add_263 = None
        reciprocal_116 = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_357 = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        unsqueeze_928 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_929 = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
        unsqueeze_930 = torch.ops.aten.unsqueeze.default(mul_357, -1);  mul_357 = None
        unsqueeze_931 = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
        sub_116 = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_929);  convolution_135 = unsqueeze_929 = None
        mul_358 = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
        unsqueeze_932 = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_933 = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
        mul_359 = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_933);  mul_358 = unsqueeze_933 = None
        unsqueeze_934 = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
        unsqueeze_935 = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
        add_264 = torch.ops.aten.add.Tensor(mul_359, unsqueeze_935);  mul_359 = unsqueeze_935 = None
        convolution_136 = torch.ops.aten.convolution.default(add_264, arg194_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40);  arg194_1 = None
        add_265 = torch.ops.aten.add.Tensor(arg196_1, 1e-05);  arg196_1 = None
        sqrt_117 = torch.ops.aten.sqrt.default(add_265);  add_265 = None
        reciprocal_117 = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_360 = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        unsqueeze_936 = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_937 = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
        unsqueeze_938 = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
        unsqueeze_939 = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
        sub_117 = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_937);  convolution_136 = unsqueeze_937 = None
        mul_361 = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
        unsqueeze_940 = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_941 = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
        mul_362 = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_941);  mul_361 = unsqueeze_941 = None
        unsqueeze_942 = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_943 = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
        add_266 = torch.ops.aten.add.Tensor(mul_362, unsqueeze_943);  mul_362 = unsqueeze_943 = None
        cat_45 = torch.ops.aten.cat.default([add_264, add_266], 1);  add_264 = add_266 = None
        add_267 = torch.ops.aten.add.Tensor(cat_45, add_258);  cat_45 = add_258 = None
        convolution_137 = torch.ops.aten.convolution.default(add_267, arg199_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg199_1 = None
        add_268 = torch.ops.aten.add.Tensor(arg201_1, 1e-05);  arg201_1 = None
        sqrt_118 = torch.ops.aten.sqrt.default(add_268);  add_268 = None
        reciprocal_118 = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_363 = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        unsqueeze_944 = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_945 = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
        unsqueeze_946 = torch.ops.aten.unsqueeze.default(mul_363, -1);  mul_363 = None
        unsqueeze_947 = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
        sub_118 = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_945);  convolution_137 = unsqueeze_945 = None
        mul_364 = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
        unsqueeze_948 = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_949 = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
        mul_365 = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_949);  mul_364 = unsqueeze_949 = None
        unsqueeze_950 = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_951 = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
        add_269 = torch.ops.aten.add.Tensor(mul_365, unsqueeze_951);  mul_365 = unsqueeze_951 = None
        relu_59 = torch.ops.aten.relu.default(add_269);  add_269 = None
        convolution_138 = torch.ops.aten.convolution.default(relu_59, arg204_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 92);  arg204_1 = None
        add_270 = torch.ops.aten.add.Tensor(arg206_1, 1e-05);  arg206_1 = None
        sqrt_119 = torch.ops.aten.sqrt.default(add_270);  add_270 = None
        reciprocal_119 = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_366 = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        unsqueeze_952 = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_953 = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
        unsqueeze_954 = torch.ops.aten.unsqueeze.default(mul_366, -1);  mul_366 = None
        unsqueeze_955 = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
        sub_119 = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_953);  convolution_138 = unsqueeze_953 = None
        mul_367 = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
        unsqueeze_956 = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_957 = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
        mul_368 = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_957);  mul_367 = unsqueeze_957 = None
        unsqueeze_958 = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_959 = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
        add_271 = torch.ops.aten.add.Tensor(mul_368, unsqueeze_959);  mul_368 = unsqueeze_959 = None
        relu_60 = torch.ops.aten.relu.default(add_271);  add_271 = None
        cat_46 = torch.ops.aten.cat.default([relu_59, relu_60], 1);  relu_59 = relu_60 = None
        convolution_139 = torch.ops.aten.convolution.default(cat_46, arg209_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_46 = arg209_1 = None
        add_272 = torch.ops.aten.add.Tensor(arg211_1, 1e-05);  arg211_1 = None
        sqrt_120 = torch.ops.aten.sqrt.default(add_272);  add_272 = None
        reciprocal_120 = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_369 = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        unsqueeze_960 = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_961 = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
        unsqueeze_962 = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
        unsqueeze_963 = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
        sub_120 = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_961);  convolution_139 = unsqueeze_961 = None
        mul_370 = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
        unsqueeze_964 = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_965 = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_965);  mul_370 = unsqueeze_965 = None
        unsqueeze_966 = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_967 = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
        add_273 = torch.ops.aten.add.Tensor(mul_371, unsqueeze_967);  mul_371 = unsqueeze_967 = None
        convolution_140 = torch.ops.aten.convolution.default(add_273, arg214_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40);  arg214_1 = None
        add_274 = torch.ops.aten.add.Tensor(arg216_1, 1e-05);  arg216_1 = None
        sqrt_121 = torch.ops.aten.sqrt.default(add_274);  add_274 = None
        reciprocal_121 = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_372 = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        unsqueeze_968 = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_969 = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
        unsqueeze_970 = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
        unsqueeze_971 = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
        sub_121 = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_969);  convolution_140 = unsqueeze_969 = None
        mul_373 = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
        unsqueeze_972 = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_973 = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_973);  mul_373 = unsqueeze_973 = None
        unsqueeze_974 = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_975 = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
        add_275 = torch.ops.aten.add.Tensor(mul_374, unsqueeze_975);  mul_374 = unsqueeze_975 = None
        cat_47 = torch.ops.aten.cat.default([add_273, add_275], 1);  add_273 = add_275 = None
        add_276 = torch.ops.aten.add.Tensor(cat_47, add_267);  cat_47 = add_267 = None
        convolution_141 = torch.ops.aten.convolution.default(add_276, arg219_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg219_1 = None
        add_277 = torch.ops.aten.add.Tensor(arg221_1, 1e-05);  arg221_1 = None
        sqrt_122 = torch.ops.aten.sqrt.default(add_277);  add_277 = None
        reciprocal_122 = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_375 = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        unsqueeze_976 = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_977 = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
        unsqueeze_978 = torch.ops.aten.unsqueeze.default(mul_375, -1);  mul_375 = None
        unsqueeze_979 = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
        sub_122 = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_977);  convolution_141 = unsqueeze_977 = None
        mul_376 = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
        unsqueeze_980 = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_981 = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
        mul_377 = torch.ops.aten.mul.Tensor(mul_376, unsqueeze_981);  mul_376 = unsqueeze_981 = None
        unsqueeze_982 = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_983 = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
        add_278 = torch.ops.aten.add.Tensor(mul_377, unsqueeze_983);  mul_377 = unsqueeze_983 = None
        relu_61 = torch.ops.aten.relu.default(add_278);  add_278 = None
        convolution_142 = torch.ops.aten.convolution.default(relu_61, arg224_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 92);  arg224_1 = None
        add_279 = torch.ops.aten.add.Tensor(arg226_1, 1e-05);  arg226_1 = None
        sqrt_123 = torch.ops.aten.sqrt.default(add_279);  add_279 = None
        reciprocal_123 = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_378 = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        unsqueeze_984 = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_985 = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
        unsqueeze_986 = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
        unsqueeze_987 = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
        sub_123 = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_985);  convolution_142 = unsqueeze_985 = None
        mul_379 = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
        unsqueeze_988 = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_989 = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_989);  mul_379 = unsqueeze_989 = None
        unsqueeze_990 = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_991 = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
        add_280 = torch.ops.aten.add.Tensor(mul_380, unsqueeze_991);  mul_380 = unsqueeze_991 = None
        relu_62 = torch.ops.aten.relu.default(add_280);  add_280 = None
        cat_48 = torch.ops.aten.cat.default([relu_61, relu_62], 1);  relu_61 = relu_62 = None
        convolution_143 = torch.ops.aten.convolution.default(cat_48, arg229_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_48 = arg229_1 = None
        add_281 = torch.ops.aten.add.Tensor(arg231_1, 1e-05);  arg231_1 = None
        sqrt_124 = torch.ops.aten.sqrt.default(add_281);  add_281 = None
        reciprocal_124 = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
        mul_381 = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
        unsqueeze_992 = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_993 = torch.ops.aten.unsqueeze.default(unsqueeze_992, -1);  unsqueeze_992 = None
        unsqueeze_994 = torch.ops.aten.unsqueeze.default(mul_381, -1);  mul_381 = None
        unsqueeze_995 = torch.ops.aten.unsqueeze.default(unsqueeze_994, -1);  unsqueeze_994 = None
        sub_124 = torch.ops.aten.sub.Tensor(convolution_143, unsqueeze_993);  convolution_143 = unsqueeze_993 = None
        mul_382 = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_995);  sub_124 = unsqueeze_995 = None
        unsqueeze_996 = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_997 = torch.ops.aten.unsqueeze.default(unsqueeze_996, -1);  unsqueeze_996 = None
        mul_383 = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_997);  mul_382 = unsqueeze_997 = None
        unsqueeze_998 = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_999 = torch.ops.aten.unsqueeze.default(unsqueeze_998, -1);  unsqueeze_998 = None
        add_282 = torch.ops.aten.add.Tensor(mul_383, unsqueeze_999);  mul_383 = unsqueeze_999 = None
        convolution_144 = torch.ops.aten.convolution.default(add_282, arg234_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40);  arg234_1 = None
        add_283 = torch.ops.aten.add.Tensor(arg236_1, 1e-05);  arg236_1 = None
        sqrt_125 = torch.ops.aten.sqrt.default(add_283);  add_283 = None
        reciprocal_125 = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
        mul_384 = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
        unsqueeze_1000 = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_1001 = torch.ops.aten.unsqueeze.default(unsqueeze_1000, -1);  unsqueeze_1000 = None
        unsqueeze_1002 = torch.ops.aten.unsqueeze.default(mul_384, -1);  mul_384 = None
        unsqueeze_1003 = torch.ops.aten.unsqueeze.default(unsqueeze_1002, -1);  unsqueeze_1002 = None
        sub_125 = torch.ops.aten.sub.Tensor(convolution_144, unsqueeze_1001);  convolution_144 = unsqueeze_1001 = None
        mul_385 = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_1003);  sub_125 = unsqueeze_1003 = None
        unsqueeze_1004 = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_1005 = torch.ops.aten.unsqueeze.default(unsqueeze_1004, -1);  unsqueeze_1004 = None
        mul_386 = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_1005);  mul_385 = unsqueeze_1005 = None
        unsqueeze_1006 = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_1007 = torch.ops.aten.unsqueeze.default(unsqueeze_1006, -1);  unsqueeze_1006 = None
        add_284 = torch.ops.aten.add.Tensor(mul_386, unsqueeze_1007);  mul_386 = unsqueeze_1007 = None
        cat_49 = torch.ops.aten.cat.default([add_282, add_284], 1);  add_282 = add_284 = None
        add_285 = torch.ops.aten.add.Tensor(cat_49, add_276);  cat_49 = add_276 = None
        convolution_145 = torch.ops.aten.convolution.default(add_285, arg239_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg239_1 = None
        add_286 = torch.ops.aten.add.Tensor(arg241_1, 1e-05);  arg241_1 = None
        sqrt_126 = torch.ops.aten.sqrt.default(add_286);  add_286 = None
        reciprocal_126 = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
        mul_387 = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
        unsqueeze_1008 = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_1009 = torch.ops.aten.unsqueeze.default(unsqueeze_1008, -1);  unsqueeze_1008 = None
        unsqueeze_1010 = torch.ops.aten.unsqueeze.default(mul_387, -1);  mul_387 = None
        unsqueeze_1011 = torch.ops.aten.unsqueeze.default(unsqueeze_1010, -1);  unsqueeze_1010 = None
        sub_126 = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_1009);  convolution_145 = unsqueeze_1009 = None
        mul_388 = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_1011);  sub_126 = unsqueeze_1011 = None
        unsqueeze_1012 = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_1013 = torch.ops.aten.unsqueeze.default(unsqueeze_1012, -1);  unsqueeze_1012 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_1013);  mul_388 = unsqueeze_1013 = None
        unsqueeze_1014 = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
        unsqueeze_1015 = torch.ops.aten.unsqueeze.default(unsqueeze_1014, -1);  unsqueeze_1014 = None
        add_287 = torch.ops.aten.add.Tensor(mul_389, unsqueeze_1015);  mul_389 = unsqueeze_1015 = None
        relu_63 = torch.ops.aten.relu.default(add_287);  add_287 = None
        convolution_146 = torch.ops.aten.convolution.default(relu_63, arg244_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 240);  arg244_1 = None
        add_288 = torch.ops.aten.add.Tensor(arg246_1, 1e-05);  arg246_1 = None
        sqrt_127 = torch.ops.aten.sqrt.default(add_288);  add_288 = None
        reciprocal_127 = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
        mul_390 = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
        unsqueeze_1016 = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_1017 = torch.ops.aten.unsqueeze.default(unsqueeze_1016, -1);  unsqueeze_1016 = None
        unsqueeze_1018 = torch.ops.aten.unsqueeze.default(mul_390, -1);  mul_390 = None
        unsqueeze_1019 = torch.ops.aten.unsqueeze.default(unsqueeze_1018, -1);  unsqueeze_1018 = None
        sub_127 = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_1017);  convolution_146 = unsqueeze_1017 = None
        mul_391 = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_1019);  sub_127 = unsqueeze_1019 = None
        unsqueeze_1020 = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_1021 = torch.ops.aten.unsqueeze.default(unsqueeze_1020, -1);  unsqueeze_1020 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_391, unsqueeze_1021);  mul_391 = unsqueeze_1021 = None
        unsqueeze_1022 = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_1023 = torch.ops.aten.unsqueeze.default(unsqueeze_1022, -1);  unsqueeze_1022 = None
        add_289 = torch.ops.aten.add.Tensor(mul_392, unsqueeze_1023);  mul_392 = unsqueeze_1023 = None
        relu_64 = torch.ops.aten.relu.default(add_289);  add_289 = None
        cat_50 = torch.ops.aten.cat.default([relu_63, relu_64], 1);  relu_63 = relu_64 = None
        mean_10 = torch.ops.aten.mean.dim(cat_50, [2, 3], True)
        convolution_147 = torch.ops.aten.convolution.default(mean_10, arg249_1, arg250_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_10 = arg249_1 = arg250_1 = None
        relu_65 = torch.ops.aten.relu.default(convolution_147);  convolution_147 = None
        convolution_148 = torch.ops.aten.convolution.default(relu_65, arg251_1, arg252_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_65 = arg251_1 = arg252_1 = None
        add_290 = torch.ops.aten.add.Tensor(convolution_148, 3);  convolution_148 = None
        clamp_min_9 = torch.ops.aten.clamp_min.default(add_290, 0);  add_290 = None
        clamp_max_9 = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
        div_9 = torch.ops.aten.div.Tensor(clamp_max_9, 6);  clamp_max_9 = None
        mul_393 = torch.ops.aten.mul.Tensor(cat_50, div_9);  cat_50 = div_9 = None
        convolution_149 = torch.ops.aten.convolution.default(mul_393, arg253_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_393 = arg253_1 = None
        add_291 = torch.ops.aten.add.Tensor(arg255_1, 1e-05);  arg255_1 = None
        sqrt_128 = torch.ops.aten.sqrt.default(add_291);  add_291 = None
        reciprocal_128 = torch.ops.aten.reciprocal.default(sqrt_128);  sqrt_128 = None
        mul_394 = torch.ops.aten.mul.Tensor(reciprocal_128, 1);  reciprocal_128 = None
        unsqueeze_1024 = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_1025 = torch.ops.aten.unsqueeze.default(unsqueeze_1024, -1);  unsqueeze_1024 = None
        unsqueeze_1026 = torch.ops.aten.unsqueeze.default(mul_394, -1);  mul_394 = None
        unsqueeze_1027 = torch.ops.aten.unsqueeze.default(unsqueeze_1026, -1);  unsqueeze_1026 = None
        sub_128 = torch.ops.aten.sub.Tensor(convolution_149, unsqueeze_1025);  convolution_149 = unsqueeze_1025 = None
        mul_395 = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_1027);  sub_128 = unsqueeze_1027 = None
        unsqueeze_1028 = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_1029 = torch.ops.aten.unsqueeze.default(unsqueeze_1028, -1);  unsqueeze_1028 = None
        mul_396 = torch.ops.aten.mul.Tensor(mul_395, unsqueeze_1029);  mul_395 = unsqueeze_1029 = None
        unsqueeze_1030 = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_1031 = torch.ops.aten.unsqueeze.default(unsqueeze_1030, -1);  unsqueeze_1030 = None
        add_292 = torch.ops.aten.add.Tensor(mul_396, unsqueeze_1031);  mul_396 = unsqueeze_1031 = None
        convolution_150 = torch.ops.aten.convolution.default(add_292, arg258_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 56);  arg258_1 = None
        add_293 = torch.ops.aten.add.Tensor(arg260_1, 1e-05);  arg260_1 = None
        sqrt_129 = torch.ops.aten.sqrt.default(add_293);  add_293 = None
        reciprocal_129 = torch.ops.aten.reciprocal.default(sqrt_129);  sqrt_129 = None
        mul_397 = torch.ops.aten.mul.Tensor(reciprocal_129, 1);  reciprocal_129 = None
        unsqueeze_1032 = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_1033 = torch.ops.aten.unsqueeze.default(unsqueeze_1032, -1);  unsqueeze_1032 = None
        unsqueeze_1034 = torch.ops.aten.unsqueeze.default(mul_397, -1);  mul_397 = None
        unsqueeze_1035 = torch.ops.aten.unsqueeze.default(unsqueeze_1034, -1);  unsqueeze_1034 = None
        sub_129 = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_1033);  convolution_150 = unsqueeze_1033 = None
        mul_398 = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_1035);  sub_129 = unsqueeze_1035 = None
        unsqueeze_1036 = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_1037 = torch.ops.aten.unsqueeze.default(unsqueeze_1036, -1);  unsqueeze_1036 = None
        mul_399 = torch.ops.aten.mul.Tensor(mul_398, unsqueeze_1037);  mul_398 = unsqueeze_1037 = None
        unsqueeze_1038 = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_1039 = torch.ops.aten.unsqueeze.default(unsqueeze_1038, -1);  unsqueeze_1038 = None
        add_294 = torch.ops.aten.add.Tensor(mul_399, unsqueeze_1039);  mul_399 = unsqueeze_1039 = None
        cat_51 = torch.ops.aten.cat.default([add_292, add_294], 1);  add_292 = add_294 = None
        convolution_151 = torch.ops.aten.convolution.default(add_285, arg263_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  add_285 = arg263_1 = None
        add_295 = torch.ops.aten.add.Tensor(arg265_1, 1e-05);  arg265_1 = None
        sqrt_130 = torch.ops.aten.sqrt.default(add_295);  add_295 = None
        reciprocal_130 = torch.ops.aten.reciprocal.default(sqrt_130);  sqrt_130 = None
        mul_400 = torch.ops.aten.mul.Tensor(reciprocal_130, 1);  reciprocal_130 = None
        unsqueeze_1040 = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_1041 = torch.ops.aten.unsqueeze.default(unsqueeze_1040, -1);  unsqueeze_1040 = None
        unsqueeze_1042 = torch.ops.aten.unsqueeze.default(mul_400, -1);  mul_400 = None
        unsqueeze_1043 = torch.ops.aten.unsqueeze.default(unsqueeze_1042, -1);  unsqueeze_1042 = None
        sub_130 = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_1041);  convolution_151 = unsqueeze_1041 = None
        mul_401 = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_1043);  sub_130 = unsqueeze_1043 = None
        unsqueeze_1044 = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
        unsqueeze_1045 = torch.ops.aten.unsqueeze.default(unsqueeze_1044, -1);  unsqueeze_1044 = None
        mul_402 = torch.ops.aten.mul.Tensor(mul_401, unsqueeze_1045);  mul_401 = unsqueeze_1045 = None
        unsqueeze_1046 = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_1047 = torch.ops.aten.unsqueeze.default(unsqueeze_1046, -1);  unsqueeze_1046 = None
        add_296 = torch.ops.aten.add.Tensor(mul_402, unsqueeze_1047);  mul_402 = unsqueeze_1047 = None
        convolution_152 = torch.ops.aten.convolution.default(add_296, arg268_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_296 = arg268_1 = None
        add_297 = torch.ops.aten.add.Tensor(arg270_1, 1e-05);  arg270_1 = None
        sqrt_131 = torch.ops.aten.sqrt.default(add_297);  add_297 = None
        reciprocal_131 = torch.ops.aten.reciprocal.default(sqrt_131);  sqrt_131 = None
        mul_403 = torch.ops.aten.mul.Tensor(reciprocal_131, 1);  reciprocal_131 = None
        unsqueeze_1048 = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_1049 = torch.ops.aten.unsqueeze.default(unsqueeze_1048, -1);  unsqueeze_1048 = None
        unsqueeze_1050 = torch.ops.aten.unsqueeze.default(mul_403, -1);  mul_403 = None
        unsqueeze_1051 = torch.ops.aten.unsqueeze.default(unsqueeze_1050, -1);  unsqueeze_1050 = None
        sub_131 = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_1049);  convolution_152 = unsqueeze_1049 = None
        mul_404 = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_1051);  sub_131 = unsqueeze_1051 = None
        unsqueeze_1052 = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_1053 = torch.ops.aten.unsqueeze.default(unsqueeze_1052, -1);  unsqueeze_1052 = None
        mul_405 = torch.ops.aten.mul.Tensor(mul_404, unsqueeze_1053);  mul_404 = unsqueeze_1053 = None
        unsqueeze_1054 = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_1055 = torch.ops.aten.unsqueeze.default(unsqueeze_1054, -1);  unsqueeze_1054 = None
        add_298 = torch.ops.aten.add.Tensor(mul_405, unsqueeze_1055);  mul_405 = unsqueeze_1055 = None
        add_299 = torch.ops.aten.add.Tensor(cat_51, add_298);  cat_51 = add_298 = None
        convolution_153 = torch.ops.aten.convolution.default(add_299, arg273_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg273_1 = None
        add_300 = torch.ops.aten.add.Tensor(arg275_1, 1e-05);  arg275_1 = None
        sqrt_132 = torch.ops.aten.sqrt.default(add_300);  add_300 = None
        reciprocal_132 = torch.ops.aten.reciprocal.default(sqrt_132);  sqrt_132 = None
        mul_406 = torch.ops.aten.mul.Tensor(reciprocal_132, 1);  reciprocal_132 = None
        unsqueeze_1056 = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_1057 = torch.ops.aten.unsqueeze.default(unsqueeze_1056, -1);  unsqueeze_1056 = None
        unsqueeze_1058 = torch.ops.aten.unsqueeze.default(mul_406, -1);  mul_406 = None
        unsqueeze_1059 = torch.ops.aten.unsqueeze.default(unsqueeze_1058, -1);  unsqueeze_1058 = None
        sub_132 = torch.ops.aten.sub.Tensor(convolution_153, unsqueeze_1057);  convolution_153 = unsqueeze_1057 = None
        mul_407 = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_1059);  sub_132 = unsqueeze_1059 = None
        unsqueeze_1060 = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_1061 = torch.ops.aten.unsqueeze.default(unsqueeze_1060, -1);  unsqueeze_1060 = None
        mul_408 = torch.ops.aten.mul.Tensor(mul_407, unsqueeze_1061);  mul_407 = unsqueeze_1061 = None
        unsqueeze_1062 = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_1063 = torch.ops.aten.unsqueeze.default(unsqueeze_1062, -1);  unsqueeze_1062 = None
        add_301 = torch.ops.aten.add.Tensor(mul_408, unsqueeze_1063);  mul_408 = unsqueeze_1063 = None
        relu_66 = torch.ops.aten.relu.default(add_301);  add_301 = None
        convolution_154 = torch.ops.aten.convolution.default(relu_66, arg278_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 336);  arg278_1 = None
        add_302 = torch.ops.aten.add.Tensor(arg280_1, 1e-05);  arg280_1 = None
        sqrt_133 = torch.ops.aten.sqrt.default(add_302);  add_302 = None
        reciprocal_133 = torch.ops.aten.reciprocal.default(sqrt_133);  sqrt_133 = None
        mul_409 = torch.ops.aten.mul.Tensor(reciprocal_133, 1);  reciprocal_133 = None
        unsqueeze_1064 = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_1065 = torch.ops.aten.unsqueeze.default(unsqueeze_1064, -1);  unsqueeze_1064 = None
        unsqueeze_1066 = torch.ops.aten.unsqueeze.default(mul_409, -1);  mul_409 = None
        unsqueeze_1067 = torch.ops.aten.unsqueeze.default(unsqueeze_1066, -1);  unsqueeze_1066 = None
        sub_133 = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_1065);  convolution_154 = unsqueeze_1065 = None
        mul_410 = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_1067);  sub_133 = unsqueeze_1067 = None
        unsqueeze_1068 = torch.ops.aten.unsqueeze.default(arg281_1, -1);  arg281_1 = None
        unsqueeze_1069 = torch.ops.aten.unsqueeze.default(unsqueeze_1068, -1);  unsqueeze_1068 = None
        mul_411 = torch.ops.aten.mul.Tensor(mul_410, unsqueeze_1069);  mul_410 = unsqueeze_1069 = None
        unsqueeze_1070 = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_1071 = torch.ops.aten.unsqueeze.default(unsqueeze_1070, -1);  unsqueeze_1070 = None
        add_303 = torch.ops.aten.add.Tensor(mul_411, unsqueeze_1071);  mul_411 = unsqueeze_1071 = None
        relu_67 = torch.ops.aten.relu.default(add_303);  add_303 = None
        cat_52 = torch.ops.aten.cat.default([relu_66, relu_67], 1);  relu_66 = relu_67 = None
        mean_11 = torch.ops.aten.mean.dim(cat_52, [2, 3], True)
        convolution_155 = torch.ops.aten.convolution.default(mean_11, arg283_1, arg284_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_11 = arg283_1 = arg284_1 = None
        relu_68 = torch.ops.aten.relu.default(convolution_155);  convolution_155 = None
        convolution_156 = torch.ops.aten.convolution.default(relu_68, arg285_1, arg286_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_68 = arg285_1 = arg286_1 = None
        add_304 = torch.ops.aten.add.Tensor(convolution_156, 3);  convolution_156 = None
        clamp_min_10 = torch.ops.aten.clamp_min.default(add_304, 0);  add_304 = None
        clamp_max_10 = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
        div_10 = torch.ops.aten.div.Tensor(clamp_max_10, 6);  clamp_max_10 = None
        mul_412 = torch.ops.aten.mul.Tensor(cat_52, div_10);  cat_52 = div_10 = None
        convolution_157 = torch.ops.aten.convolution.default(mul_412, arg287_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_412 = arg287_1 = None
        add_305 = torch.ops.aten.add.Tensor(arg289_1, 1e-05);  arg289_1 = None
        sqrt_134 = torch.ops.aten.sqrt.default(add_305);  add_305 = None
        reciprocal_134 = torch.ops.aten.reciprocal.default(sqrt_134);  sqrt_134 = None
        mul_413 = torch.ops.aten.mul.Tensor(reciprocal_134, 1);  reciprocal_134 = None
        unsqueeze_1072 = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_1073 = torch.ops.aten.unsqueeze.default(unsqueeze_1072, -1);  unsqueeze_1072 = None
        unsqueeze_1074 = torch.ops.aten.unsqueeze.default(mul_413, -1);  mul_413 = None
        unsqueeze_1075 = torch.ops.aten.unsqueeze.default(unsqueeze_1074, -1);  unsqueeze_1074 = None
        sub_134 = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_1073);  convolution_157 = unsqueeze_1073 = None
        mul_414 = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_1075);  sub_134 = unsqueeze_1075 = None
        unsqueeze_1076 = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_1077 = torch.ops.aten.unsqueeze.default(unsqueeze_1076, -1);  unsqueeze_1076 = None
        mul_415 = torch.ops.aten.mul.Tensor(mul_414, unsqueeze_1077);  mul_414 = unsqueeze_1077 = None
        unsqueeze_1078 = torch.ops.aten.unsqueeze.default(arg291_1, -1);  arg291_1 = None
        unsqueeze_1079 = torch.ops.aten.unsqueeze.default(unsqueeze_1078, -1);  unsqueeze_1078 = None
        add_306 = torch.ops.aten.add.Tensor(mul_415, unsqueeze_1079);  mul_415 = unsqueeze_1079 = None
        convolution_158 = torch.ops.aten.convolution.default(add_306, arg292_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 56);  arg292_1 = None
        add_307 = torch.ops.aten.add.Tensor(arg294_1, 1e-05);  arg294_1 = None
        sqrt_135 = torch.ops.aten.sqrt.default(add_307);  add_307 = None
        reciprocal_135 = torch.ops.aten.reciprocal.default(sqrt_135);  sqrt_135 = None
        mul_416 = torch.ops.aten.mul.Tensor(reciprocal_135, 1);  reciprocal_135 = None
        unsqueeze_1080 = torch.ops.aten.unsqueeze.default(arg293_1, -1);  arg293_1 = None
        unsqueeze_1081 = torch.ops.aten.unsqueeze.default(unsqueeze_1080, -1);  unsqueeze_1080 = None
        unsqueeze_1082 = torch.ops.aten.unsqueeze.default(mul_416, -1);  mul_416 = None
        unsqueeze_1083 = torch.ops.aten.unsqueeze.default(unsqueeze_1082, -1);  unsqueeze_1082 = None
        sub_135 = torch.ops.aten.sub.Tensor(convolution_158, unsqueeze_1081);  convolution_158 = unsqueeze_1081 = None
        mul_417 = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_1083);  sub_135 = unsqueeze_1083 = None
        unsqueeze_1084 = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_1085 = torch.ops.aten.unsqueeze.default(unsqueeze_1084, -1);  unsqueeze_1084 = None
        mul_418 = torch.ops.aten.mul.Tensor(mul_417, unsqueeze_1085);  mul_417 = unsqueeze_1085 = None
        unsqueeze_1086 = torch.ops.aten.unsqueeze.default(arg296_1, -1);  arg296_1 = None
        unsqueeze_1087 = torch.ops.aten.unsqueeze.default(unsqueeze_1086, -1);  unsqueeze_1086 = None
        add_308 = torch.ops.aten.add.Tensor(mul_418, unsqueeze_1087);  mul_418 = unsqueeze_1087 = None
        cat_53 = torch.ops.aten.cat.default([add_306, add_308], 1);  add_306 = add_308 = None
        add_309 = torch.ops.aten.add.Tensor(cat_53, add_299);  cat_53 = add_299 = None
        convolution_159 = torch.ops.aten.convolution.default(add_309, arg297_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg297_1 = None
        add_310 = torch.ops.aten.add.Tensor(arg299_1, 1e-05);  arg299_1 = None
        sqrt_136 = torch.ops.aten.sqrt.default(add_310);  add_310 = None
        reciprocal_136 = torch.ops.aten.reciprocal.default(sqrt_136);  sqrt_136 = None
        mul_419 = torch.ops.aten.mul.Tensor(reciprocal_136, 1);  reciprocal_136 = None
        unsqueeze_1088 = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
        unsqueeze_1089 = torch.ops.aten.unsqueeze.default(unsqueeze_1088, -1);  unsqueeze_1088 = None
        unsqueeze_1090 = torch.ops.aten.unsqueeze.default(mul_419, -1);  mul_419 = None
        unsqueeze_1091 = torch.ops.aten.unsqueeze.default(unsqueeze_1090, -1);  unsqueeze_1090 = None
        sub_136 = torch.ops.aten.sub.Tensor(convolution_159, unsqueeze_1089);  convolution_159 = unsqueeze_1089 = None
        mul_420 = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_1091);  sub_136 = unsqueeze_1091 = None
        unsqueeze_1092 = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_1093 = torch.ops.aten.unsqueeze.default(unsqueeze_1092, -1);  unsqueeze_1092 = None
        mul_421 = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_1093);  mul_420 = unsqueeze_1093 = None
        unsqueeze_1094 = torch.ops.aten.unsqueeze.default(arg301_1, -1);  arg301_1 = None
        unsqueeze_1095 = torch.ops.aten.unsqueeze.default(unsqueeze_1094, -1);  unsqueeze_1094 = None
        add_311 = torch.ops.aten.add.Tensor(mul_421, unsqueeze_1095);  mul_421 = unsqueeze_1095 = None
        relu_69 = torch.ops.aten.relu.default(add_311);  add_311 = None
        convolution_160 = torch.ops.aten.convolution.default(relu_69, arg302_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 336);  arg302_1 = None
        add_312 = torch.ops.aten.add.Tensor(arg304_1, 1e-05);  arg304_1 = None
        sqrt_137 = torch.ops.aten.sqrt.default(add_312);  add_312 = None
        reciprocal_137 = torch.ops.aten.reciprocal.default(sqrt_137);  sqrt_137 = None
        mul_422 = torch.ops.aten.mul.Tensor(reciprocal_137, 1);  reciprocal_137 = None
        unsqueeze_1096 = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_1097 = torch.ops.aten.unsqueeze.default(unsqueeze_1096, -1);  unsqueeze_1096 = None
        unsqueeze_1098 = torch.ops.aten.unsqueeze.default(mul_422, -1);  mul_422 = None
        unsqueeze_1099 = torch.ops.aten.unsqueeze.default(unsqueeze_1098, -1);  unsqueeze_1098 = None
        sub_137 = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_1097);  convolution_160 = unsqueeze_1097 = None
        mul_423 = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_1099);  sub_137 = unsqueeze_1099 = None
        unsqueeze_1100 = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_1101 = torch.ops.aten.unsqueeze.default(unsqueeze_1100, -1);  unsqueeze_1100 = None
        mul_424 = torch.ops.aten.mul.Tensor(mul_423, unsqueeze_1101);  mul_423 = unsqueeze_1101 = None
        unsqueeze_1102 = torch.ops.aten.unsqueeze.default(arg306_1, -1);  arg306_1 = None
        unsqueeze_1103 = torch.ops.aten.unsqueeze.default(unsqueeze_1102, -1);  unsqueeze_1102 = None
        add_313 = torch.ops.aten.add.Tensor(mul_424, unsqueeze_1103);  mul_424 = unsqueeze_1103 = None
        relu_70 = torch.ops.aten.relu.default(add_313);  add_313 = None
        cat_54 = torch.ops.aten.cat.default([relu_69, relu_70], 1);  relu_69 = relu_70 = None
        convolution_161 = torch.ops.aten.convolution.default(cat_54, arg307_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672);  cat_54 = arg307_1 = None
        add_314 = torch.ops.aten.add.Tensor(arg309_1, 1e-05);  arg309_1 = None
        sqrt_138 = torch.ops.aten.sqrt.default(add_314);  add_314 = None
        reciprocal_138 = torch.ops.aten.reciprocal.default(sqrt_138);  sqrt_138 = None
        mul_425 = torch.ops.aten.mul.Tensor(reciprocal_138, 1);  reciprocal_138 = None
        unsqueeze_1104 = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_1105 = torch.ops.aten.unsqueeze.default(unsqueeze_1104, -1);  unsqueeze_1104 = None
        unsqueeze_1106 = torch.ops.aten.unsqueeze.default(mul_425, -1);  mul_425 = None
        unsqueeze_1107 = torch.ops.aten.unsqueeze.default(unsqueeze_1106, -1);  unsqueeze_1106 = None
        sub_138 = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_1105);  convolution_161 = unsqueeze_1105 = None
        mul_426 = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_1107);  sub_138 = unsqueeze_1107 = None
        unsqueeze_1108 = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
        unsqueeze_1109 = torch.ops.aten.unsqueeze.default(unsqueeze_1108, -1);  unsqueeze_1108 = None
        mul_427 = torch.ops.aten.mul.Tensor(mul_426, unsqueeze_1109);  mul_426 = unsqueeze_1109 = None
        unsqueeze_1110 = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_1111 = torch.ops.aten.unsqueeze.default(unsqueeze_1110, -1);  unsqueeze_1110 = None
        add_315 = torch.ops.aten.add.Tensor(mul_427, unsqueeze_1111);  mul_427 = unsqueeze_1111 = None
        mean_12 = torch.ops.aten.mean.dim(add_315, [2, 3], True)
        convolution_162 = torch.ops.aten.convolution.default(mean_12, arg312_1, arg313_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_12 = arg312_1 = arg313_1 = None
        relu_71 = torch.ops.aten.relu.default(convolution_162);  convolution_162 = None
        convolution_163 = torch.ops.aten.convolution.default(relu_71, arg314_1, arg315_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_71 = arg314_1 = arg315_1 = None
        add_316 = torch.ops.aten.add.Tensor(convolution_163, 3);  convolution_163 = None
        clamp_min_11 = torch.ops.aten.clamp_min.default(add_316, 0);  add_316 = None
        clamp_max_11 = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
        div_11 = torch.ops.aten.div.Tensor(clamp_max_11, 6);  clamp_max_11 = None
        mul_428 = torch.ops.aten.mul.Tensor(add_315, div_11);  add_315 = div_11 = None
        convolution_164 = torch.ops.aten.convolution.default(mul_428, arg316_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_428 = arg316_1 = None
        add_317 = torch.ops.aten.add.Tensor(arg318_1, 1e-05);  arg318_1 = None
        sqrt_139 = torch.ops.aten.sqrt.default(add_317);  add_317 = None
        reciprocal_139 = torch.ops.aten.reciprocal.default(sqrt_139);  sqrt_139 = None
        mul_429 = torch.ops.aten.mul.Tensor(reciprocal_139, 1);  reciprocal_139 = None
        unsqueeze_1112 = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
        unsqueeze_1113 = torch.ops.aten.unsqueeze.default(unsqueeze_1112, -1);  unsqueeze_1112 = None
        unsqueeze_1114 = torch.ops.aten.unsqueeze.default(mul_429, -1);  mul_429 = None
        unsqueeze_1115 = torch.ops.aten.unsqueeze.default(unsqueeze_1114, -1);  unsqueeze_1114 = None
        sub_139 = torch.ops.aten.sub.Tensor(convolution_164, unsqueeze_1113);  convolution_164 = unsqueeze_1113 = None
        mul_430 = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_1115);  sub_139 = unsqueeze_1115 = None
        unsqueeze_1116 = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_1117 = torch.ops.aten.unsqueeze.default(unsqueeze_1116, -1);  unsqueeze_1116 = None
        mul_431 = torch.ops.aten.mul.Tensor(mul_430, unsqueeze_1117);  mul_430 = unsqueeze_1117 = None
        unsqueeze_1118 = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_1119 = torch.ops.aten.unsqueeze.default(unsqueeze_1118, -1);  unsqueeze_1118 = None
        add_318 = torch.ops.aten.add.Tensor(mul_431, unsqueeze_1119);  mul_431 = unsqueeze_1119 = None
        convolution_165 = torch.ops.aten.convolution.default(add_318, arg321_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg321_1 = None
        add_319 = torch.ops.aten.add.Tensor(arg323_1, 1e-05);  arg323_1 = None
        sqrt_140 = torch.ops.aten.sqrt.default(add_319);  add_319 = None
        reciprocal_140 = torch.ops.aten.reciprocal.default(sqrt_140);  sqrt_140 = None
        mul_432 = torch.ops.aten.mul.Tensor(reciprocal_140, 1);  reciprocal_140 = None
        unsqueeze_1120 = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_1121 = torch.ops.aten.unsqueeze.default(unsqueeze_1120, -1);  unsqueeze_1120 = None
        unsqueeze_1122 = torch.ops.aten.unsqueeze.default(mul_432, -1);  mul_432 = None
        unsqueeze_1123 = torch.ops.aten.unsqueeze.default(unsqueeze_1122, -1);  unsqueeze_1122 = None
        sub_140 = torch.ops.aten.sub.Tensor(convolution_165, unsqueeze_1121);  convolution_165 = unsqueeze_1121 = None
        mul_433 = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_1123);  sub_140 = unsqueeze_1123 = None
        unsqueeze_1124 = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_1125 = torch.ops.aten.unsqueeze.default(unsqueeze_1124, -1);  unsqueeze_1124 = None
        mul_434 = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_1125);  mul_433 = unsqueeze_1125 = None
        unsqueeze_1126 = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_1127 = torch.ops.aten.unsqueeze.default(unsqueeze_1126, -1);  unsqueeze_1126 = None
        add_320 = torch.ops.aten.add.Tensor(mul_434, unsqueeze_1127);  mul_434 = unsqueeze_1127 = None
        cat_55 = torch.ops.aten.cat.default([add_318, add_320], 1);  add_318 = add_320 = None
        convolution_166 = torch.ops.aten.convolution.default(add_309, arg326_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 112);  add_309 = arg326_1 = None
        add_321 = torch.ops.aten.add.Tensor(arg328_1, 1e-05);  arg328_1 = None
        sqrt_141 = torch.ops.aten.sqrt.default(add_321);  add_321 = None
        reciprocal_141 = torch.ops.aten.reciprocal.default(sqrt_141);  sqrt_141 = None
        mul_435 = torch.ops.aten.mul.Tensor(reciprocal_141, 1);  reciprocal_141 = None
        unsqueeze_1128 = torch.ops.aten.unsqueeze.default(arg327_1, -1);  arg327_1 = None
        unsqueeze_1129 = torch.ops.aten.unsqueeze.default(unsqueeze_1128, -1);  unsqueeze_1128 = None
        unsqueeze_1130 = torch.ops.aten.unsqueeze.default(mul_435, -1);  mul_435 = None
        unsqueeze_1131 = torch.ops.aten.unsqueeze.default(unsqueeze_1130, -1);  unsqueeze_1130 = None
        sub_141 = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_1129);  convolution_166 = unsqueeze_1129 = None
        mul_436 = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_1131);  sub_141 = unsqueeze_1131 = None
        unsqueeze_1132 = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_1133 = torch.ops.aten.unsqueeze.default(unsqueeze_1132, -1);  unsqueeze_1132 = None
        mul_437 = torch.ops.aten.mul.Tensor(mul_436, unsqueeze_1133);  mul_436 = unsqueeze_1133 = None
        unsqueeze_1134 = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_1135 = torch.ops.aten.unsqueeze.default(unsqueeze_1134, -1);  unsqueeze_1134 = None
        add_322 = torch.ops.aten.add.Tensor(mul_437, unsqueeze_1135);  mul_437 = unsqueeze_1135 = None
        convolution_167 = torch.ops.aten.convolution.default(add_322, arg331_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_322 = arg331_1 = None
        add_323 = torch.ops.aten.add.Tensor(arg333_1, 1e-05);  arg333_1 = None
        sqrt_142 = torch.ops.aten.sqrt.default(add_323);  add_323 = None
        reciprocal_142 = torch.ops.aten.reciprocal.default(sqrt_142);  sqrt_142 = None
        mul_438 = torch.ops.aten.mul.Tensor(reciprocal_142, 1);  reciprocal_142 = None
        unsqueeze_1136 = torch.ops.aten.unsqueeze.default(arg332_1, -1);  arg332_1 = None
        unsqueeze_1137 = torch.ops.aten.unsqueeze.default(unsqueeze_1136, -1);  unsqueeze_1136 = None
        unsqueeze_1138 = torch.ops.aten.unsqueeze.default(mul_438, -1);  mul_438 = None
        unsqueeze_1139 = torch.ops.aten.unsqueeze.default(unsqueeze_1138, -1);  unsqueeze_1138 = None
        sub_142 = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_1137);  convolution_167 = unsqueeze_1137 = None
        mul_439 = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_1139);  sub_142 = unsqueeze_1139 = None
        unsqueeze_1140 = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_1141 = torch.ops.aten.unsqueeze.default(unsqueeze_1140, -1);  unsqueeze_1140 = None
        mul_440 = torch.ops.aten.mul.Tensor(mul_439, unsqueeze_1141);  mul_439 = unsqueeze_1141 = None
        unsqueeze_1142 = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
        unsqueeze_1143 = torch.ops.aten.unsqueeze.default(unsqueeze_1142, -1);  unsqueeze_1142 = None
        add_324 = torch.ops.aten.add.Tensor(mul_440, unsqueeze_1143);  mul_440 = unsqueeze_1143 = None
        add_325 = torch.ops.aten.add.Tensor(cat_55, add_324);  cat_55 = add_324 = None
        convolution_168 = torch.ops.aten.convolution.default(add_325, arg336_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg336_1 = None
        add_326 = torch.ops.aten.add.Tensor(arg338_1, 1e-05);  arg338_1 = None
        sqrt_143 = torch.ops.aten.sqrt.default(add_326);  add_326 = None
        reciprocal_143 = torch.ops.aten.reciprocal.default(sqrt_143);  sqrt_143 = None
        mul_441 = torch.ops.aten.mul.Tensor(reciprocal_143, 1);  reciprocal_143 = None
        unsqueeze_1144 = torch.ops.aten.unsqueeze.default(arg337_1, -1);  arg337_1 = None
        unsqueeze_1145 = torch.ops.aten.unsqueeze.default(unsqueeze_1144, -1);  unsqueeze_1144 = None
        unsqueeze_1146 = torch.ops.aten.unsqueeze.default(mul_441, -1);  mul_441 = None
        unsqueeze_1147 = torch.ops.aten.unsqueeze.default(unsqueeze_1146, -1);  unsqueeze_1146 = None
        sub_143 = torch.ops.aten.sub.Tensor(convolution_168, unsqueeze_1145);  convolution_168 = unsqueeze_1145 = None
        mul_442 = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_1147);  sub_143 = unsqueeze_1147 = None
        unsqueeze_1148 = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_1149 = torch.ops.aten.unsqueeze.default(unsqueeze_1148, -1);  unsqueeze_1148 = None
        mul_443 = torch.ops.aten.mul.Tensor(mul_442, unsqueeze_1149);  mul_442 = unsqueeze_1149 = None
        unsqueeze_1150 = torch.ops.aten.unsqueeze.default(arg340_1, -1);  arg340_1 = None
        unsqueeze_1151 = torch.ops.aten.unsqueeze.default(unsqueeze_1150, -1);  unsqueeze_1150 = None
        add_327 = torch.ops.aten.add.Tensor(mul_443, unsqueeze_1151);  mul_443 = unsqueeze_1151 = None
        relu_72 = torch.ops.aten.relu.default(add_327);  add_327 = None
        convolution_169 = torch.ops.aten.convolution.default(relu_72, arg341_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  arg341_1 = None
        add_328 = torch.ops.aten.add.Tensor(arg343_1, 1e-05);  arg343_1 = None
        sqrt_144 = torch.ops.aten.sqrt.default(add_328);  add_328 = None
        reciprocal_144 = torch.ops.aten.reciprocal.default(sqrt_144);  sqrt_144 = None
        mul_444 = torch.ops.aten.mul.Tensor(reciprocal_144, 1);  reciprocal_144 = None
        unsqueeze_1152 = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_1153 = torch.ops.aten.unsqueeze.default(unsqueeze_1152, -1);  unsqueeze_1152 = None
        unsqueeze_1154 = torch.ops.aten.unsqueeze.default(mul_444, -1);  mul_444 = None
        unsqueeze_1155 = torch.ops.aten.unsqueeze.default(unsqueeze_1154, -1);  unsqueeze_1154 = None
        sub_144 = torch.ops.aten.sub.Tensor(convolution_169, unsqueeze_1153);  convolution_169 = unsqueeze_1153 = None
        mul_445 = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_1155);  sub_144 = unsqueeze_1155 = None
        unsqueeze_1156 = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_1157 = torch.ops.aten.unsqueeze.default(unsqueeze_1156, -1);  unsqueeze_1156 = None
        mul_446 = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_1157);  mul_445 = unsqueeze_1157 = None
        unsqueeze_1158 = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_1159 = torch.ops.aten.unsqueeze.default(unsqueeze_1158, -1);  unsqueeze_1158 = None
        add_329 = torch.ops.aten.add.Tensor(mul_446, unsqueeze_1159);  mul_446 = unsqueeze_1159 = None
        relu_73 = torch.ops.aten.relu.default(add_329);  add_329 = None
        cat_56 = torch.ops.aten.cat.default([relu_72, relu_73], 1);  relu_72 = relu_73 = None
        convolution_170 = torch.ops.aten.convolution.default(cat_56, arg346_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_56 = arg346_1 = None
        add_330 = torch.ops.aten.add.Tensor(arg348_1, 1e-05);  arg348_1 = None
        sqrt_145 = torch.ops.aten.sqrt.default(add_330);  add_330 = None
        reciprocal_145 = torch.ops.aten.reciprocal.default(sqrt_145);  sqrt_145 = None
        mul_447 = torch.ops.aten.mul.Tensor(reciprocal_145, 1);  reciprocal_145 = None
        unsqueeze_1160 = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
        unsqueeze_1161 = torch.ops.aten.unsqueeze.default(unsqueeze_1160, -1);  unsqueeze_1160 = None
        unsqueeze_1162 = torch.ops.aten.unsqueeze.default(mul_447, -1);  mul_447 = None
        unsqueeze_1163 = torch.ops.aten.unsqueeze.default(unsqueeze_1162, -1);  unsqueeze_1162 = None
        sub_145 = torch.ops.aten.sub.Tensor(convolution_170, unsqueeze_1161);  convolution_170 = unsqueeze_1161 = None
        mul_448 = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_1163);  sub_145 = unsqueeze_1163 = None
        unsqueeze_1164 = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_1165 = torch.ops.aten.unsqueeze.default(unsqueeze_1164, -1);  unsqueeze_1164 = None
        mul_449 = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_1165);  mul_448 = unsqueeze_1165 = None
        unsqueeze_1166 = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
        unsqueeze_1167 = torch.ops.aten.unsqueeze.default(unsqueeze_1166, -1);  unsqueeze_1166 = None
        add_331 = torch.ops.aten.add.Tensor(mul_449, unsqueeze_1167);  mul_449 = unsqueeze_1167 = None
        convolution_171 = torch.ops.aten.convolution.default(add_331, arg351_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg351_1 = None
        add_332 = torch.ops.aten.add.Tensor(arg353_1, 1e-05);  arg353_1 = None
        sqrt_146 = torch.ops.aten.sqrt.default(add_332);  add_332 = None
        reciprocal_146 = torch.ops.aten.reciprocal.default(sqrt_146);  sqrt_146 = None
        mul_450 = torch.ops.aten.mul.Tensor(reciprocal_146, 1);  reciprocal_146 = None
        unsqueeze_1168 = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
        unsqueeze_1169 = torch.ops.aten.unsqueeze.default(unsqueeze_1168, -1);  unsqueeze_1168 = None
        unsqueeze_1170 = torch.ops.aten.unsqueeze.default(mul_450, -1);  mul_450 = None
        unsqueeze_1171 = torch.ops.aten.unsqueeze.default(unsqueeze_1170, -1);  unsqueeze_1170 = None
        sub_146 = torch.ops.aten.sub.Tensor(convolution_171, unsqueeze_1169);  convolution_171 = unsqueeze_1169 = None
        mul_451 = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_1171);  sub_146 = unsqueeze_1171 = None
        unsqueeze_1172 = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_1173 = torch.ops.aten.unsqueeze.default(unsqueeze_1172, -1);  unsqueeze_1172 = None
        mul_452 = torch.ops.aten.mul.Tensor(mul_451, unsqueeze_1173);  mul_451 = unsqueeze_1173 = None
        unsqueeze_1174 = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
        unsqueeze_1175 = torch.ops.aten.unsqueeze.default(unsqueeze_1174, -1);  unsqueeze_1174 = None
        add_333 = torch.ops.aten.add.Tensor(mul_452, unsqueeze_1175);  mul_452 = unsqueeze_1175 = None
        cat_57 = torch.ops.aten.cat.default([add_331, add_333], 1);  add_331 = add_333 = None
        add_334 = torch.ops.aten.add.Tensor(cat_57, add_325);  cat_57 = add_325 = None
        convolution_172 = torch.ops.aten.convolution.default(add_334, arg356_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg356_1 = None
        add_335 = torch.ops.aten.add.Tensor(arg358_1, 1e-05);  arg358_1 = None
        sqrt_147 = torch.ops.aten.sqrt.default(add_335);  add_335 = None
        reciprocal_147 = torch.ops.aten.reciprocal.default(sqrt_147);  sqrt_147 = None
        mul_453 = torch.ops.aten.mul.Tensor(reciprocal_147, 1);  reciprocal_147 = None
        unsqueeze_1176 = torch.ops.aten.unsqueeze.default(arg357_1, -1);  arg357_1 = None
        unsqueeze_1177 = torch.ops.aten.unsqueeze.default(unsqueeze_1176, -1);  unsqueeze_1176 = None
        unsqueeze_1178 = torch.ops.aten.unsqueeze.default(mul_453, -1);  mul_453 = None
        unsqueeze_1179 = torch.ops.aten.unsqueeze.default(unsqueeze_1178, -1);  unsqueeze_1178 = None
        sub_147 = torch.ops.aten.sub.Tensor(convolution_172, unsqueeze_1177);  convolution_172 = unsqueeze_1177 = None
        mul_454 = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_1179);  sub_147 = unsqueeze_1179 = None
        unsqueeze_1180 = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_1181 = torch.ops.aten.unsqueeze.default(unsqueeze_1180, -1);  unsqueeze_1180 = None
        mul_455 = torch.ops.aten.mul.Tensor(mul_454, unsqueeze_1181);  mul_454 = unsqueeze_1181 = None
        unsqueeze_1182 = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_1183 = torch.ops.aten.unsqueeze.default(unsqueeze_1182, -1);  unsqueeze_1182 = None
        add_336 = torch.ops.aten.add.Tensor(mul_455, unsqueeze_1183);  mul_455 = unsqueeze_1183 = None
        relu_74 = torch.ops.aten.relu.default(add_336);  add_336 = None
        convolution_173 = torch.ops.aten.convolution.default(relu_74, arg361_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  arg361_1 = None
        add_337 = torch.ops.aten.add.Tensor(arg363_1, 1e-05);  arg363_1 = None
        sqrt_148 = torch.ops.aten.sqrt.default(add_337);  add_337 = None
        reciprocal_148 = torch.ops.aten.reciprocal.default(sqrt_148);  sqrt_148 = None
        mul_456 = torch.ops.aten.mul.Tensor(reciprocal_148, 1);  reciprocal_148 = None
        unsqueeze_1184 = torch.ops.aten.unsqueeze.default(arg362_1, -1);  arg362_1 = None
        unsqueeze_1185 = torch.ops.aten.unsqueeze.default(unsqueeze_1184, -1);  unsqueeze_1184 = None
        unsqueeze_1186 = torch.ops.aten.unsqueeze.default(mul_456, -1);  mul_456 = None
        unsqueeze_1187 = torch.ops.aten.unsqueeze.default(unsqueeze_1186, -1);  unsqueeze_1186 = None
        sub_148 = torch.ops.aten.sub.Tensor(convolution_173, unsqueeze_1185);  convolution_173 = unsqueeze_1185 = None
        mul_457 = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_1187);  sub_148 = unsqueeze_1187 = None
        unsqueeze_1188 = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
        unsqueeze_1189 = torch.ops.aten.unsqueeze.default(unsqueeze_1188, -1);  unsqueeze_1188 = None
        mul_458 = torch.ops.aten.mul.Tensor(mul_457, unsqueeze_1189);  mul_457 = unsqueeze_1189 = None
        unsqueeze_1190 = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
        unsqueeze_1191 = torch.ops.aten.unsqueeze.default(unsqueeze_1190, -1);  unsqueeze_1190 = None
        add_338 = torch.ops.aten.add.Tensor(mul_458, unsqueeze_1191);  mul_458 = unsqueeze_1191 = None
        relu_75 = torch.ops.aten.relu.default(add_338);  add_338 = None
        cat_58 = torch.ops.aten.cat.default([relu_74, relu_75], 1);  relu_74 = relu_75 = None
        mean_13 = torch.ops.aten.mean.dim(cat_58, [2, 3], True)
        convolution_174 = torch.ops.aten.convolution.default(mean_13, arg366_1, arg367_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_13 = arg366_1 = arg367_1 = None
        relu_76 = torch.ops.aten.relu.default(convolution_174);  convolution_174 = None
        convolution_175 = torch.ops.aten.convolution.default(relu_76, arg368_1, arg369_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_76 = arg368_1 = arg369_1 = None
        add_339 = torch.ops.aten.add.Tensor(convolution_175, 3);  convolution_175 = None
        clamp_min_12 = torch.ops.aten.clamp_min.default(add_339, 0);  add_339 = None
        clamp_max_12 = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
        div_12 = torch.ops.aten.div.Tensor(clamp_max_12, 6);  clamp_max_12 = None
        mul_459 = torch.ops.aten.mul.Tensor(cat_58, div_12);  cat_58 = div_12 = None
        convolution_176 = torch.ops.aten.convolution.default(mul_459, arg370_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_459 = arg370_1 = None
        add_340 = torch.ops.aten.add.Tensor(arg372_1, 1e-05);  arg372_1 = None
        sqrt_149 = torch.ops.aten.sqrt.default(add_340);  add_340 = None
        reciprocal_149 = torch.ops.aten.reciprocal.default(sqrt_149);  sqrt_149 = None
        mul_460 = torch.ops.aten.mul.Tensor(reciprocal_149, 1);  reciprocal_149 = None
        unsqueeze_1192 = torch.ops.aten.unsqueeze.default(arg371_1, -1);  arg371_1 = None
        unsqueeze_1193 = torch.ops.aten.unsqueeze.default(unsqueeze_1192, -1);  unsqueeze_1192 = None
        unsqueeze_1194 = torch.ops.aten.unsqueeze.default(mul_460, -1);  mul_460 = None
        unsqueeze_1195 = torch.ops.aten.unsqueeze.default(unsqueeze_1194, -1);  unsqueeze_1194 = None
        sub_149 = torch.ops.aten.sub.Tensor(convolution_176, unsqueeze_1193);  convolution_176 = unsqueeze_1193 = None
        mul_461 = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_1195);  sub_149 = unsqueeze_1195 = None
        unsqueeze_1196 = torch.ops.aten.unsqueeze.default(arg373_1, -1);  arg373_1 = None
        unsqueeze_1197 = torch.ops.aten.unsqueeze.default(unsqueeze_1196, -1);  unsqueeze_1196 = None
        mul_462 = torch.ops.aten.mul.Tensor(mul_461, unsqueeze_1197);  mul_461 = unsqueeze_1197 = None
        unsqueeze_1198 = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
        unsqueeze_1199 = torch.ops.aten.unsqueeze.default(unsqueeze_1198, -1);  unsqueeze_1198 = None
        add_341 = torch.ops.aten.add.Tensor(mul_462, unsqueeze_1199);  mul_462 = unsqueeze_1199 = None
        convolution_177 = torch.ops.aten.convolution.default(add_341, arg375_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg375_1 = None
        add_342 = torch.ops.aten.add.Tensor(arg377_1, 1e-05);  arg377_1 = None
        sqrt_150 = torch.ops.aten.sqrt.default(add_342);  add_342 = None
        reciprocal_150 = torch.ops.aten.reciprocal.default(sqrt_150);  sqrt_150 = None
        mul_463 = torch.ops.aten.mul.Tensor(reciprocal_150, 1);  reciprocal_150 = None
        unsqueeze_1200 = torch.ops.aten.unsqueeze.default(arg376_1, -1);  arg376_1 = None
        unsqueeze_1201 = torch.ops.aten.unsqueeze.default(unsqueeze_1200, -1);  unsqueeze_1200 = None
        unsqueeze_1202 = torch.ops.aten.unsqueeze.default(mul_463, -1);  mul_463 = None
        unsqueeze_1203 = torch.ops.aten.unsqueeze.default(unsqueeze_1202, -1);  unsqueeze_1202 = None
        sub_150 = torch.ops.aten.sub.Tensor(convolution_177, unsqueeze_1201);  convolution_177 = unsqueeze_1201 = None
        mul_464 = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_1203);  sub_150 = unsqueeze_1203 = None
        unsqueeze_1204 = torch.ops.aten.unsqueeze.default(arg378_1, -1);  arg378_1 = None
        unsqueeze_1205 = torch.ops.aten.unsqueeze.default(unsqueeze_1204, -1);  unsqueeze_1204 = None
        mul_465 = torch.ops.aten.mul.Tensor(mul_464, unsqueeze_1205);  mul_464 = unsqueeze_1205 = None
        unsqueeze_1206 = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_1207 = torch.ops.aten.unsqueeze.default(unsqueeze_1206, -1);  unsqueeze_1206 = None
        add_343 = torch.ops.aten.add.Tensor(mul_465, unsqueeze_1207);  mul_465 = unsqueeze_1207 = None
        cat_59 = torch.ops.aten.cat.default([add_341, add_343], 1);  add_341 = add_343 = None
        add_344 = torch.ops.aten.add.Tensor(cat_59, add_334);  cat_59 = add_334 = None
        convolution_178 = torch.ops.aten.convolution.default(add_344, arg380_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg380_1 = None
        add_345 = torch.ops.aten.add.Tensor(arg382_1, 1e-05);  arg382_1 = None
        sqrt_151 = torch.ops.aten.sqrt.default(add_345);  add_345 = None
        reciprocal_151 = torch.ops.aten.reciprocal.default(sqrt_151);  sqrt_151 = None
        mul_466 = torch.ops.aten.mul.Tensor(reciprocal_151, 1);  reciprocal_151 = None
        unsqueeze_1208 = torch.ops.aten.unsqueeze.default(arg381_1, -1);  arg381_1 = None
        unsqueeze_1209 = torch.ops.aten.unsqueeze.default(unsqueeze_1208, -1);  unsqueeze_1208 = None
        unsqueeze_1210 = torch.ops.aten.unsqueeze.default(mul_466, -1);  mul_466 = None
        unsqueeze_1211 = torch.ops.aten.unsqueeze.default(unsqueeze_1210, -1);  unsqueeze_1210 = None
        sub_151 = torch.ops.aten.sub.Tensor(convolution_178, unsqueeze_1209);  convolution_178 = unsqueeze_1209 = None
        mul_467 = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_1211);  sub_151 = unsqueeze_1211 = None
        unsqueeze_1212 = torch.ops.aten.unsqueeze.default(arg383_1, -1);  arg383_1 = None
        unsqueeze_1213 = torch.ops.aten.unsqueeze.default(unsqueeze_1212, -1);  unsqueeze_1212 = None
        mul_468 = torch.ops.aten.mul.Tensor(mul_467, unsqueeze_1213);  mul_467 = unsqueeze_1213 = None
        unsqueeze_1214 = torch.ops.aten.unsqueeze.default(arg384_1, -1);  arg384_1 = None
        unsqueeze_1215 = torch.ops.aten.unsqueeze.default(unsqueeze_1214, -1);  unsqueeze_1214 = None
        add_346 = torch.ops.aten.add.Tensor(mul_468, unsqueeze_1215);  mul_468 = unsqueeze_1215 = None
        relu_77 = torch.ops.aten.relu.default(add_346);  add_346 = None
        convolution_179 = torch.ops.aten.convolution.default(relu_77, arg385_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  arg385_1 = None
        add_347 = torch.ops.aten.add.Tensor(arg387_1, 1e-05);  arg387_1 = None
        sqrt_152 = torch.ops.aten.sqrt.default(add_347);  add_347 = None
        reciprocal_152 = torch.ops.aten.reciprocal.default(sqrt_152);  sqrt_152 = None
        mul_469 = torch.ops.aten.mul.Tensor(reciprocal_152, 1);  reciprocal_152 = None
        unsqueeze_1216 = torch.ops.aten.unsqueeze.default(arg386_1, -1);  arg386_1 = None
        unsqueeze_1217 = torch.ops.aten.unsqueeze.default(unsqueeze_1216, -1);  unsqueeze_1216 = None
        unsqueeze_1218 = torch.ops.aten.unsqueeze.default(mul_469, -1);  mul_469 = None
        unsqueeze_1219 = torch.ops.aten.unsqueeze.default(unsqueeze_1218, -1);  unsqueeze_1218 = None
        sub_152 = torch.ops.aten.sub.Tensor(convolution_179, unsqueeze_1217);  convolution_179 = unsqueeze_1217 = None
        mul_470 = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_1219);  sub_152 = unsqueeze_1219 = None
        unsqueeze_1220 = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
        unsqueeze_1221 = torch.ops.aten.unsqueeze.default(unsqueeze_1220, -1);  unsqueeze_1220 = None
        mul_471 = torch.ops.aten.mul.Tensor(mul_470, unsqueeze_1221);  mul_470 = unsqueeze_1221 = None
        unsqueeze_1222 = torch.ops.aten.unsqueeze.default(arg389_1, -1);  arg389_1 = None
        unsqueeze_1223 = torch.ops.aten.unsqueeze.default(unsqueeze_1222, -1);  unsqueeze_1222 = None
        add_348 = torch.ops.aten.add.Tensor(mul_471, unsqueeze_1223);  mul_471 = unsqueeze_1223 = None
        relu_78 = torch.ops.aten.relu.default(add_348);  add_348 = None
        cat_60 = torch.ops.aten.cat.default([relu_77, relu_78], 1);  relu_77 = relu_78 = None
        convolution_180 = torch.ops.aten.convolution.default(cat_60, arg390_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_60 = arg390_1 = None
        add_349 = torch.ops.aten.add.Tensor(arg392_1, 1e-05);  arg392_1 = None
        sqrt_153 = torch.ops.aten.sqrt.default(add_349);  add_349 = None
        reciprocal_153 = torch.ops.aten.reciprocal.default(sqrt_153);  sqrt_153 = None
        mul_472 = torch.ops.aten.mul.Tensor(reciprocal_153, 1);  reciprocal_153 = None
        unsqueeze_1224 = torch.ops.aten.unsqueeze.default(arg391_1, -1);  arg391_1 = None
        unsqueeze_1225 = torch.ops.aten.unsqueeze.default(unsqueeze_1224, -1);  unsqueeze_1224 = None
        unsqueeze_1226 = torch.ops.aten.unsqueeze.default(mul_472, -1);  mul_472 = None
        unsqueeze_1227 = torch.ops.aten.unsqueeze.default(unsqueeze_1226, -1);  unsqueeze_1226 = None
        sub_153 = torch.ops.aten.sub.Tensor(convolution_180, unsqueeze_1225);  convolution_180 = unsqueeze_1225 = None
        mul_473 = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_1227);  sub_153 = unsqueeze_1227 = None
        unsqueeze_1228 = torch.ops.aten.unsqueeze.default(arg393_1, -1);  arg393_1 = None
        unsqueeze_1229 = torch.ops.aten.unsqueeze.default(unsqueeze_1228, -1);  unsqueeze_1228 = None
        mul_474 = torch.ops.aten.mul.Tensor(mul_473, unsqueeze_1229);  mul_473 = unsqueeze_1229 = None
        unsqueeze_1230 = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
        unsqueeze_1231 = torch.ops.aten.unsqueeze.default(unsqueeze_1230, -1);  unsqueeze_1230 = None
        add_350 = torch.ops.aten.add.Tensor(mul_474, unsqueeze_1231);  mul_474 = unsqueeze_1231 = None
        convolution_181 = torch.ops.aten.convolution.default(add_350, arg395_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg395_1 = None
        add_351 = torch.ops.aten.add.Tensor(arg397_1, 1e-05);  arg397_1 = None
        sqrt_154 = torch.ops.aten.sqrt.default(add_351);  add_351 = None
        reciprocal_154 = torch.ops.aten.reciprocal.default(sqrt_154);  sqrt_154 = None
        mul_475 = torch.ops.aten.mul.Tensor(reciprocal_154, 1);  reciprocal_154 = None
        unsqueeze_1232 = torch.ops.aten.unsqueeze.default(arg396_1, -1);  arg396_1 = None
        unsqueeze_1233 = torch.ops.aten.unsqueeze.default(unsqueeze_1232, -1);  unsqueeze_1232 = None
        unsqueeze_1234 = torch.ops.aten.unsqueeze.default(mul_475, -1);  mul_475 = None
        unsqueeze_1235 = torch.ops.aten.unsqueeze.default(unsqueeze_1234, -1);  unsqueeze_1234 = None
        sub_154 = torch.ops.aten.sub.Tensor(convolution_181, unsqueeze_1233);  convolution_181 = unsqueeze_1233 = None
        mul_476 = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_1235);  sub_154 = unsqueeze_1235 = None
        unsqueeze_1236 = torch.ops.aten.unsqueeze.default(arg398_1, -1);  arg398_1 = None
        unsqueeze_1237 = torch.ops.aten.unsqueeze.default(unsqueeze_1236, -1);  unsqueeze_1236 = None
        mul_477 = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_1237);  mul_476 = unsqueeze_1237 = None
        unsqueeze_1238 = torch.ops.aten.unsqueeze.default(arg399_1, -1);  arg399_1 = None
        unsqueeze_1239 = torch.ops.aten.unsqueeze.default(unsqueeze_1238, -1);  unsqueeze_1238 = None
        add_352 = torch.ops.aten.add.Tensor(mul_477, unsqueeze_1239);  mul_477 = unsqueeze_1239 = None
        cat_61 = torch.ops.aten.cat.default([add_350, add_352], 1);  add_350 = add_352 = None
        add_353 = torch.ops.aten.add.Tensor(cat_61, add_344);  cat_61 = add_344 = None
        convolution_182 = torch.ops.aten.convolution.default(add_353, arg400_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg400_1 = None
        add_354 = torch.ops.aten.add.Tensor(arg402_1, 1e-05);  arg402_1 = None
        sqrt_155 = torch.ops.aten.sqrt.default(add_354);  add_354 = None
        reciprocal_155 = torch.ops.aten.reciprocal.default(sqrt_155);  sqrt_155 = None
        mul_478 = torch.ops.aten.mul.Tensor(reciprocal_155, 1);  reciprocal_155 = None
        unsqueeze_1240 = torch.ops.aten.unsqueeze.default(arg401_1, -1);  arg401_1 = None
        unsqueeze_1241 = torch.ops.aten.unsqueeze.default(unsqueeze_1240, -1);  unsqueeze_1240 = None
        unsqueeze_1242 = torch.ops.aten.unsqueeze.default(mul_478, -1);  mul_478 = None
        unsqueeze_1243 = torch.ops.aten.unsqueeze.default(unsqueeze_1242, -1);  unsqueeze_1242 = None
        sub_155 = torch.ops.aten.sub.Tensor(convolution_182, unsqueeze_1241);  convolution_182 = unsqueeze_1241 = None
        mul_479 = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_1243);  sub_155 = unsqueeze_1243 = None
        unsqueeze_1244 = torch.ops.aten.unsqueeze.default(arg403_1, -1);  arg403_1 = None
        unsqueeze_1245 = torch.ops.aten.unsqueeze.default(unsqueeze_1244, -1);  unsqueeze_1244 = None
        mul_480 = torch.ops.aten.mul.Tensor(mul_479, unsqueeze_1245);  mul_479 = unsqueeze_1245 = None
        unsqueeze_1246 = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
        unsqueeze_1247 = torch.ops.aten.unsqueeze.default(unsqueeze_1246, -1);  unsqueeze_1246 = None
        add_355 = torch.ops.aten.add.Tensor(mul_480, unsqueeze_1247);  mul_480 = unsqueeze_1247 = None
        relu_79 = torch.ops.aten.relu.default(add_355);  add_355 = None
        convolution_183 = torch.ops.aten.convolution.default(relu_79, arg405_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  arg405_1 = None
        add_356 = torch.ops.aten.add.Tensor(arg407_1, 1e-05);  arg407_1 = None
        sqrt_156 = torch.ops.aten.sqrt.default(add_356);  add_356 = None
        reciprocal_156 = torch.ops.aten.reciprocal.default(sqrt_156);  sqrt_156 = None
        mul_481 = torch.ops.aten.mul.Tensor(reciprocal_156, 1);  reciprocal_156 = None
        unsqueeze_1248 = torch.ops.aten.unsqueeze.default(arg406_1, -1);  arg406_1 = None
        unsqueeze_1249 = torch.ops.aten.unsqueeze.default(unsqueeze_1248, -1);  unsqueeze_1248 = None
        unsqueeze_1250 = torch.ops.aten.unsqueeze.default(mul_481, -1);  mul_481 = None
        unsqueeze_1251 = torch.ops.aten.unsqueeze.default(unsqueeze_1250, -1);  unsqueeze_1250 = None
        sub_156 = torch.ops.aten.sub.Tensor(convolution_183, unsqueeze_1249);  convolution_183 = unsqueeze_1249 = None
        mul_482 = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_1251);  sub_156 = unsqueeze_1251 = None
        unsqueeze_1252 = torch.ops.aten.unsqueeze.default(arg408_1, -1);  arg408_1 = None
        unsqueeze_1253 = torch.ops.aten.unsqueeze.default(unsqueeze_1252, -1);  unsqueeze_1252 = None
        mul_483 = torch.ops.aten.mul.Tensor(mul_482, unsqueeze_1253);  mul_482 = unsqueeze_1253 = None
        unsqueeze_1254 = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
        unsqueeze_1255 = torch.ops.aten.unsqueeze.default(unsqueeze_1254, -1);  unsqueeze_1254 = None
        add_357 = torch.ops.aten.add.Tensor(mul_483, unsqueeze_1255);  mul_483 = unsqueeze_1255 = None
        relu_80 = torch.ops.aten.relu.default(add_357);  add_357 = None
        cat_62 = torch.ops.aten.cat.default([relu_79, relu_80], 1);  relu_79 = relu_80 = None
        mean_14 = torch.ops.aten.mean.dim(cat_62, [2, 3], True)
        convolution_184 = torch.ops.aten.convolution.default(mean_14, arg410_1, arg411_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_14 = arg410_1 = arg411_1 = None
        relu_81 = torch.ops.aten.relu.default(convolution_184);  convolution_184 = None
        convolution_185 = torch.ops.aten.convolution.default(relu_81, arg412_1, arg413_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_81 = arg412_1 = arg413_1 = None
        add_358 = torch.ops.aten.add.Tensor(convolution_185, 3);  convolution_185 = None
        clamp_min_13 = torch.ops.aten.clamp_min.default(add_358, 0);  add_358 = None
        clamp_max_13 = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
        div_13 = torch.ops.aten.div.Tensor(clamp_max_13, 6);  clamp_max_13 = None
        mul_484 = torch.ops.aten.mul.Tensor(cat_62, div_13);  cat_62 = div_13 = None
        convolution_186 = torch.ops.aten.convolution.default(mul_484, arg414_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_484 = arg414_1 = None
        add_359 = torch.ops.aten.add.Tensor(arg416_1, 1e-05);  arg416_1 = None
        sqrt_157 = torch.ops.aten.sqrt.default(add_359);  add_359 = None
        reciprocal_157 = torch.ops.aten.reciprocal.default(sqrt_157);  sqrt_157 = None
        mul_485 = torch.ops.aten.mul.Tensor(reciprocal_157, 1);  reciprocal_157 = None
        unsqueeze_1256 = torch.ops.aten.unsqueeze.default(arg415_1, -1);  arg415_1 = None
        unsqueeze_1257 = torch.ops.aten.unsqueeze.default(unsqueeze_1256, -1);  unsqueeze_1256 = None
        unsqueeze_1258 = torch.ops.aten.unsqueeze.default(mul_485, -1);  mul_485 = None
        unsqueeze_1259 = torch.ops.aten.unsqueeze.default(unsqueeze_1258, -1);  unsqueeze_1258 = None
        sub_157 = torch.ops.aten.sub.Tensor(convolution_186, unsqueeze_1257);  convolution_186 = unsqueeze_1257 = None
        mul_486 = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_1259);  sub_157 = unsqueeze_1259 = None
        unsqueeze_1260 = torch.ops.aten.unsqueeze.default(arg417_1, -1);  arg417_1 = None
        unsqueeze_1261 = torch.ops.aten.unsqueeze.default(unsqueeze_1260, -1);  unsqueeze_1260 = None
        mul_487 = torch.ops.aten.mul.Tensor(mul_486, unsqueeze_1261);  mul_486 = unsqueeze_1261 = None
        unsqueeze_1262 = torch.ops.aten.unsqueeze.default(arg418_1, -1);  arg418_1 = None
        unsqueeze_1263 = torch.ops.aten.unsqueeze.default(unsqueeze_1262, -1);  unsqueeze_1262 = None
        add_360 = torch.ops.aten.add.Tensor(mul_487, unsqueeze_1263);  mul_487 = unsqueeze_1263 = None
        convolution_187 = torch.ops.aten.convolution.default(add_360, arg419_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg419_1 = None
        add_361 = torch.ops.aten.add.Tensor(arg421_1, 1e-05);  arg421_1 = None
        sqrt_158 = torch.ops.aten.sqrt.default(add_361);  add_361 = None
        reciprocal_158 = torch.ops.aten.reciprocal.default(sqrt_158);  sqrt_158 = None
        mul_488 = torch.ops.aten.mul.Tensor(reciprocal_158, 1);  reciprocal_158 = None
        unsqueeze_1264 = torch.ops.aten.unsqueeze.default(arg420_1, -1);  arg420_1 = None
        unsqueeze_1265 = torch.ops.aten.unsqueeze.default(unsqueeze_1264, -1);  unsqueeze_1264 = None
        unsqueeze_1266 = torch.ops.aten.unsqueeze.default(mul_488, -1);  mul_488 = None
        unsqueeze_1267 = torch.ops.aten.unsqueeze.default(unsqueeze_1266, -1);  unsqueeze_1266 = None
        sub_158 = torch.ops.aten.sub.Tensor(convolution_187, unsqueeze_1265);  convolution_187 = unsqueeze_1265 = None
        mul_489 = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_1267);  sub_158 = unsqueeze_1267 = None
        unsqueeze_1268 = torch.ops.aten.unsqueeze.default(arg422_1, -1);  arg422_1 = None
        unsqueeze_1269 = torch.ops.aten.unsqueeze.default(unsqueeze_1268, -1);  unsqueeze_1268 = None
        mul_490 = torch.ops.aten.mul.Tensor(mul_489, unsqueeze_1269);  mul_489 = unsqueeze_1269 = None
        unsqueeze_1270 = torch.ops.aten.unsqueeze.default(arg423_1, -1);  arg423_1 = None
        unsqueeze_1271 = torch.ops.aten.unsqueeze.default(unsqueeze_1270, -1);  unsqueeze_1270 = None
        add_362 = torch.ops.aten.add.Tensor(mul_490, unsqueeze_1271);  mul_490 = unsqueeze_1271 = None
        cat_63 = torch.ops.aten.cat.default([add_360, add_362], 1);  add_360 = add_362 = None
        add_363 = torch.ops.aten.add.Tensor(cat_63, add_353);  cat_63 = add_353 = None
        convolution_188 = torch.ops.aten.convolution.default(add_363, arg424_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_363 = arg424_1 = None
        add_364 = torch.ops.aten.add.Tensor(arg426_1, 1e-05);  arg426_1 = None
        sqrt_159 = torch.ops.aten.sqrt.default(add_364);  add_364 = None
        reciprocal_159 = torch.ops.aten.reciprocal.default(sqrt_159);  sqrt_159 = None
        mul_491 = torch.ops.aten.mul.Tensor(reciprocal_159, 1);  reciprocal_159 = None
        unsqueeze_1272 = torch.ops.aten.unsqueeze.default(arg425_1, -1);  arg425_1 = None
        unsqueeze_1273 = torch.ops.aten.unsqueeze.default(unsqueeze_1272, -1);  unsqueeze_1272 = None
        unsqueeze_1274 = torch.ops.aten.unsqueeze.default(mul_491, -1);  mul_491 = None
        unsqueeze_1275 = torch.ops.aten.unsqueeze.default(unsqueeze_1274, -1);  unsqueeze_1274 = None
        sub_159 = torch.ops.aten.sub.Tensor(convolution_188, unsqueeze_1273);  convolution_188 = unsqueeze_1273 = None
        mul_492 = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_1275);  sub_159 = unsqueeze_1275 = None
        unsqueeze_1276 = torch.ops.aten.unsqueeze.default(arg427_1, -1);  arg427_1 = None
        unsqueeze_1277 = torch.ops.aten.unsqueeze.default(unsqueeze_1276, -1);  unsqueeze_1276 = None
        mul_493 = torch.ops.aten.mul.Tensor(mul_492, unsqueeze_1277);  mul_492 = unsqueeze_1277 = None
        unsqueeze_1278 = torch.ops.aten.unsqueeze.default(arg428_1, -1);  arg428_1 = None
        unsqueeze_1279 = torch.ops.aten.unsqueeze.default(unsqueeze_1278, -1);  unsqueeze_1278 = None
        add_365 = torch.ops.aten.add.Tensor(mul_493, unsqueeze_1279);  mul_493 = unsqueeze_1279 = None
        relu_82 = torch.ops.aten.relu.default(add_365);  add_365 = None
        mean_15 = torch.ops.aten.mean.dim(relu_82, [-1, -2], True);  relu_82 = None
        convolution_189 = torch.ops.aten.convolution.default(mean_15, arg429_1, arg430_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_15 = arg429_1 = arg430_1 = None
        relu_83 = torch.ops.aten.relu.default(convolution_189);  convolution_189 = None
        permute_1 = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        view_3 = torch.ops.aten.view.default(relu_83, [8, 1280]);  relu_83 = None
        addmm_1 = torch.ops.aten.addmm.default(arg432_1, view_3, permute_1);  arg432_1 = view_3 = permute_1 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf0, (16, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 224, 224), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf2, (16,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf3, (16,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf4, (16,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf5, (16,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf6, (8, 16, 1, 1), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf7, (8,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf8, (8,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf9, (8,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf10, (8,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf11, (8, 1, 3, 3), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf12, (8,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf13, (8,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf14, (8,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf15, (8,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf16, (8, 16, 1, 1), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf17, (8,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf18, (8,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf19, (8,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf20, (8,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf21, (8, 1, 3, 3), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf22, (8,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf23, (8,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf24, (8,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf25, (8,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf26, (24, 16, 1, 1), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf27, (24,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf28, (24,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf29, (24,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf30, (24,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf31, (24, 1, 3, 3), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf32, (24,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf33, (24,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf34, (24,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf35, (24,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf36, (48, 1, 3, 3), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf37, (48,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf38, (48,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf39, (48,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf40, (48,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf41, (12, 48, 1, 1), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf42, (12,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf43, (12,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf44, (12,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf45, (12,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf46, (12, 1, 3, 3), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf47, (12,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf48, (12,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf49, (12,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf50, (12,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf51, (16, 1, 3, 3), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf52, (16,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf53, (16,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf54, (16,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf55, (16,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf56, (24, 16, 1, 1), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf57, (24,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf58, (24,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf59, (24,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf60, (24,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf61, (36, 24, 1, 1), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf62, (36,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf63, (36,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf64, (36,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf65, (36,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1296, device=device(type='cuda', index=0))
    reader.tensor(buf66, (36, 1, 3, 3), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf67, (36,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf68, (36,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf69, (36,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf70, (36,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf71, (12, 72, 1, 1), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf72, (12,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf73, (12,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf74, (12,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf75, (12,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 432, device=device(type='cuda', index=0))
    reader.tensor(buf76, (12, 1, 3, 3), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf77, (12,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf78, (12,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf79, (12,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf80, (12,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf81, (36, 24, 1, 1), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf82, (36,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf83, (36,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf84, (36,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf85, (36,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1296, device=device(type='cuda', index=0))
    reader.tensor(buf86, (36, 1, 3, 3), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf87, (36,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf88, (36,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf89, (36,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf90, (36,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 7200, device=device(type='cuda', index=0))
    reader.tensor(buf91, (72, 1, 5, 5), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf92, (72,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf93, (72,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf94, (72,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf95, (72,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 5760, device=device(type='cuda', index=0))
    reader.tensor(buf96, (20, 72, 1, 1), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf97, (20,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 5760, device=device(type='cuda', index=0))
    reader.tensor(buf98, (72, 20, 1, 1), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf99, (72,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 5760, device=device(type='cuda', index=0))
    reader.tensor(buf100, (20, 72, 1, 1), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf101, (20,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf102, (20,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf103, (20,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf104, (20,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 720, device=device(type='cuda', index=0))
    reader.tensor(buf105, (20, 1, 3, 3), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf106, (20,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf107, (20,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf108, (20,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf109, (20,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 2400, device=device(type='cuda', index=0))
    reader.tensor(buf110, (24, 1, 5, 5), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf111, (24,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf112, (24,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf113, (24,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf114, (24,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf115, (40, 24, 1, 1), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf116, (40,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf117, (40,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf118, (40,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf119, (40,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 9600, device=device(type='cuda', index=0))
    reader.tensor(buf120, (60, 40, 1, 1), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 240, device=device(type='cuda', index=0))
    reader.tensor(buf121, (60,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 240, device=device(type='cuda', index=0))
    reader.tensor(buf122, (60,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 240, device=device(type='cuda', index=0))
    reader.tensor(buf123, (60,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 240, device=device(type='cuda', index=0))
    reader.tensor(buf124, (60,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 2160, device=device(type='cuda', index=0))
    reader.tensor(buf125, (60, 1, 3, 3), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 240, device=device(type='cuda', index=0))
    reader.tensor(buf126, (60,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 240, device=device(type='cuda', index=0))
    reader.tensor(buf127, (60,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 240, device=device(type='cuda', index=0))
    reader.tensor(buf128, (60,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 240, device=device(type='cuda', index=0))
    reader.tensor(buf129, (60,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 15360, device=device(type='cuda', index=0))
    reader.tensor(buf130, (32, 120, 1, 1), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf131, (32,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 15360, device=device(type='cuda', index=0))
    reader.tensor(buf132, (120, 32, 1, 1), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf133, (120,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 9600, device=device(type='cuda', index=0))
    reader.tensor(buf134, (20, 120, 1, 1), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf135, (20,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf136, (20,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf137, (20,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf138, (20,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 720, device=device(type='cuda', index=0))
    reader.tensor(buf139, (20, 1, 3, 3), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf140, (20,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf141, (20,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf142, (20,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf143, (20,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf144, (120, 40, 1, 1), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf145, (120,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf146, (120,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf147, (120,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf148, (120,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 4320, device=device(type='cuda', index=0))
    reader.tensor(buf149, (120, 1, 3, 3), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf150, (120,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf151, (120,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf152, (120,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf153, (120,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 8640, device=device(type='cuda', index=0))
    reader.tensor(buf154, (240, 1, 3, 3), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf155, (240,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf156, (240,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf157, (240,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf158, (240,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf159, (40, 240, 1, 1), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf160, (40,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf161, (40,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf162, (40,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf163, (40,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf164, (40, 1, 3, 3), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf165, (40,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf166, (40,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf167, (40,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf168, (40,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf169, (40, 1, 3, 3), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf170, (40,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf171, (40,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf172, (40,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf173, (40,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 12800, device=device(type='cuda', index=0))
    reader.tensor(buf174, (80, 40, 1, 1), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf175, (80,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf176, (80,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf177, (80,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf178, (80,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 32000, device=device(type='cuda', index=0))
    reader.tensor(buf179, (100, 80, 1, 1), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf180, (100,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf181, (100,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf182, (100,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf183, (100,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf184, (100, 1, 3, 3), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf185, (100,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf186, (100,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf187, (100,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf188, (100,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 32000, device=device(type='cuda', index=0))
    reader.tensor(buf189, (40, 200, 1, 1), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf190, (40,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf191, (40,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf192, (40,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf193, (40,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf194, (40, 1, 3, 3), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf195, (40,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf196, (40,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf197, (40,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf198, (40,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 29440, device=device(type='cuda', index=0))
    reader.tensor(buf199, (92, 80, 1, 1), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf200, (92,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf201, (92,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf202, (92,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf203, (92,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 3312, device=device(type='cuda', index=0))
    reader.tensor(buf204, (92, 1, 3, 3), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf205, (92,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf206, (92,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf207, (92,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf208, (92,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 29440, device=device(type='cuda', index=0))
    reader.tensor(buf209, (40, 184, 1, 1), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf210, (40,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf211, (40,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf212, (40,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf213, (40,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf214, (40, 1, 3, 3), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf215, (40,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf216, (40,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf217, (40,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf218, (40,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 29440, device=device(type='cuda', index=0))
    reader.tensor(buf219, (92, 80, 1, 1), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf220, (92,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf221, (92,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf222, (92,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf223, (92,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 3312, device=device(type='cuda', index=0))
    reader.tensor(buf224, (92, 1, 3, 3), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf225, (92,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf226, (92,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf227, (92,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 368, device=device(type='cuda', index=0))
    reader.tensor(buf228, (92,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 29440, device=device(type='cuda', index=0))
    reader.tensor(buf229, (40, 184, 1, 1), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf230, (40,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf231, (40,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf232, (40,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf233, (40,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 1440, device=device(type='cuda', index=0))
    reader.tensor(buf234, (40, 1, 3, 3), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf235, (40,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf236, (40,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf237, (40,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf238, (40,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf239, (240, 80, 1, 1), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf240, (240,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf241, (240,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf242, (240,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf243, (240,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 8640, device=device(type='cuda', index=0))
    reader.tensor(buf244, (240, 1, 3, 3), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf245, (240,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf246, (240,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf247, (240,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf248, (240,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 230400, device=device(type='cuda', index=0))
    reader.tensor(buf249, (120, 480, 1, 1), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf250, (120,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 230400, device=device(type='cuda', index=0))
    reader.tensor(buf251, (480, 120, 1, 1), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf252, (480,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 107520, device=device(type='cuda', index=0))
    reader.tensor(buf253, (56, 480, 1, 1), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf254, (56,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf255, (56,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf256, (56,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf257, (56,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 2016, device=device(type='cuda', index=0))
    reader.tensor(buf258, (56, 1, 3, 3), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf259, (56,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf260, (56,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf261, (56,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf262, (56,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf263, (80, 1, 3, 3), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf264, (80,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf265, (80,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf266, (80,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf267, (80,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 35840, device=device(type='cuda', index=0))
    reader.tensor(buf268, (112, 80, 1, 1), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf269, (112,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf270, (112,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf271, (112,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf272, (112,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf273, (336, 112, 1, 1), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf274, (336,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf275, (336,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf276, (336,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf277, (336,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 12096, device=device(type='cuda', index=0))
    reader.tensor(buf278, (336, 1, 3, 3), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf279, (336,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf280, (336,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf281, (336,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf282, (336,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 451584, device=device(type='cuda', index=0))
    reader.tensor(buf283, (168, 672, 1, 1), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 672, device=device(type='cuda', index=0))
    reader.tensor(buf284, (168,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 451584, device=device(type='cuda', index=0))
    reader.tensor(buf285, (672, 168, 1, 1), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf286, (672,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf287, (56, 672, 1, 1), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf288, (56,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf289, (56,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf290, (56,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf291, (56,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 2016, device=device(type='cuda', index=0))
    reader.tensor(buf292, (56, 1, 3, 3), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf293, (56,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf294, (56,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf295, (56,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf296, (56,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf297, (336, 112, 1, 1), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf298, (336,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf299, (336,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf300, (336,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf301, (336,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 12096, device=device(type='cuda', index=0))
    reader.tensor(buf302, (336, 1, 3, 3), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf303, (336,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf304, (336,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf305, (336,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf306, (336,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 67200, device=device(type='cuda', index=0))
    reader.tensor(buf307, (672, 1, 5, 5), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf308, (672,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf309, (672,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf310, (672,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf311, (672,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 451584, device=device(type='cuda', index=0))
    reader.tensor(buf312, (168, 672, 1, 1), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 672, device=device(type='cuda', index=0))
    reader.tensor(buf313, (168,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 451584, device=device(type='cuda', index=0))
    reader.tensor(buf314, (672, 168, 1, 1), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf315, (672,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 215040, device=device(type='cuda', index=0))
    reader.tensor(buf316, (80, 672, 1, 1), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf317, (80,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf318, (80,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf319, (80,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf320, (80,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf321, (80, 1, 3, 3), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf322, (80,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf323, (80,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf324, (80,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf325, (80,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 11200, device=device(type='cuda', index=0))
    reader.tensor(buf326, (112, 1, 5, 5), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf327, (112,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf328, (112,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf329, (112,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf330, (112,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 71680, device=device(type='cuda', index=0))
    reader.tensor(buf331, (160, 112, 1, 1), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf332, (160,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf333, (160,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf334, (160,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf335, (160,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf336, (480, 160, 1, 1), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf337, (480,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf338, (480,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf339, (480,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf340, (480,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 17280, device=device(type='cuda', index=0))
    reader.tensor(buf341, (480, 1, 3, 3), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf342, (480,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf343, (480,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf344, (480,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf345, (480,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf346, (80, 960, 1, 1), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf347, (80,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf348, (80,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf349, (80,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf350, (80,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf351, (80, 1, 3, 3), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf352, (80,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf353, (80,), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf354, (80,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf355, (80,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf356, (480, 160, 1, 1), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf357, (480,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf358, (480,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf359, (480,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf360, (480,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 17280, device=device(type='cuda', index=0))
    reader.tensor(buf361, (480, 1, 3, 3), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf362, (480,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf363, (480,), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf364, (480,), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf365, (480,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf366, (240, 960, 1, 1), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf367, (240,), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf368, (960, 240, 1, 1), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf369, (960,), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf370, (80, 960, 1, 1), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf371, (80,), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf372, (80,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf373, (80,), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf374, (80,), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf375, (80, 1, 3, 3), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf376, (80,), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf377, (80,), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf378, (80,), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf379, (80,), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf380, (480, 160, 1, 1), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf381, (480,), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf382, (480,), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf383, (480,), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf384, (480,), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 17280, device=device(type='cuda', index=0))
    reader.tensor(buf385, (480, 1, 3, 3), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf386, (480,), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf387, (480,), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf388, (480,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf389, (480,), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf390, (80, 960, 1, 1), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf391, (80,), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf392, (80,), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf393, (80,), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf394, (80,), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf395, (80, 1, 3, 3), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf396, (80,), is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf397, (80,), is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf398, (80,), is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf399, (80,), is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf400, (480, 160, 1, 1), is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf401, (480,), is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf402, (480,), is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf403, (480,), is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf404, (480,), is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 17280, device=device(type='cuda', index=0))
    reader.tensor(buf405, (480, 1, 3, 3), is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf406, (480,), is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf407, (480,), is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf408, (480,), is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf409, (480,), is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf410, (240, 960, 1, 1), is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf411, (240,), is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf412, (960, 240, 1, 1), is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf413, (960,), is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf414, (80, 960, 1, 1), is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf415, (80,), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf416, (80,), is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf417, (80,), is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf418, (80,), is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf419, (80, 1, 3, 3), is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf420, (80,), is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf421, (80,), is_leaf=True)  # arg421_1
    buf422 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf422, (80,), is_leaf=True)  # arg422_1
    buf423 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf423, (80,), is_leaf=True)  # arg423_1
    buf424 = reader.storage(None, 614400, device=device(type='cuda', index=0))
    reader.tensor(buf424, (960, 160, 1, 1), is_leaf=True)  # arg424_1
    buf425 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf425, (960,), is_leaf=True)  # arg425_1
    buf426 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf426, (960,), is_leaf=True)  # arg426_1
    buf427 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf427, (960,), is_leaf=True)  # arg427_1
    buf428 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf428, (960,), is_leaf=True)  # arg428_1
    buf429 = reader.storage(None, 4915200, device=device(type='cuda', index=0))
    reader.tensor(buf429, (1280, 960, 1, 1), is_leaf=True)  # arg429_1
    buf430 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf430, (1280,), is_leaf=True)  # arg430_1
    buf431 = reader.storage(None, 5120000, device=device(type='cuda', index=0))
    reader.tensor(buf431, (1000, 1280), is_leaf=True)  # arg431_1
    buf432 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf432, (1000,), is_leaf=True)  # arg432_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)