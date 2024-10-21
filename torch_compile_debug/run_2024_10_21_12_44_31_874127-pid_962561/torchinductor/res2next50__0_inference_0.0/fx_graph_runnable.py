
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1):
        convolution_85 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_210 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_85 = torch.ops.aten.sqrt.default(add_210);  add_210 = None
        reciprocal_85 = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_255 = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_681 = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682 = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
        unsqueeze_683 = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85 = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_681);  convolution_85 = unsqueeze_681 = None
        mul_256 = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_685 = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_685);  mul_256 = unsqueeze_685 = None
        unsqueeze_686 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_687 = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_211 = torch.ops.aten.add.Tensor(mul_257, unsqueeze_687);  mul_257 = unsqueeze_687 = None
        relu_81 = torch.ops.aten.relu.default(add_211);  add_211 = None
        _low_memory_max_pool2d_with_offsets_1 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_81, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu_81 = None
        getitem_322 = _low_memory_max_pool2d_with_offsets_1[0];  _low_memory_max_pool2d_with_offsets_1 = None
        convolution_86 = torch.ops.aten.convolution.default(getitem_322, arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg6_1 = None
        add_212 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_86 = torch.ops.aten.sqrt.default(add_212);  add_212 = None
        reciprocal_86 = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_258 = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_689 = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690 = torch.ops.aten.unsqueeze.default(mul_258, -1);  mul_258 = None
        unsqueeze_691 = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86 = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_689);  convolution_86 = unsqueeze_689 = None
        mul_259 = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_693 = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_260 = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_693);  mul_259 = unsqueeze_693 = None
        unsqueeze_694 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_695 = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_213 = torch.ops.aten.add.Tensor(mul_260, unsqueeze_695);  mul_260 = unsqueeze_695 = None
        relu_82 = torch.ops.aten.relu.default(add_213);  add_213 = None
        split_81 = torch.ops.aten.split.Tensor(relu_82, 32, 1)
        getitem_328 = split_81[0];  split_81 = None
        convolution_87 = torch.ops.aten.convolution.default(getitem_328, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_328 = arg11_1 = None
        add_214 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_87 = torch.ops.aten.sqrt.default(add_214);  add_214 = None
        reciprocal_87 = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_261 = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_697 = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698 = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
        unsqueeze_699 = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87 = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_697);  convolution_87 = unsqueeze_697 = None
        mul_262 = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_701 = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_701);  mul_262 = unsqueeze_701 = None
        unsqueeze_702 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_703 = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_215 = torch.ops.aten.add.Tensor(mul_263, unsqueeze_703);  mul_263 = unsqueeze_703 = None
        relu_83 = torch.ops.aten.relu.default(add_215);  add_215 = None
        split_82 = torch.ops.aten.split.Tensor(relu_82, 32, 1)
        getitem_333 = split_82[1];  split_82 = None
        convolution_88 = torch.ops.aten.convolution.default(getitem_333, arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_333 = arg16_1 = None
        add_216 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_88 = torch.ops.aten.sqrt.default(add_216);  add_216 = None
        reciprocal_88 = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_264 = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_705 = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706 = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
        unsqueeze_707 = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88 = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_705);  convolution_88 = unsqueeze_705 = None
        mul_265 = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_709 = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_709);  mul_265 = unsqueeze_709 = None
        unsqueeze_710 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_711 = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_217 = torch.ops.aten.add.Tensor(mul_266, unsqueeze_711);  mul_266 = unsqueeze_711 = None
        relu_84 = torch.ops.aten.relu.default(add_217);  add_217 = None
        split_83 = torch.ops.aten.split.Tensor(relu_82, 32, 1)
        getitem_338 = split_83[2];  split_83 = None
        convolution_89 = torch.ops.aten.convolution.default(getitem_338, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_338 = arg21_1 = None
        add_218 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_89 = torch.ops.aten.sqrt.default(add_218);  add_218 = None
        reciprocal_89 = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_267 = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_713 = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714 = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
        unsqueeze_715 = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89 = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_713);  convolution_89 = unsqueeze_713 = None
        mul_268 = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_717 = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_269 = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_717);  mul_268 = unsqueeze_717 = None
        unsqueeze_718 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_719 = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_219 = torch.ops.aten.add.Tensor(mul_269, unsqueeze_719);  mul_269 = unsqueeze_719 = None
        relu_85 = torch.ops.aten.relu.default(add_219);  add_219 = None
        split_84 = torch.ops.aten.split.Tensor(relu_82, 32, 1);  relu_82 = None
        getitem_343 = split_84[3];  split_84 = None
        avg_pool2d_4 = torch.ops.aten.avg_pool2d.default(getitem_343, [3, 3], [1, 1], [1, 1]);  getitem_343 = None
        cat_16 = torch.ops.aten.cat.default([relu_83, relu_84, relu_85, avg_pool2d_4], 1);  relu_83 = relu_84 = relu_85 = avg_pool2d_4 = None
        convolution_90 = torch.ops.aten.convolution.default(cat_16, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_16 = arg26_1 = None
        add_220 = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_90 = torch.ops.aten.sqrt.default(add_220);  add_220 = None
        reciprocal_90 = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_270 = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_721 = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722 = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
        unsqueeze_723 = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90 = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_721);  convolution_90 = unsqueeze_721 = None
        mul_271 = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_725 = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_725);  mul_271 = unsqueeze_725 = None
        unsqueeze_726 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_727 = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_221 = torch.ops.aten.add.Tensor(mul_272, unsqueeze_727);  mul_272 = unsqueeze_727 = None
        convolution_91 = torch.ops.aten.convolution.default(getitem_322, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_322 = arg31_1 = None
        add_222 = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_91 = torch.ops.aten.sqrt.default(add_222);  add_222 = None
        reciprocal_91 = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_273 = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728 = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_729 = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730 = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
        unsqueeze_731 = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91 = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_729);  convolution_91 = unsqueeze_729 = None
        mul_274 = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_733 = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_733);  mul_274 = unsqueeze_733 = None
        unsqueeze_734 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_735 = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_223 = torch.ops.aten.add.Tensor(mul_275, unsqueeze_735);  mul_275 = unsqueeze_735 = None
        add_224 = torch.ops.aten.add.Tensor(add_221, add_223);  add_221 = add_223 = None
        relu_86 = torch.ops.aten.relu.default(add_224);  add_224 = None
        convolution_92 = torch.ops.aten.convolution.default(relu_86, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg36_1 = None
        add_225 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_92 = torch.ops.aten.sqrt.default(add_225);  add_225 = None
        reciprocal_92 = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_276 = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_736 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_737 = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        unsqueeze_738 = torch.ops.aten.unsqueeze.default(mul_276, -1);  mul_276 = None
        unsqueeze_739 = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        sub_92 = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_737);  convolution_92 = unsqueeze_737 = None
        mul_277 = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_741 = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_278 = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_741);  mul_277 = unsqueeze_741 = None
        unsqueeze_742 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_743 = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_226 = torch.ops.aten.add.Tensor(mul_278, unsqueeze_743);  mul_278 = unsqueeze_743 = None
        relu_87 = torch.ops.aten.relu.default(add_226);  add_226 = None
        split_86 = torch.ops.aten.split.Tensor(relu_87, 32, 1)
        getitem_348 = split_86[0];  split_86 = None
        convolution_93 = torch.ops.aten.convolution.default(getitem_348, arg41_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_348 = arg41_1 = None
        add_227 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_93 = torch.ops.aten.sqrt.default(add_227);  add_227 = None
        reciprocal_93 = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_279 = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_744 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_745 = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        unsqueeze_746 = torch.ops.aten.unsqueeze.default(mul_279, -1);  mul_279 = None
        unsqueeze_747 = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        sub_93 = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_745);  convolution_93 = unsqueeze_745 = None
        mul_280 = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_749 = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_749);  mul_280 = unsqueeze_749 = None
        unsqueeze_750 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_751 = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_228 = torch.ops.aten.add.Tensor(mul_281, unsqueeze_751);  mul_281 = unsqueeze_751 = None
        relu_88 = torch.ops.aten.relu.default(add_228);  add_228 = None
        split_87 = torch.ops.aten.split.Tensor(relu_87, 32, 1)
        getitem_353 = split_87[1];  split_87 = None
        add_229 = torch.ops.aten.add.Tensor(relu_88, getitem_353);  getitem_353 = None
        convolution_94 = torch.ops.aten.convolution.default(add_229, arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_229 = arg46_1 = None
        add_230 = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_94 = torch.ops.aten.sqrt.default(add_230);  add_230 = None
        reciprocal_94 = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_282 = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_752 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_753 = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        unsqueeze_754 = torch.ops.aten.unsqueeze.default(mul_282, -1);  mul_282 = None
        unsqueeze_755 = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        sub_94 = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_753);  convolution_94 = unsqueeze_753 = None
        mul_283 = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_757 = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_757);  mul_283 = unsqueeze_757 = None
        unsqueeze_758 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_759 = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_231 = torch.ops.aten.add.Tensor(mul_284, unsqueeze_759);  mul_284 = unsqueeze_759 = None
        relu_89 = torch.ops.aten.relu.default(add_231);  add_231 = None
        split_88 = torch.ops.aten.split.Tensor(relu_87, 32, 1)
        getitem_358 = split_88[2];  split_88 = None
        add_232 = torch.ops.aten.add.Tensor(relu_89, getitem_358);  getitem_358 = None
        convolution_95 = torch.ops.aten.convolution.default(add_232, arg51_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_232 = arg51_1 = None
        add_233 = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_95 = torch.ops.aten.sqrt.default(add_233);  add_233 = None
        reciprocal_95 = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_285 = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_760 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_761 = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        unsqueeze_762 = torch.ops.aten.unsqueeze.default(mul_285, -1);  mul_285 = None
        unsqueeze_763 = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        sub_95 = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_761);  convolution_95 = unsqueeze_761 = None
        mul_286 = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_765 = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_287 = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_765);  mul_286 = unsqueeze_765 = None
        unsqueeze_766 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_767 = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_234 = torch.ops.aten.add.Tensor(mul_287, unsqueeze_767);  mul_287 = unsqueeze_767 = None
        relu_90 = torch.ops.aten.relu.default(add_234);  add_234 = None
        split_89 = torch.ops.aten.split.Tensor(relu_87, 32, 1);  relu_87 = None
        getitem_363 = split_89[3];  split_89 = None
        cat_17 = torch.ops.aten.cat.default([relu_88, relu_89, relu_90, getitem_363], 1);  relu_88 = relu_89 = relu_90 = getitem_363 = None
        convolution_96 = torch.ops.aten.convolution.default(cat_17, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_17 = arg56_1 = None
        add_235 = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_96 = torch.ops.aten.sqrt.default(add_235);  add_235 = None
        reciprocal_96 = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_288 = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_768 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_769 = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        unsqueeze_770 = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
        unsqueeze_771 = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        sub_96 = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_769);  convolution_96 = unsqueeze_769 = None
        mul_289 = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_773 = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_773);  mul_289 = unsqueeze_773 = None
        unsqueeze_774 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_775 = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_236 = torch.ops.aten.add.Tensor(mul_290, unsqueeze_775);  mul_290 = unsqueeze_775 = None
        add_237 = torch.ops.aten.add.Tensor(add_236, relu_86);  add_236 = relu_86 = None
        relu_91 = torch.ops.aten.relu.default(add_237);  add_237 = None
        convolution_97 = torch.ops.aten.convolution.default(relu_91, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg61_1 = None
        add_238 = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_97 = torch.ops.aten.sqrt.default(add_238);  add_238 = None
        reciprocal_97 = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_291 = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_776 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_777 = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        unsqueeze_778 = torch.ops.aten.unsqueeze.default(mul_291, -1);  mul_291 = None
        unsqueeze_779 = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        sub_97 = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_777);  convolution_97 = unsqueeze_777 = None
        mul_292 = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_781 = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_781);  mul_292 = unsqueeze_781 = None
        unsqueeze_782 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_783 = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_239 = torch.ops.aten.add.Tensor(mul_293, unsqueeze_783);  mul_293 = unsqueeze_783 = None
        relu_92 = torch.ops.aten.relu.default(add_239);  add_239 = None
        split_91 = torch.ops.aten.split.Tensor(relu_92, 32, 1)
        getitem_368 = split_91[0];  split_91 = None
        convolution_98 = torch.ops.aten.convolution.default(getitem_368, arg66_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_368 = arg66_1 = None
        add_240 = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_98 = torch.ops.aten.sqrt.default(add_240);  add_240 = None
        reciprocal_98 = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_294 = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_784 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_785 = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        unsqueeze_786 = torch.ops.aten.unsqueeze.default(mul_294, -1);  mul_294 = None
        unsqueeze_787 = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        sub_98 = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_785);  convolution_98 = unsqueeze_785 = None
        mul_295 = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_789 = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_296 = torch.ops.aten.mul.Tensor(mul_295, unsqueeze_789);  mul_295 = unsqueeze_789 = None
        unsqueeze_790 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_791 = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_241 = torch.ops.aten.add.Tensor(mul_296, unsqueeze_791);  mul_296 = unsqueeze_791 = None
        relu_93 = torch.ops.aten.relu.default(add_241);  add_241 = None
        split_92 = torch.ops.aten.split.Tensor(relu_92, 32, 1)
        getitem_373 = split_92[1];  split_92 = None
        add_242 = torch.ops.aten.add.Tensor(relu_93, getitem_373);  getitem_373 = None
        convolution_99 = torch.ops.aten.convolution.default(add_242, arg71_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_242 = arg71_1 = None
        add_243 = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_99 = torch.ops.aten.sqrt.default(add_243);  add_243 = None
        reciprocal_99 = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_297 = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_792 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_793 = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        unsqueeze_794 = torch.ops.aten.unsqueeze.default(mul_297, -1);  mul_297 = None
        unsqueeze_795 = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        sub_99 = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_793);  convolution_99 = unsqueeze_793 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_797 = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, unsqueeze_797);  mul_298 = unsqueeze_797 = None
        unsqueeze_798 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_799 = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_244 = torch.ops.aten.add.Tensor(mul_299, unsqueeze_799);  mul_299 = unsqueeze_799 = None
        relu_94 = torch.ops.aten.relu.default(add_244);  add_244 = None
        split_93 = torch.ops.aten.split.Tensor(relu_92, 32, 1)
        getitem_378 = split_93[2];  split_93 = None
        add_245 = torch.ops.aten.add.Tensor(relu_94, getitem_378);  getitem_378 = None
        convolution_100 = torch.ops.aten.convolution.default(add_245, arg76_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_245 = arg76_1 = None
        add_246 = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_100 = torch.ops.aten.sqrt.default(add_246);  add_246 = None
        reciprocal_100 = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_300 = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_800 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_801 = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        unsqueeze_802 = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
        unsqueeze_803 = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        sub_100 = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_801);  convolution_100 = unsqueeze_801 = None
        mul_301 = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_805 = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_805);  mul_301 = unsqueeze_805 = None
        unsqueeze_806 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_807 = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_247 = torch.ops.aten.add.Tensor(mul_302, unsqueeze_807);  mul_302 = unsqueeze_807 = None
        relu_95 = torch.ops.aten.relu.default(add_247);  add_247 = None
        split_94 = torch.ops.aten.split.Tensor(relu_92, 32, 1);  relu_92 = None
        getitem_383 = split_94[3];  split_94 = None
        cat_18 = torch.ops.aten.cat.default([relu_93, relu_94, relu_95, getitem_383], 1);  relu_93 = relu_94 = relu_95 = getitem_383 = None
        convolution_101 = torch.ops.aten.convolution.default(cat_18, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_18 = arg81_1 = None
        add_248 = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_101 = torch.ops.aten.sqrt.default(add_248);  add_248 = None
        reciprocal_101 = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_303 = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_808 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_809 = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        unsqueeze_810 = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
        unsqueeze_811 = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        sub_101 = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_809);  convolution_101 = unsqueeze_809 = None
        mul_304 = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_813 = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_813);  mul_304 = unsqueeze_813 = None
        unsqueeze_814 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_815 = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_249 = torch.ops.aten.add.Tensor(mul_305, unsqueeze_815);  mul_305 = unsqueeze_815 = None
        add_250 = torch.ops.aten.add.Tensor(add_249, relu_91);  add_249 = relu_91 = None
        relu_96 = torch.ops.aten.relu.default(add_250);  add_250 = None
        convolution_102 = torch.ops.aten.convolution.default(relu_96, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg86_1 = None
        add_251 = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_102 = torch.ops.aten.sqrt.default(add_251);  add_251 = None
        reciprocal_102 = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_306 = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_816 = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_817 = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        unsqueeze_818 = torch.ops.aten.unsqueeze.default(mul_306, -1);  mul_306 = None
        unsqueeze_819 = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        sub_102 = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_817);  convolution_102 = unsqueeze_817 = None
        mul_307 = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_821 = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_308 = torch.ops.aten.mul.Tensor(mul_307, unsqueeze_821);  mul_307 = unsqueeze_821 = None
        unsqueeze_822 = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_823 = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_252 = torch.ops.aten.add.Tensor(mul_308, unsqueeze_823);  mul_308 = unsqueeze_823 = None
        relu_97 = torch.ops.aten.relu.default(add_252);  add_252 = None
        split_96 = torch.ops.aten.split.Tensor(relu_97, 64, 1)
        getitem_388 = split_96[0];  split_96 = None
        convolution_103 = torch.ops.aten.convolution.default(getitem_388, arg91_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8);  getitem_388 = arg91_1 = None
        add_253 = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_103 = torch.ops.aten.sqrt.default(add_253);  add_253 = None
        reciprocal_103 = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_309 = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_824 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_825 = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        unsqueeze_826 = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
        unsqueeze_827 = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        sub_103 = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_825);  convolution_103 = unsqueeze_825 = None
        mul_310 = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_829 = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_829);  mul_310 = unsqueeze_829 = None
        unsqueeze_830 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_831 = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_254 = torch.ops.aten.add.Tensor(mul_311, unsqueeze_831);  mul_311 = unsqueeze_831 = None
        relu_98 = torch.ops.aten.relu.default(add_254);  add_254 = None
        split_97 = torch.ops.aten.split.Tensor(relu_97, 64, 1)
        getitem_393 = split_97[1];  split_97 = None
        convolution_104 = torch.ops.aten.convolution.default(getitem_393, arg96_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8);  getitem_393 = arg96_1 = None
        add_255 = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_104 = torch.ops.aten.sqrt.default(add_255);  add_255 = None
        reciprocal_104 = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_312 = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_832 = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_833 = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        unsqueeze_834 = torch.ops.aten.unsqueeze.default(mul_312, -1);  mul_312 = None
        unsqueeze_835 = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        sub_104 = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_833);  convolution_104 = unsqueeze_833 = None
        mul_313 = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836 = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_837 = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_314 = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_837);  mul_313 = unsqueeze_837 = None
        unsqueeze_838 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_839 = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_256 = torch.ops.aten.add.Tensor(mul_314, unsqueeze_839);  mul_314 = unsqueeze_839 = None
        relu_99 = torch.ops.aten.relu.default(add_256);  add_256 = None
        split_98 = torch.ops.aten.split.Tensor(relu_97, 64, 1)
        getitem_398 = split_98[2];  split_98 = None
        convolution_105 = torch.ops.aten.convolution.default(getitem_398, arg101_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8);  getitem_398 = arg101_1 = None
        add_257 = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_105 = torch.ops.aten.sqrt.default(add_257);  add_257 = None
        reciprocal_105 = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_315 = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_840 = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_841 = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        unsqueeze_842 = torch.ops.aten.unsqueeze.default(mul_315, -1);  mul_315 = None
        unsqueeze_843 = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        sub_105 = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_841);  convolution_105 = unsqueeze_841 = None
        mul_316 = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_845 = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_317 = torch.ops.aten.mul.Tensor(mul_316, unsqueeze_845);  mul_316 = unsqueeze_845 = None
        unsqueeze_846 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_847 = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_258 = torch.ops.aten.add.Tensor(mul_317, unsqueeze_847);  mul_317 = unsqueeze_847 = None
        relu_100 = torch.ops.aten.relu.default(add_258);  add_258 = None
        split_99 = torch.ops.aten.split.Tensor(relu_97, 64, 1);  relu_97 = None
        getitem_403 = split_99[3];  split_99 = None
        avg_pool2d_5 = torch.ops.aten.avg_pool2d.default(getitem_403, [3, 3], [2, 2], [1, 1]);  getitem_403 = None
        cat_19 = torch.ops.aten.cat.default([relu_98, relu_99, relu_100, avg_pool2d_5], 1);  relu_98 = relu_99 = relu_100 = avg_pool2d_5 = None
        convolution_106 = torch.ops.aten.convolution.default(cat_19, arg106_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_19 = arg106_1 = None
        add_259 = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_106 = torch.ops.aten.sqrt.default(add_259);  add_259 = None
        reciprocal_106 = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_318 = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_848 = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_849 = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        unsqueeze_850 = torch.ops.aten.unsqueeze.default(mul_318, -1);  mul_318 = None
        unsqueeze_851 = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        sub_106 = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_849);  convolution_106 = unsqueeze_849 = None
        mul_319 = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_853 = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_320 = torch.ops.aten.mul.Tensor(mul_319, unsqueeze_853);  mul_319 = unsqueeze_853 = None
        unsqueeze_854 = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_855 = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_260 = torch.ops.aten.add.Tensor(mul_320, unsqueeze_855);  mul_320 = unsqueeze_855 = None
        convolution_107 = torch.ops.aten.convolution.default(relu_96, arg111_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_96 = arg111_1 = None
        add_261 = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_107 = torch.ops.aten.sqrt.default(add_261);  add_261 = None
        reciprocal_107 = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_321 = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_856 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_857 = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        unsqueeze_858 = torch.ops.aten.unsqueeze.default(mul_321, -1);  mul_321 = None
        unsqueeze_859 = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        sub_107 = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_857);  convolution_107 = unsqueeze_857 = None
        mul_322 = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_861 = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_323 = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_861);  mul_322 = unsqueeze_861 = None
        unsqueeze_862 = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_863 = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_262 = torch.ops.aten.add.Tensor(mul_323, unsqueeze_863);  mul_323 = unsqueeze_863 = None
        add_263 = torch.ops.aten.add.Tensor(add_260, add_262);  add_260 = add_262 = None
        relu_101 = torch.ops.aten.relu.default(add_263);  add_263 = None
        convolution_108 = torch.ops.aten.convolution.default(relu_101, arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg116_1 = None
        add_264 = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
        sqrt_108 = torch.ops.aten.sqrt.default(add_264);  add_264 = None
        reciprocal_108 = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_324 = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_864 = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_865 = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        unsqueeze_866 = torch.ops.aten.unsqueeze.default(mul_324, -1);  mul_324 = None
        unsqueeze_867 = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        sub_108 = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_865);  convolution_108 = unsqueeze_865 = None
        mul_325 = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_869 = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_869);  mul_325 = unsqueeze_869 = None
        unsqueeze_870 = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_871 = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_265 = torch.ops.aten.add.Tensor(mul_326, unsqueeze_871);  mul_326 = unsqueeze_871 = None
        relu_102 = torch.ops.aten.relu.default(add_265);  add_265 = None
        split_101 = torch.ops.aten.split.Tensor(relu_102, 64, 1)
        getitem_408 = split_101[0];  split_101 = None
        convolution_109 = torch.ops.aten.convolution.default(getitem_408, arg121_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_408 = arg121_1 = None
        add_266 = torch.ops.aten.add.Tensor(arg123_1, 1e-05);  arg123_1 = None
        sqrt_109 = torch.ops.aten.sqrt.default(add_266);  add_266 = None
        reciprocal_109 = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_327 = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_872 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_873 = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        unsqueeze_874 = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
        unsqueeze_875 = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        sub_109 = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_873);  convolution_109 = unsqueeze_873 = None
        mul_328 = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_877 = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_329 = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_877);  mul_328 = unsqueeze_877 = None
        unsqueeze_878 = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_879 = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_267 = torch.ops.aten.add.Tensor(mul_329, unsqueeze_879);  mul_329 = unsqueeze_879 = None
        relu_103 = torch.ops.aten.relu.default(add_267);  add_267 = None
        split_102 = torch.ops.aten.split.Tensor(relu_102, 64, 1)
        getitem_413 = split_102[1];  split_102 = None
        add_268 = torch.ops.aten.add.Tensor(relu_103, getitem_413);  getitem_413 = None
        convolution_110 = torch.ops.aten.convolution.default(add_268, arg126_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_268 = arg126_1 = None
        add_269 = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_110 = torch.ops.aten.sqrt.default(add_269);  add_269 = None
        reciprocal_110 = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_330 = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_880 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_881 = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        unsqueeze_882 = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
        unsqueeze_883 = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        sub_110 = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_881);  convolution_110 = unsqueeze_881 = None
        mul_331 = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884 = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_885 = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_885);  mul_331 = unsqueeze_885 = None
        unsqueeze_886 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_887 = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_270 = torch.ops.aten.add.Tensor(mul_332, unsqueeze_887);  mul_332 = unsqueeze_887 = None
        relu_104 = torch.ops.aten.relu.default(add_270);  add_270 = None
        split_103 = torch.ops.aten.split.Tensor(relu_102, 64, 1)
        getitem_418 = split_103[2];  split_103 = None
        add_271 = torch.ops.aten.add.Tensor(relu_104, getitem_418);  getitem_418 = None
        convolution_111 = torch.ops.aten.convolution.default(add_271, arg131_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_271 = arg131_1 = None
        add_272 = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_111 = torch.ops.aten.sqrt.default(add_272);  add_272 = None
        reciprocal_111 = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_333 = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_888 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_889 = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        unsqueeze_890 = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
        unsqueeze_891 = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        sub_111 = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_889);  convolution_111 = unsqueeze_889 = None
        mul_334 = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892 = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_893 = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_893);  mul_334 = unsqueeze_893 = None
        unsqueeze_894 = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_895 = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_273 = torch.ops.aten.add.Tensor(mul_335, unsqueeze_895);  mul_335 = unsqueeze_895 = None
        relu_105 = torch.ops.aten.relu.default(add_273);  add_273 = None
        split_104 = torch.ops.aten.split.Tensor(relu_102, 64, 1);  relu_102 = None
        getitem_423 = split_104[3];  split_104 = None
        cat_20 = torch.ops.aten.cat.default([relu_103, relu_104, relu_105, getitem_423], 1);  relu_103 = relu_104 = relu_105 = getitem_423 = None
        convolution_112 = torch.ops.aten.convolution.default(cat_20, arg136_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_20 = arg136_1 = None
        add_274 = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
        sqrt_112 = torch.ops.aten.sqrt.default(add_274);  add_274 = None
        reciprocal_112 = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_336 = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_896 = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_897 = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        unsqueeze_898 = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
        unsqueeze_899 = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        sub_112 = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_897);  convolution_112 = unsqueeze_897 = None
        mul_337 = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_901 = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_338 = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_901);  mul_337 = unsqueeze_901 = None
        unsqueeze_902 = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_903 = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_275 = torch.ops.aten.add.Tensor(mul_338, unsqueeze_903);  mul_338 = unsqueeze_903 = None
        add_276 = torch.ops.aten.add.Tensor(add_275, relu_101);  add_275 = relu_101 = None
        relu_106 = torch.ops.aten.relu.default(add_276);  add_276 = None
        convolution_113 = torch.ops.aten.convolution.default(relu_106, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg141_1 = None
        add_277 = torch.ops.aten.add.Tensor(arg143_1, 1e-05);  arg143_1 = None
        sqrt_113 = torch.ops.aten.sqrt.default(add_277);  add_277 = None
        reciprocal_113 = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_339 = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_904 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_905 = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        unsqueeze_906 = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
        unsqueeze_907 = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        sub_113 = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_905);  convolution_113 = unsqueeze_905 = None
        mul_340 = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908 = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_909 = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_909);  mul_340 = unsqueeze_909 = None
        unsqueeze_910 = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_911 = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_278 = torch.ops.aten.add.Tensor(mul_341, unsqueeze_911);  mul_341 = unsqueeze_911 = None
        relu_107 = torch.ops.aten.relu.default(add_278);  add_278 = None
        split_106 = torch.ops.aten.split.Tensor(relu_107, 64, 1)
        getitem_428 = split_106[0];  split_106 = None
        convolution_114 = torch.ops.aten.convolution.default(getitem_428, arg146_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_428 = arg146_1 = None
        add_279 = torch.ops.aten.add.Tensor(arg148_1, 1e-05);  arg148_1 = None
        sqrt_114 = torch.ops.aten.sqrt.default(add_279);  add_279 = None
        reciprocal_114 = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_342 = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_912 = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_913 = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        unsqueeze_914 = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
        unsqueeze_915 = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        sub_114 = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_913);  convolution_114 = unsqueeze_913 = None
        mul_343 = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916 = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_917 = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_917);  mul_343 = unsqueeze_917 = None
        unsqueeze_918 = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_919 = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_280 = torch.ops.aten.add.Tensor(mul_344, unsqueeze_919);  mul_344 = unsqueeze_919 = None
        relu_108 = torch.ops.aten.relu.default(add_280);  add_280 = None
        split_107 = torch.ops.aten.split.Tensor(relu_107, 64, 1)
        getitem_433 = split_107[1];  split_107 = None
        add_281 = torch.ops.aten.add.Tensor(relu_108, getitem_433);  getitem_433 = None
        convolution_115 = torch.ops.aten.convolution.default(add_281, arg151_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_281 = arg151_1 = None
        add_282 = torch.ops.aten.add.Tensor(arg153_1, 1e-05);  arg153_1 = None
        sqrt_115 = torch.ops.aten.sqrt.default(add_282);  add_282 = None
        reciprocal_115 = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_345 = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_920 = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_921 = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        unsqueeze_922 = torch.ops.aten.unsqueeze.default(mul_345, -1);  mul_345 = None
        unsqueeze_923 = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        sub_115 = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_921);  convolution_115 = unsqueeze_921 = None
        mul_346 = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924 = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_925 = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_925);  mul_346 = unsqueeze_925 = None
        unsqueeze_926 = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_927 = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_283 = torch.ops.aten.add.Tensor(mul_347, unsqueeze_927);  mul_347 = unsqueeze_927 = None
        relu_109 = torch.ops.aten.relu.default(add_283);  add_283 = None
        split_108 = torch.ops.aten.split.Tensor(relu_107, 64, 1)
        getitem_438 = split_108[2];  split_108 = None
        add_284 = torch.ops.aten.add.Tensor(relu_109, getitem_438);  getitem_438 = None
        convolution_116 = torch.ops.aten.convolution.default(add_284, arg156_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_284 = arg156_1 = None
        add_285 = torch.ops.aten.add.Tensor(arg158_1, 1e-05);  arg158_1 = None
        sqrt_116 = torch.ops.aten.sqrt.default(add_285);  add_285 = None
        reciprocal_116 = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_348 = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        unsqueeze_928 = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_929 = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
        unsqueeze_930 = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
        unsqueeze_931 = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
        sub_116 = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_929);  convolution_116 = unsqueeze_929 = None
        mul_349 = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
        unsqueeze_932 = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_933 = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
        mul_350 = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_933);  mul_349 = unsqueeze_933 = None
        unsqueeze_934 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_935 = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
        add_286 = torch.ops.aten.add.Tensor(mul_350, unsqueeze_935);  mul_350 = unsqueeze_935 = None
        relu_110 = torch.ops.aten.relu.default(add_286);  add_286 = None
        split_109 = torch.ops.aten.split.Tensor(relu_107, 64, 1);  relu_107 = None
        getitem_443 = split_109[3];  split_109 = None
        cat_21 = torch.ops.aten.cat.default([relu_108, relu_109, relu_110, getitem_443], 1);  relu_108 = relu_109 = relu_110 = getitem_443 = None
        convolution_117 = torch.ops.aten.convolution.default(cat_21, arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_21 = arg161_1 = None
        add_287 = torch.ops.aten.add.Tensor(arg163_1, 1e-05);  arg163_1 = None
        sqrt_117 = torch.ops.aten.sqrt.default(add_287);  add_287 = None
        reciprocal_117 = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_351 = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        unsqueeze_936 = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_937 = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
        unsqueeze_938 = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
        unsqueeze_939 = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
        sub_117 = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_937);  convolution_117 = unsqueeze_937 = None
        mul_352 = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
        unsqueeze_940 = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_941 = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_941);  mul_352 = unsqueeze_941 = None
        unsqueeze_942 = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_943 = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
        add_288 = torch.ops.aten.add.Tensor(mul_353, unsqueeze_943);  mul_353 = unsqueeze_943 = None
        add_289 = torch.ops.aten.add.Tensor(add_288, relu_106);  add_288 = relu_106 = None
        relu_111 = torch.ops.aten.relu.default(add_289);  add_289 = None
        convolution_118 = torch.ops.aten.convolution.default(relu_111, arg166_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg166_1 = None
        add_290 = torch.ops.aten.add.Tensor(arg168_1, 1e-05);  arg168_1 = None
        sqrt_118 = torch.ops.aten.sqrt.default(add_290);  add_290 = None
        reciprocal_118 = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_354 = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        unsqueeze_944 = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_945 = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
        unsqueeze_946 = torch.ops.aten.unsqueeze.default(mul_354, -1);  mul_354 = None
        unsqueeze_947 = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
        sub_118 = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_945);  convolution_118 = unsqueeze_945 = None
        mul_355 = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
        unsqueeze_948 = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_949 = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
        mul_356 = torch.ops.aten.mul.Tensor(mul_355, unsqueeze_949);  mul_355 = unsqueeze_949 = None
        unsqueeze_950 = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_951 = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
        add_291 = torch.ops.aten.add.Tensor(mul_356, unsqueeze_951);  mul_356 = unsqueeze_951 = None
        relu_112 = torch.ops.aten.relu.default(add_291);  add_291 = None
        split_111 = torch.ops.aten.split.Tensor(relu_112, 64, 1)
        getitem_448 = split_111[0];  split_111 = None
        convolution_119 = torch.ops.aten.convolution.default(getitem_448, arg171_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_448 = arg171_1 = None
        add_292 = torch.ops.aten.add.Tensor(arg173_1, 1e-05);  arg173_1 = None
        sqrt_119 = torch.ops.aten.sqrt.default(add_292);  add_292 = None
        reciprocal_119 = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_357 = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        unsqueeze_952 = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_953 = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
        unsqueeze_954 = torch.ops.aten.unsqueeze.default(mul_357, -1);  mul_357 = None
        unsqueeze_955 = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
        sub_119 = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_953);  convolution_119 = unsqueeze_953 = None
        mul_358 = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
        unsqueeze_956 = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_957 = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
        mul_359 = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_957);  mul_358 = unsqueeze_957 = None
        unsqueeze_958 = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_959 = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
        add_293 = torch.ops.aten.add.Tensor(mul_359, unsqueeze_959);  mul_359 = unsqueeze_959 = None
        relu_113 = torch.ops.aten.relu.default(add_293);  add_293 = None
        split_112 = torch.ops.aten.split.Tensor(relu_112, 64, 1)
        getitem_453 = split_112[1];  split_112 = None
        add_294 = torch.ops.aten.add.Tensor(relu_113, getitem_453);  getitem_453 = None
        convolution_120 = torch.ops.aten.convolution.default(add_294, arg176_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_294 = arg176_1 = None
        add_295 = torch.ops.aten.add.Tensor(arg178_1, 1e-05);  arg178_1 = None
        sqrt_120 = torch.ops.aten.sqrt.default(add_295);  add_295 = None
        reciprocal_120 = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_360 = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        unsqueeze_960 = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_961 = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
        unsqueeze_962 = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
        unsqueeze_963 = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
        sub_120 = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_961);  convolution_120 = unsqueeze_961 = None
        mul_361 = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
        unsqueeze_964 = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_965 = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
        mul_362 = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_965);  mul_361 = unsqueeze_965 = None
        unsqueeze_966 = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_967 = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
        add_296 = torch.ops.aten.add.Tensor(mul_362, unsqueeze_967);  mul_362 = unsqueeze_967 = None
        relu_114 = torch.ops.aten.relu.default(add_296);  add_296 = None
        split_113 = torch.ops.aten.split.Tensor(relu_112, 64, 1)
        getitem_458 = split_113[2];  split_113 = None
        add_297 = torch.ops.aten.add.Tensor(relu_114, getitem_458);  getitem_458 = None
        convolution_121 = torch.ops.aten.convolution.default(add_297, arg181_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_297 = arg181_1 = None
        add_298 = torch.ops.aten.add.Tensor(arg183_1, 1e-05);  arg183_1 = None
        sqrt_121 = torch.ops.aten.sqrt.default(add_298);  add_298 = None
        reciprocal_121 = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_363 = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        unsqueeze_968 = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_969 = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
        unsqueeze_970 = torch.ops.aten.unsqueeze.default(mul_363, -1);  mul_363 = None
        unsqueeze_971 = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
        sub_121 = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_969);  convolution_121 = unsqueeze_969 = None
        mul_364 = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
        unsqueeze_972 = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_973 = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
        mul_365 = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_973);  mul_364 = unsqueeze_973 = None
        unsqueeze_974 = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_975 = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
        add_299 = torch.ops.aten.add.Tensor(mul_365, unsqueeze_975);  mul_365 = unsqueeze_975 = None
        relu_115 = torch.ops.aten.relu.default(add_299);  add_299 = None
        split_114 = torch.ops.aten.split.Tensor(relu_112, 64, 1);  relu_112 = None
        getitem_463 = split_114[3];  split_114 = None
        cat_22 = torch.ops.aten.cat.default([relu_113, relu_114, relu_115, getitem_463], 1);  relu_113 = relu_114 = relu_115 = getitem_463 = None
        convolution_122 = torch.ops.aten.convolution.default(cat_22, arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_22 = arg186_1 = None
        add_300 = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_122 = torch.ops.aten.sqrt.default(add_300);  add_300 = None
        reciprocal_122 = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_366 = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        unsqueeze_976 = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_977 = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
        unsqueeze_978 = torch.ops.aten.unsqueeze.default(mul_366, -1);  mul_366 = None
        unsqueeze_979 = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
        sub_122 = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_977);  convolution_122 = unsqueeze_977 = None
        mul_367 = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
        unsqueeze_980 = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_981 = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
        mul_368 = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_981);  mul_367 = unsqueeze_981 = None
        unsqueeze_982 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_983 = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
        add_301 = torch.ops.aten.add.Tensor(mul_368, unsqueeze_983);  mul_368 = unsqueeze_983 = None
        add_302 = torch.ops.aten.add.Tensor(add_301, relu_111);  add_301 = relu_111 = None
        relu_116 = torch.ops.aten.relu.default(add_302);  add_302 = None
        convolution_123 = torch.ops.aten.convolution.default(relu_116, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg191_1 = None
        add_303 = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_123 = torch.ops.aten.sqrt.default(add_303);  add_303 = None
        reciprocal_123 = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_369 = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        unsqueeze_984 = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_985 = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
        unsqueeze_986 = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
        unsqueeze_987 = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
        sub_123 = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_985);  convolution_123 = unsqueeze_985 = None
        mul_370 = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
        unsqueeze_988 = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_989 = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_989);  mul_370 = unsqueeze_989 = None
        unsqueeze_990 = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_991 = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
        add_304 = torch.ops.aten.add.Tensor(mul_371, unsqueeze_991);  mul_371 = unsqueeze_991 = None
        relu_117 = torch.ops.aten.relu.default(add_304);  add_304 = None
        split_116 = torch.ops.aten.split.Tensor(relu_117, 128, 1)
        getitem_468 = split_116[0];  split_116 = None
        convolution_124 = torch.ops.aten.convolution.default(getitem_468, arg196_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8);  getitem_468 = arg196_1 = None
        add_305 = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_124 = torch.ops.aten.sqrt.default(add_305);  add_305 = None
        reciprocal_124 = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
        mul_372 = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
        unsqueeze_992 = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_993 = torch.ops.aten.unsqueeze.default(unsqueeze_992, -1);  unsqueeze_992 = None
        unsqueeze_994 = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
        unsqueeze_995 = torch.ops.aten.unsqueeze.default(unsqueeze_994, -1);  unsqueeze_994 = None
        sub_124 = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_993);  convolution_124 = unsqueeze_993 = None
        mul_373 = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_995);  sub_124 = unsqueeze_995 = None
        unsqueeze_996 = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_997 = torch.ops.aten.unsqueeze.default(unsqueeze_996, -1);  unsqueeze_996 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_997);  mul_373 = unsqueeze_997 = None
        unsqueeze_998 = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_999 = torch.ops.aten.unsqueeze.default(unsqueeze_998, -1);  unsqueeze_998 = None
        add_306 = torch.ops.aten.add.Tensor(mul_374, unsqueeze_999);  mul_374 = unsqueeze_999 = None
        relu_118 = torch.ops.aten.relu.default(add_306);  add_306 = None
        split_117 = torch.ops.aten.split.Tensor(relu_117, 128, 1)
        getitem_473 = split_117[1];  split_117 = None
        convolution_125 = torch.ops.aten.convolution.default(getitem_473, arg201_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8);  getitem_473 = arg201_1 = None
        add_307 = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_125 = torch.ops.aten.sqrt.default(add_307);  add_307 = None
        reciprocal_125 = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
        mul_375 = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
        unsqueeze_1000 = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_1001 = torch.ops.aten.unsqueeze.default(unsqueeze_1000, -1);  unsqueeze_1000 = None
        unsqueeze_1002 = torch.ops.aten.unsqueeze.default(mul_375, -1);  mul_375 = None
        unsqueeze_1003 = torch.ops.aten.unsqueeze.default(unsqueeze_1002, -1);  unsqueeze_1002 = None
        sub_125 = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_1001);  convolution_125 = unsqueeze_1001 = None
        mul_376 = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_1003);  sub_125 = unsqueeze_1003 = None
        unsqueeze_1004 = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_1005 = torch.ops.aten.unsqueeze.default(unsqueeze_1004, -1);  unsqueeze_1004 = None
        mul_377 = torch.ops.aten.mul.Tensor(mul_376, unsqueeze_1005);  mul_376 = unsqueeze_1005 = None
        unsqueeze_1006 = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_1007 = torch.ops.aten.unsqueeze.default(unsqueeze_1006, -1);  unsqueeze_1006 = None
        add_308 = torch.ops.aten.add.Tensor(mul_377, unsqueeze_1007);  mul_377 = unsqueeze_1007 = None
        relu_119 = torch.ops.aten.relu.default(add_308);  add_308 = None
        split_118 = torch.ops.aten.split.Tensor(relu_117, 128, 1)
        getitem_478 = split_118[2];  split_118 = None
        convolution_126 = torch.ops.aten.convolution.default(getitem_478, arg206_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8);  getitem_478 = arg206_1 = None
        add_309 = torch.ops.aten.add.Tensor(arg208_1, 1e-05);  arg208_1 = None
        sqrt_126 = torch.ops.aten.sqrt.default(add_309);  add_309 = None
        reciprocal_126 = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
        mul_378 = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
        unsqueeze_1008 = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_1009 = torch.ops.aten.unsqueeze.default(unsqueeze_1008, -1);  unsqueeze_1008 = None
        unsqueeze_1010 = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
        unsqueeze_1011 = torch.ops.aten.unsqueeze.default(unsqueeze_1010, -1);  unsqueeze_1010 = None
        sub_126 = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_1009);  convolution_126 = unsqueeze_1009 = None
        mul_379 = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_1011);  sub_126 = unsqueeze_1011 = None
        unsqueeze_1012 = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_1013 = torch.ops.aten.unsqueeze.default(unsqueeze_1012, -1);  unsqueeze_1012 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_1013);  mul_379 = unsqueeze_1013 = None
        unsqueeze_1014 = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_1015 = torch.ops.aten.unsqueeze.default(unsqueeze_1014, -1);  unsqueeze_1014 = None
        add_310 = torch.ops.aten.add.Tensor(mul_380, unsqueeze_1015);  mul_380 = unsqueeze_1015 = None
        relu_120 = torch.ops.aten.relu.default(add_310);  add_310 = None
        split_119 = torch.ops.aten.split.Tensor(relu_117, 128, 1);  relu_117 = None
        getitem_483 = split_119[3];  split_119 = None
        avg_pool2d_6 = torch.ops.aten.avg_pool2d.default(getitem_483, [3, 3], [2, 2], [1, 1]);  getitem_483 = None
        cat_23 = torch.ops.aten.cat.default([relu_118, relu_119, relu_120, avg_pool2d_6], 1);  relu_118 = relu_119 = relu_120 = avg_pool2d_6 = None
        convolution_127 = torch.ops.aten.convolution.default(cat_23, arg211_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_23 = arg211_1 = None
        add_311 = torch.ops.aten.add.Tensor(arg213_1, 1e-05);  arg213_1 = None
        sqrt_127 = torch.ops.aten.sqrt.default(add_311);  add_311 = None
        reciprocal_127 = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
        mul_381 = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
        unsqueeze_1016 = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_1017 = torch.ops.aten.unsqueeze.default(unsqueeze_1016, -1);  unsqueeze_1016 = None
        unsqueeze_1018 = torch.ops.aten.unsqueeze.default(mul_381, -1);  mul_381 = None
        unsqueeze_1019 = torch.ops.aten.unsqueeze.default(unsqueeze_1018, -1);  unsqueeze_1018 = None
        sub_127 = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_1017);  convolution_127 = unsqueeze_1017 = None
        mul_382 = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_1019);  sub_127 = unsqueeze_1019 = None
        unsqueeze_1020 = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_1021 = torch.ops.aten.unsqueeze.default(unsqueeze_1020, -1);  unsqueeze_1020 = None
        mul_383 = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_1021);  mul_382 = unsqueeze_1021 = None
        unsqueeze_1022 = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_1023 = torch.ops.aten.unsqueeze.default(unsqueeze_1022, -1);  unsqueeze_1022 = None
        add_312 = torch.ops.aten.add.Tensor(mul_383, unsqueeze_1023);  mul_383 = unsqueeze_1023 = None
        convolution_128 = torch.ops.aten.convolution.default(relu_116, arg216_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_116 = arg216_1 = None
        add_313 = torch.ops.aten.add.Tensor(arg218_1, 1e-05);  arg218_1 = None
        sqrt_128 = torch.ops.aten.sqrt.default(add_313);  add_313 = None
        reciprocal_128 = torch.ops.aten.reciprocal.default(sqrt_128);  sqrt_128 = None
        mul_384 = torch.ops.aten.mul.Tensor(reciprocal_128, 1);  reciprocal_128 = None
        unsqueeze_1024 = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_1025 = torch.ops.aten.unsqueeze.default(unsqueeze_1024, -1);  unsqueeze_1024 = None
        unsqueeze_1026 = torch.ops.aten.unsqueeze.default(mul_384, -1);  mul_384 = None
        unsqueeze_1027 = torch.ops.aten.unsqueeze.default(unsqueeze_1026, -1);  unsqueeze_1026 = None
        sub_128 = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_1025);  convolution_128 = unsqueeze_1025 = None
        mul_385 = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_1027);  sub_128 = unsqueeze_1027 = None
        unsqueeze_1028 = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_1029 = torch.ops.aten.unsqueeze.default(unsqueeze_1028, -1);  unsqueeze_1028 = None
        mul_386 = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_1029);  mul_385 = unsqueeze_1029 = None
        unsqueeze_1030 = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_1031 = torch.ops.aten.unsqueeze.default(unsqueeze_1030, -1);  unsqueeze_1030 = None
        add_314 = torch.ops.aten.add.Tensor(mul_386, unsqueeze_1031);  mul_386 = unsqueeze_1031 = None
        add_315 = torch.ops.aten.add.Tensor(add_312, add_314);  add_312 = add_314 = None
        relu_121 = torch.ops.aten.relu.default(add_315);  add_315 = None
        convolution_129 = torch.ops.aten.convolution.default(relu_121, arg221_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg221_1 = None
        add_316 = torch.ops.aten.add.Tensor(arg223_1, 1e-05);  arg223_1 = None
        sqrt_129 = torch.ops.aten.sqrt.default(add_316);  add_316 = None
        reciprocal_129 = torch.ops.aten.reciprocal.default(sqrt_129);  sqrt_129 = None
        mul_387 = torch.ops.aten.mul.Tensor(reciprocal_129, 1);  reciprocal_129 = None
        unsqueeze_1032 = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_1033 = torch.ops.aten.unsqueeze.default(unsqueeze_1032, -1);  unsqueeze_1032 = None
        unsqueeze_1034 = torch.ops.aten.unsqueeze.default(mul_387, -1);  mul_387 = None
        unsqueeze_1035 = torch.ops.aten.unsqueeze.default(unsqueeze_1034, -1);  unsqueeze_1034 = None
        sub_129 = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_1033);  convolution_129 = unsqueeze_1033 = None
        mul_388 = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_1035);  sub_129 = unsqueeze_1035 = None
        unsqueeze_1036 = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_1037 = torch.ops.aten.unsqueeze.default(unsqueeze_1036, -1);  unsqueeze_1036 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_1037);  mul_388 = unsqueeze_1037 = None
        unsqueeze_1038 = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_1039 = torch.ops.aten.unsqueeze.default(unsqueeze_1038, -1);  unsqueeze_1038 = None
        add_317 = torch.ops.aten.add.Tensor(mul_389, unsqueeze_1039);  mul_389 = unsqueeze_1039 = None
        relu_122 = torch.ops.aten.relu.default(add_317);  add_317 = None
        split_121 = torch.ops.aten.split.Tensor(relu_122, 128, 1)
        getitem_488 = split_121[0];  split_121 = None
        convolution_130 = torch.ops.aten.convolution.default(getitem_488, arg226_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_488 = arg226_1 = None
        add_318 = torch.ops.aten.add.Tensor(arg228_1, 1e-05);  arg228_1 = None
        sqrt_130 = torch.ops.aten.sqrt.default(add_318);  add_318 = None
        reciprocal_130 = torch.ops.aten.reciprocal.default(sqrt_130);  sqrt_130 = None
        mul_390 = torch.ops.aten.mul.Tensor(reciprocal_130, 1);  reciprocal_130 = None
        unsqueeze_1040 = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_1041 = torch.ops.aten.unsqueeze.default(unsqueeze_1040, -1);  unsqueeze_1040 = None
        unsqueeze_1042 = torch.ops.aten.unsqueeze.default(mul_390, -1);  mul_390 = None
        unsqueeze_1043 = torch.ops.aten.unsqueeze.default(unsqueeze_1042, -1);  unsqueeze_1042 = None
        sub_130 = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_1041);  convolution_130 = unsqueeze_1041 = None
        mul_391 = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_1043);  sub_130 = unsqueeze_1043 = None
        unsqueeze_1044 = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_1045 = torch.ops.aten.unsqueeze.default(unsqueeze_1044, -1);  unsqueeze_1044 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_391, unsqueeze_1045);  mul_391 = unsqueeze_1045 = None
        unsqueeze_1046 = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_1047 = torch.ops.aten.unsqueeze.default(unsqueeze_1046, -1);  unsqueeze_1046 = None
        add_319 = torch.ops.aten.add.Tensor(mul_392, unsqueeze_1047);  mul_392 = unsqueeze_1047 = None
        relu_123 = torch.ops.aten.relu.default(add_319);  add_319 = None
        split_122 = torch.ops.aten.split.Tensor(relu_122, 128, 1)
        getitem_493 = split_122[1];  split_122 = None
        add_320 = torch.ops.aten.add.Tensor(relu_123, getitem_493);  getitem_493 = None
        convolution_131 = torch.ops.aten.convolution.default(add_320, arg231_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_320 = arg231_1 = None
        add_321 = torch.ops.aten.add.Tensor(arg233_1, 1e-05);  arg233_1 = None
        sqrt_131 = torch.ops.aten.sqrt.default(add_321);  add_321 = None
        reciprocal_131 = torch.ops.aten.reciprocal.default(sqrt_131);  sqrt_131 = None
        mul_393 = torch.ops.aten.mul.Tensor(reciprocal_131, 1);  reciprocal_131 = None
        unsqueeze_1048 = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_1049 = torch.ops.aten.unsqueeze.default(unsqueeze_1048, -1);  unsqueeze_1048 = None
        unsqueeze_1050 = torch.ops.aten.unsqueeze.default(mul_393, -1);  mul_393 = None
        unsqueeze_1051 = torch.ops.aten.unsqueeze.default(unsqueeze_1050, -1);  unsqueeze_1050 = None
        sub_131 = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_1049);  convolution_131 = unsqueeze_1049 = None
        mul_394 = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_1051);  sub_131 = unsqueeze_1051 = None
        unsqueeze_1052 = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_1053 = torch.ops.aten.unsqueeze.default(unsqueeze_1052, -1);  unsqueeze_1052 = None
        mul_395 = torch.ops.aten.mul.Tensor(mul_394, unsqueeze_1053);  mul_394 = unsqueeze_1053 = None
        unsqueeze_1054 = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_1055 = torch.ops.aten.unsqueeze.default(unsqueeze_1054, -1);  unsqueeze_1054 = None
        add_322 = torch.ops.aten.add.Tensor(mul_395, unsqueeze_1055);  mul_395 = unsqueeze_1055 = None
        relu_124 = torch.ops.aten.relu.default(add_322);  add_322 = None
        split_123 = torch.ops.aten.split.Tensor(relu_122, 128, 1)
        getitem_498 = split_123[2];  split_123 = None
        add_323 = torch.ops.aten.add.Tensor(relu_124, getitem_498);  getitem_498 = None
        convolution_132 = torch.ops.aten.convolution.default(add_323, arg236_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_323 = arg236_1 = None
        add_324 = torch.ops.aten.add.Tensor(arg238_1, 1e-05);  arg238_1 = None
        sqrt_132 = torch.ops.aten.sqrt.default(add_324);  add_324 = None
        reciprocal_132 = torch.ops.aten.reciprocal.default(sqrt_132);  sqrt_132 = None
        mul_396 = torch.ops.aten.mul.Tensor(reciprocal_132, 1);  reciprocal_132 = None
        unsqueeze_1056 = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_1057 = torch.ops.aten.unsqueeze.default(unsqueeze_1056, -1);  unsqueeze_1056 = None
        unsqueeze_1058 = torch.ops.aten.unsqueeze.default(mul_396, -1);  mul_396 = None
        unsqueeze_1059 = torch.ops.aten.unsqueeze.default(unsqueeze_1058, -1);  unsqueeze_1058 = None
        sub_132 = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_1057);  convolution_132 = unsqueeze_1057 = None
        mul_397 = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_1059);  sub_132 = unsqueeze_1059 = None
        unsqueeze_1060 = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_1061 = torch.ops.aten.unsqueeze.default(unsqueeze_1060, -1);  unsqueeze_1060 = None
        mul_398 = torch.ops.aten.mul.Tensor(mul_397, unsqueeze_1061);  mul_397 = unsqueeze_1061 = None
        unsqueeze_1062 = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_1063 = torch.ops.aten.unsqueeze.default(unsqueeze_1062, -1);  unsqueeze_1062 = None
        add_325 = torch.ops.aten.add.Tensor(mul_398, unsqueeze_1063);  mul_398 = unsqueeze_1063 = None
        relu_125 = torch.ops.aten.relu.default(add_325);  add_325 = None
        split_124 = torch.ops.aten.split.Tensor(relu_122, 128, 1);  relu_122 = None
        getitem_503 = split_124[3];  split_124 = None
        cat_24 = torch.ops.aten.cat.default([relu_123, relu_124, relu_125, getitem_503], 1);  relu_123 = relu_124 = relu_125 = getitem_503 = None
        convolution_133 = torch.ops.aten.convolution.default(cat_24, arg241_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_24 = arg241_1 = None
        add_326 = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_133 = torch.ops.aten.sqrt.default(add_326);  add_326 = None
        reciprocal_133 = torch.ops.aten.reciprocal.default(sqrt_133);  sqrt_133 = None
        mul_399 = torch.ops.aten.mul.Tensor(reciprocal_133, 1);  reciprocal_133 = None
        unsqueeze_1064 = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_1065 = torch.ops.aten.unsqueeze.default(unsqueeze_1064, -1);  unsqueeze_1064 = None
        unsqueeze_1066 = torch.ops.aten.unsqueeze.default(mul_399, -1);  mul_399 = None
        unsqueeze_1067 = torch.ops.aten.unsqueeze.default(unsqueeze_1066, -1);  unsqueeze_1066 = None
        sub_133 = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_1065);  convolution_133 = unsqueeze_1065 = None
        mul_400 = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_1067);  sub_133 = unsqueeze_1067 = None
        unsqueeze_1068 = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_1069 = torch.ops.aten.unsqueeze.default(unsqueeze_1068, -1);  unsqueeze_1068 = None
        mul_401 = torch.ops.aten.mul.Tensor(mul_400, unsqueeze_1069);  mul_400 = unsqueeze_1069 = None
        unsqueeze_1070 = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_1071 = torch.ops.aten.unsqueeze.default(unsqueeze_1070, -1);  unsqueeze_1070 = None
        add_327 = torch.ops.aten.add.Tensor(mul_401, unsqueeze_1071);  mul_401 = unsqueeze_1071 = None
        add_328 = torch.ops.aten.add.Tensor(add_327, relu_121);  add_327 = relu_121 = None
        relu_126 = torch.ops.aten.relu.default(add_328);  add_328 = None
        convolution_134 = torch.ops.aten.convolution.default(relu_126, arg246_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg246_1 = None
        add_329 = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
        sqrt_134 = torch.ops.aten.sqrt.default(add_329);  add_329 = None
        reciprocal_134 = torch.ops.aten.reciprocal.default(sqrt_134);  sqrt_134 = None
        mul_402 = torch.ops.aten.mul.Tensor(reciprocal_134, 1);  reciprocal_134 = None
        unsqueeze_1072 = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_1073 = torch.ops.aten.unsqueeze.default(unsqueeze_1072, -1);  unsqueeze_1072 = None
        unsqueeze_1074 = torch.ops.aten.unsqueeze.default(mul_402, -1);  mul_402 = None
        unsqueeze_1075 = torch.ops.aten.unsqueeze.default(unsqueeze_1074, -1);  unsqueeze_1074 = None
        sub_134 = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_1073);  convolution_134 = unsqueeze_1073 = None
        mul_403 = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_1075);  sub_134 = unsqueeze_1075 = None
        unsqueeze_1076 = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_1077 = torch.ops.aten.unsqueeze.default(unsqueeze_1076, -1);  unsqueeze_1076 = None
        mul_404 = torch.ops.aten.mul.Tensor(mul_403, unsqueeze_1077);  mul_403 = unsqueeze_1077 = None
        unsqueeze_1078 = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_1079 = torch.ops.aten.unsqueeze.default(unsqueeze_1078, -1);  unsqueeze_1078 = None
        add_330 = torch.ops.aten.add.Tensor(mul_404, unsqueeze_1079);  mul_404 = unsqueeze_1079 = None
        relu_127 = torch.ops.aten.relu.default(add_330);  add_330 = None
        split_126 = torch.ops.aten.split.Tensor(relu_127, 128, 1)
        getitem_508 = split_126[0];  split_126 = None
        convolution_135 = torch.ops.aten.convolution.default(getitem_508, arg251_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_508 = arg251_1 = None
        add_331 = torch.ops.aten.add.Tensor(arg253_1, 1e-05);  arg253_1 = None
        sqrt_135 = torch.ops.aten.sqrt.default(add_331);  add_331 = None
        reciprocal_135 = torch.ops.aten.reciprocal.default(sqrt_135);  sqrt_135 = None
        mul_405 = torch.ops.aten.mul.Tensor(reciprocal_135, 1);  reciprocal_135 = None
        unsqueeze_1080 = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_1081 = torch.ops.aten.unsqueeze.default(unsqueeze_1080, -1);  unsqueeze_1080 = None
        unsqueeze_1082 = torch.ops.aten.unsqueeze.default(mul_405, -1);  mul_405 = None
        unsqueeze_1083 = torch.ops.aten.unsqueeze.default(unsqueeze_1082, -1);  unsqueeze_1082 = None
        sub_135 = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_1081);  convolution_135 = unsqueeze_1081 = None
        mul_406 = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_1083);  sub_135 = unsqueeze_1083 = None
        unsqueeze_1084 = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_1085 = torch.ops.aten.unsqueeze.default(unsqueeze_1084, -1);  unsqueeze_1084 = None
        mul_407 = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_1085);  mul_406 = unsqueeze_1085 = None
        unsqueeze_1086 = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_1087 = torch.ops.aten.unsqueeze.default(unsqueeze_1086, -1);  unsqueeze_1086 = None
        add_332 = torch.ops.aten.add.Tensor(mul_407, unsqueeze_1087);  mul_407 = unsqueeze_1087 = None
        relu_128 = torch.ops.aten.relu.default(add_332);  add_332 = None
        split_127 = torch.ops.aten.split.Tensor(relu_127, 128, 1)
        getitem_513 = split_127[1];  split_127 = None
        add_333 = torch.ops.aten.add.Tensor(relu_128, getitem_513);  getitem_513 = None
        convolution_136 = torch.ops.aten.convolution.default(add_333, arg256_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_333 = arg256_1 = None
        add_334 = torch.ops.aten.add.Tensor(arg258_1, 1e-05);  arg258_1 = None
        sqrt_136 = torch.ops.aten.sqrt.default(add_334);  add_334 = None
        reciprocal_136 = torch.ops.aten.reciprocal.default(sqrt_136);  sqrt_136 = None
        mul_408 = torch.ops.aten.mul.Tensor(reciprocal_136, 1);  reciprocal_136 = None
        unsqueeze_1088 = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_1089 = torch.ops.aten.unsqueeze.default(unsqueeze_1088, -1);  unsqueeze_1088 = None
        unsqueeze_1090 = torch.ops.aten.unsqueeze.default(mul_408, -1);  mul_408 = None
        unsqueeze_1091 = torch.ops.aten.unsqueeze.default(unsqueeze_1090, -1);  unsqueeze_1090 = None
        sub_136 = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_1089);  convolution_136 = unsqueeze_1089 = None
        mul_409 = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_1091);  sub_136 = unsqueeze_1091 = None
        unsqueeze_1092 = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_1093 = torch.ops.aten.unsqueeze.default(unsqueeze_1092, -1);  unsqueeze_1092 = None
        mul_410 = torch.ops.aten.mul.Tensor(mul_409, unsqueeze_1093);  mul_409 = unsqueeze_1093 = None
        unsqueeze_1094 = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_1095 = torch.ops.aten.unsqueeze.default(unsqueeze_1094, -1);  unsqueeze_1094 = None
        add_335 = torch.ops.aten.add.Tensor(mul_410, unsqueeze_1095);  mul_410 = unsqueeze_1095 = None
        relu_129 = torch.ops.aten.relu.default(add_335);  add_335 = None
        split_128 = torch.ops.aten.split.Tensor(relu_127, 128, 1)
        getitem_518 = split_128[2];  split_128 = None
        add_336 = torch.ops.aten.add.Tensor(relu_129, getitem_518);  getitem_518 = None
        convolution_137 = torch.ops.aten.convolution.default(add_336, arg261_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_336 = arg261_1 = None
        add_337 = torch.ops.aten.add.Tensor(arg263_1, 1e-05);  arg263_1 = None
        sqrt_137 = torch.ops.aten.sqrt.default(add_337);  add_337 = None
        reciprocal_137 = torch.ops.aten.reciprocal.default(sqrt_137);  sqrt_137 = None
        mul_411 = torch.ops.aten.mul.Tensor(reciprocal_137, 1);  reciprocal_137 = None
        unsqueeze_1096 = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_1097 = torch.ops.aten.unsqueeze.default(unsqueeze_1096, -1);  unsqueeze_1096 = None
        unsqueeze_1098 = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
        unsqueeze_1099 = torch.ops.aten.unsqueeze.default(unsqueeze_1098, -1);  unsqueeze_1098 = None
        sub_137 = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_1097);  convolution_137 = unsqueeze_1097 = None
        mul_412 = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_1099);  sub_137 = unsqueeze_1099 = None
        unsqueeze_1100 = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_1101 = torch.ops.aten.unsqueeze.default(unsqueeze_1100, -1);  unsqueeze_1100 = None
        mul_413 = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_1101);  mul_412 = unsqueeze_1101 = None
        unsqueeze_1102 = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_1103 = torch.ops.aten.unsqueeze.default(unsqueeze_1102, -1);  unsqueeze_1102 = None
        add_338 = torch.ops.aten.add.Tensor(mul_413, unsqueeze_1103);  mul_413 = unsqueeze_1103 = None
        relu_130 = torch.ops.aten.relu.default(add_338);  add_338 = None
        split_129 = torch.ops.aten.split.Tensor(relu_127, 128, 1);  relu_127 = None
        getitem_523 = split_129[3];  split_129 = None
        cat_25 = torch.ops.aten.cat.default([relu_128, relu_129, relu_130, getitem_523], 1);  relu_128 = relu_129 = relu_130 = getitem_523 = None
        convolution_138 = torch.ops.aten.convolution.default(cat_25, arg266_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_25 = arg266_1 = None
        add_339 = torch.ops.aten.add.Tensor(arg268_1, 1e-05);  arg268_1 = None
        sqrt_138 = torch.ops.aten.sqrt.default(add_339);  add_339 = None
        reciprocal_138 = torch.ops.aten.reciprocal.default(sqrt_138);  sqrt_138 = None
        mul_414 = torch.ops.aten.mul.Tensor(reciprocal_138, 1);  reciprocal_138 = None
        unsqueeze_1104 = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_1105 = torch.ops.aten.unsqueeze.default(unsqueeze_1104, -1);  unsqueeze_1104 = None
        unsqueeze_1106 = torch.ops.aten.unsqueeze.default(mul_414, -1);  mul_414 = None
        unsqueeze_1107 = torch.ops.aten.unsqueeze.default(unsqueeze_1106, -1);  unsqueeze_1106 = None
        sub_138 = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_1105);  convolution_138 = unsqueeze_1105 = None
        mul_415 = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_1107);  sub_138 = unsqueeze_1107 = None
        unsqueeze_1108 = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_1109 = torch.ops.aten.unsqueeze.default(unsqueeze_1108, -1);  unsqueeze_1108 = None
        mul_416 = torch.ops.aten.mul.Tensor(mul_415, unsqueeze_1109);  mul_415 = unsqueeze_1109 = None
        unsqueeze_1110 = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_1111 = torch.ops.aten.unsqueeze.default(unsqueeze_1110, -1);  unsqueeze_1110 = None
        add_340 = torch.ops.aten.add.Tensor(mul_416, unsqueeze_1111);  mul_416 = unsqueeze_1111 = None
        add_341 = torch.ops.aten.add.Tensor(add_340, relu_126);  add_340 = relu_126 = None
        relu_131 = torch.ops.aten.relu.default(add_341);  add_341 = None
        convolution_139 = torch.ops.aten.convolution.default(relu_131, arg271_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg271_1 = None
        add_342 = torch.ops.aten.add.Tensor(arg273_1, 1e-05);  arg273_1 = None
        sqrt_139 = torch.ops.aten.sqrt.default(add_342);  add_342 = None
        reciprocal_139 = torch.ops.aten.reciprocal.default(sqrt_139);  sqrt_139 = None
        mul_417 = torch.ops.aten.mul.Tensor(reciprocal_139, 1);  reciprocal_139 = None
        unsqueeze_1112 = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_1113 = torch.ops.aten.unsqueeze.default(unsqueeze_1112, -1);  unsqueeze_1112 = None
        unsqueeze_1114 = torch.ops.aten.unsqueeze.default(mul_417, -1);  mul_417 = None
        unsqueeze_1115 = torch.ops.aten.unsqueeze.default(unsqueeze_1114, -1);  unsqueeze_1114 = None
        sub_139 = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_1113);  convolution_139 = unsqueeze_1113 = None
        mul_418 = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_1115);  sub_139 = unsqueeze_1115 = None
        unsqueeze_1116 = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_1117 = torch.ops.aten.unsqueeze.default(unsqueeze_1116, -1);  unsqueeze_1116 = None
        mul_419 = torch.ops.aten.mul.Tensor(mul_418, unsqueeze_1117);  mul_418 = unsqueeze_1117 = None
        unsqueeze_1118 = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_1119 = torch.ops.aten.unsqueeze.default(unsqueeze_1118, -1);  unsqueeze_1118 = None
        add_343 = torch.ops.aten.add.Tensor(mul_419, unsqueeze_1119);  mul_419 = unsqueeze_1119 = None
        relu_132 = torch.ops.aten.relu.default(add_343);  add_343 = None
        split_131 = torch.ops.aten.split.Tensor(relu_132, 128, 1)
        getitem_528 = split_131[0];  split_131 = None
        convolution_140 = torch.ops.aten.convolution.default(getitem_528, arg276_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_528 = arg276_1 = None
        add_344 = torch.ops.aten.add.Tensor(arg278_1, 1e-05);  arg278_1 = None
        sqrt_140 = torch.ops.aten.sqrt.default(add_344);  add_344 = None
        reciprocal_140 = torch.ops.aten.reciprocal.default(sqrt_140);  sqrt_140 = None
        mul_420 = torch.ops.aten.mul.Tensor(reciprocal_140, 1);  reciprocal_140 = None
        unsqueeze_1120 = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_1121 = torch.ops.aten.unsqueeze.default(unsqueeze_1120, -1);  unsqueeze_1120 = None
        unsqueeze_1122 = torch.ops.aten.unsqueeze.default(mul_420, -1);  mul_420 = None
        unsqueeze_1123 = torch.ops.aten.unsqueeze.default(unsqueeze_1122, -1);  unsqueeze_1122 = None
        sub_140 = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_1121);  convolution_140 = unsqueeze_1121 = None
        mul_421 = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_1123);  sub_140 = unsqueeze_1123 = None
        unsqueeze_1124 = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_1125 = torch.ops.aten.unsqueeze.default(unsqueeze_1124, -1);  unsqueeze_1124 = None
        mul_422 = torch.ops.aten.mul.Tensor(mul_421, unsqueeze_1125);  mul_421 = unsqueeze_1125 = None
        unsqueeze_1126 = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_1127 = torch.ops.aten.unsqueeze.default(unsqueeze_1126, -1);  unsqueeze_1126 = None
        add_345 = torch.ops.aten.add.Tensor(mul_422, unsqueeze_1127);  mul_422 = unsqueeze_1127 = None
        relu_133 = torch.ops.aten.relu.default(add_345);  add_345 = None
        split_132 = torch.ops.aten.split.Tensor(relu_132, 128, 1)
        getitem_533 = split_132[1];  split_132 = None
        add_346 = torch.ops.aten.add.Tensor(relu_133, getitem_533);  getitem_533 = None
        convolution_141 = torch.ops.aten.convolution.default(add_346, arg281_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_346 = arg281_1 = None
        add_347 = torch.ops.aten.add.Tensor(arg283_1, 1e-05);  arg283_1 = None
        sqrt_141 = torch.ops.aten.sqrt.default(add_347);  add_347 = None
        reciprocal_141 = torch.ops.aten.reciprocal.default(sqrt_141);  sqrt_141 = None
        mul_423 = torch.ops.aten.mul.Tensor(reciprocal_141, 1);  reciprocal_141 = None
        unsqueeze_1128 = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_1129 = torch.ops.aten.unsqueeze.default(unsqueeze_1128, -1);  unsqueeze_1128 = None
        unsqueeze_1130 = torch.ops.aten.unsqueeze.default(mul_423, -1);  mul_423 = None
        unsqueeze_1131 = torch.ops.aten.unsqueeze.default(unsqueeze_1130, -1);  unsqueeze_1130 = None
        sub_141 = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_1129);  convolution_141 = unsqueeze_1129 = None
        mul_424 = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_1131);  sub_141 = unsqueeze_1131 = None
        unsqueeze_1132 = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_1133 = torch.ops.aten.unsqueeze.default(unsqueeze_1132, -1);  unsqueeze_1132 = None
        mul_425 = torch.ops.aten.mul.Tensor(mul_424, unsqueeze_1133);  mul_424 = unsqueeze_1133 = None
        unsqueeze_1134 = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_1135 = torch.ops.aten.unsqueeze.default(unsqueeze_1134, -1);  unsqueeze_1134 = None
        add_348 = torch.ops.aten.add.Tensor(mul_425, unsqueeze_1135);  mul_425 = unsqueeze_1135 = None
        relu_134 = torch.ops.aten.relu.default(add_348);  add_348 = None
        split_133 = torch.ops.aten.split.Tensor(relu_132, 128, 1)
        getitem_538 = split_133[2];  split_133 = None
        add_349 = torch.ops.aten.add.Tensor(relu_134, getitem_538);  getitem_538 = None
        convolution_142 = torch.ops.aten.convolution.default(add_349, arg286_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_349 = arg286_1 = None
        add_350 = torch.ops.aten.add.Tensor(arg288_1, 1e-05);  arg288_1 = None
        sqrt_142 = torch.ops.aten.sqrt.default(add_350);  add_350 = None
        reciprocal_142 = torch.ops.aten.reciprocal.default(sqrt_142);  sqrt_142 = None
        mul_426 = torch.ops.aten.mul.Tensor(reciprocal_142, 1);  reciprocal_142 = None
        unsqueeze_1136 = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_1137 = torch.ops.aten.unsqueeze.default(unsqueeze_1136, -1);  unsqueeze_1136 = None
        unsqueeze_1138 = torch.ops.aten.unsqueeze.default(mul_426, -1);  mul_426 = None
        unsqueeze_1139 = torch.ops.aten.unsqueeze.default(unsqueeze_1138, -1);  unsqueeze_1138 = None
        sub_142 = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_1137);  convolution_142 = unsqueeze_1137 = None
        mul_427 = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_1139);  sub_142 = unsqueeze_1139 = None
        unsqueeze_1140 = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_1141 = torch.ops.aten.unsqueeze.default(unsqueeze_1140, -1);  unsqueeze_1140 = None
        mul_428 = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_1141);  mul_427 = unsqueeze_1141 = None
        unsqueeze_1142 = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_1143 = torch.ops.aten.unsqueeze.default(unsqueeze_1142, -1);  unsqueeze_1142 = None
        add_351 = torch.ops.aten.add.Tensor(mul_428, unsqueeze_1143);  mul_428 = unsqueeze_1143 = None
        relu_135 = torch.ops.aten.relu.default(add_351);  add_351 = None
        split_134 = torch.ops.aten.split.Tensor(relu_132, 128, 1);  relu_132 = None
        getitem_543 = split_134[3];  split_134 = None
        cat_26 = torch.ops.aten.cat.default([relu_133, relu_134, relu_135, getitem_543], 1);  relu_133 = relu_134 = relu_135 = getitem_543 = None
        convolution_143 = torch.ops.aten.convolution.default(cat_26, arg291_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_26 = arg291_1 = None
        add_352 = torch.ops.aten.add.Tensor(arg293_1, 1e-05);  arg293_1 = None
        sqrt_143 = torch.ops.aten.sqrt.default(add_352);  add_352 = None
        reciprocal_143 = torch.ops.aten.reciprocal.default(sqrt_143);  sqrt_143 = None
        mul_429 = torch.ops.aten.mul.Tensor(reciprocal_143, 1);  reciprocal_143 = None
        unsqueeze_1144 = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_1145 = torch.ops.aten.unsqueeze.default(unsqueeze_1144, -1);  unsqueeze_1144 = None
        unsqueeze_1146 = torch.ops.aten.unsqueeze.default(mul_429, -1);  mul_429 = None
        unsqueeze_1147 = torch.ops.aten.unsqueeze.default(unsqueeze_1146, -1);  unsqueeze_1146 = None
        sub_143 = torch.ops.aten.sub.Tensor(convolution_143, unsqueeze_1145);  convolution_143 = unsqueeze_1145 = None
        mul_430 = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_1147);  sub_143 = unsqueeze_1147 = None
        unsqueeze_1148 = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_1149 = torch.ops.aten.unsqueeze.default(unsqueeze_1148, -1);  unsqueeze_1148 = None
        mul_431 = torch.ops.aten.mul.Tensor(mul_430, unsqueeze_1149);  mul_430 = unsqueeze_1149 = None
        unsqueeze_1150 = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_1151 = torch.ops.aten.unsqueeze.default(unsqueeze_1150, -1);  unsqueeze_1150 = None
        add_353 = torch.ops.aten.add.Tensor(mul_431, unsqueeze_1151);  mul_431 = unsqueeze_1151 = None
        add_354 = torch.ops.aten.add.Tensor(add_353, relu_131);  add_353 = relu_131 = None
        relu_136 = torch.ops.aten.relu.default(add_354);  add_354 = None
        convolution_144 = torch.ops.aten.convolution.default(relu_136, arg296_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg296_1 = None
        add_355 = torch.ops.aten.add.Tensor(arg298_1, 1e-05);  arg298_1 = None
        sqrt_144 = torch.ops.aten.sqrt.default(add_355);  add_355 = None
        reciprocal_144 = torch.ops.aten.reciprocal.default(sqrt_144);  sqrt_144 = None
        mul_432 = torch.ops.aten.mul.Tensor(reciprocal_144, 1);  reciprocal_144 = None
        unsqueeze_1152 = torch.ops.aten.unsqueeze.default(arg297_1, -1);  arg297_1 = None
        unsqueeze_1153 = torch.ops.aten.unsqueeze.default(unsqueeze_1152, -1);  unsqueeze_1152 = None
        unsqueeze_1154 = torch.ops.aten.unsqueeze.default(mul_432, -1);  mul_432 = None
        unsqueeze_1155 = torch.ops.aten.unsqueeze.default(unsqueeze_1154, -1);  unsqueeze_1154 = None
        sub_144 = torch.ops.aten.sub.Tensor(convolution_144, unsqueeze_1153);  convolution_144 = unsqueeze_1153 = None
        mul_433 = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_1155);  sub_144 = unsqueeze_1155 = None
        unsqueeze_1156 = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_1157 = torch.ops.aten.unsqueeze.default(unsqueeze_1156, -1);  unsqueeze_1156 = None
        mul_434 = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_1157);  mul_433 = unsqueeze_1157 = None
        unsqueeze_1158 = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_1159 = torch.ops.aten.unsqueeze.default(unsqueeze_1158, -1);  unsqueeze_1158 = None
        add_356 = torch.ops.aten.add.Tensor(mul_434, unsqueeze_1159);  mul_434 = unsqueeze_1159 = None
        relu_137 = torch.ops.aten.relu.default(add_356);  add_356 = None
        split_136 = torch.ops.aten.split.Tensor(relu_137, 128, 1)
        getitem_548 = split_136[0];  split_136 = None
        convolution_145 = torch.ops.aten.convolution.default(getitem_548, arg301_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_548 = arg301_1 = None
        add_357 = torch.ops.aten.add.Tensor(arg303_1, 1e-05);  arg303_1 = None
        sqrt_145 = torch.ops.aten.sqrt.default(add_357);  add_357 = None
        reciprocal_145 = torch.ops.aten.reciprocal.default(sqrt_145);  sqrt_145 = None
        mul_435 = torch.ops.aten.mul.Tensor(reciprocal_145, 1);  reciprocal_145 = None
        unsqueeze_1160 = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_1161 = torch.ops.aten.unsqueeze.default(unsqueeze_1160, -1);  unsqueeze_1160 = None
        unsqueeze_1162 = torch.ops.aten.unsqueeze.default(mul_435, -1);  mul_435 = None
        unsqueeze_1163 = torch.ops.aten.unsqueeze.default(unsqueeze_1162, -1);  unsqueeze_1162 = None
        sub_145 = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_1161);  convolution_145 = unsqueeze_1161 = None
        mul_436 = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_1163);  sub_145 = unsqueeze_1163 = None
        unsqueeze_1164 = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_1165 = torch.ops.aten.unsqueeze.default(unsqueeze_1164, -1);  unsqueeze_1164 = None
        mul_437 = torch.ops.aten.mul.Tensor(mul_436, unsqueeze_1165);  mul_436 = unsqueeze_1165 = None
        unsqueeze_1166 = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_1167 = torch.ops.aten.unsqueeze.default(unsqueeze_1166, -1);  unsqueeze_1166 = None
        add_358 = torch.ops.aten.add.Tensor(mul_437, unsqueeze_1167);  mul_437 = unsqueeze_1167 = None
        relu_138 = torch.ops.aten.relu.default(add_358);  add_358 = None
        split_137 = torch.ops.aten.split.Tensor(relu_137, 128, 1)
        getitem_553 = split_137[1];  split_137 = None
        add_359 = torch.ops.aten.add.Tensor(relu_138, getitem_553);  getitem_553 = None
        convolution_146 = torch.ops.aten.convolution.default(add_359, arg306_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_359 = arg306_1 = None
        add_360 = torch.ops.aten.add.Tensor(arg308_1, 1e-05);  arg308_1 = None
        sqrt_146 = torch.ops.aten.sqrt.default(add_360);  add_360 = None
        reciprocal_146 = torch.ops.aten.reciprocal.default(sqrt_146);  sqrt_146 = None
        mul_438 = torch.ops.aten.mul.Tensor(reciprocal_146, 1);  reciprocal_146 = None
        unsqueeze_1168 = torch.ops.aten.unsqueeze.default(arg307_1, -1);  arg307_1 = None
        unsqueeze_1169 = torch.ops.aten.unsqueeze.default(unsqueeze_1168, -1);  unsqueeze_1168 = None
        unsqueeze_1170 = torch.ops.aten.unsqueeze.default(mul_438, -1);  mul_438 = None
        unsqueeze_1171 = torch.ops.aten.unsqueeze.default(unsqueeze_1170, -1);  unsqueeze_1170 = None
        sub_146 = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_1169);  convolution_146 = unsqueeze_1169 = None
        mul_439 = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_1171);  sub_146 = unsqueeze_1171 = None
        unsqueeze_1172 = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_1173 = torch.ops.aten.unsqueeze.default(unsqueeze_1172, -1);  unsqueeze_1172 = None
        mul_440 = torch.ops.aten.mul.Tensor(mul_439, unsqueeze_1173);  mul_439 = unsqueeze_1173 = None
        unsqueeze_1174 = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
        unsqueeze_1175 = torch.ops.aten.unsqueeze.default(unsqueeze_1174, -1);  unsqueeze_1174 = None
        add_361 = torch.ops.aten.add.Tensor(mul_440, unsqueeze_1175);  mul_440 = unsqueeze_1175 = None
        relu_139 = torch.ops.aten.relu.default(add_361);  add_361 = None
        split_138 = torch.ops.aten.split.Tensor(relu_137, 128, 1)
        getitem_558 = split_138[2];  split_138 = None
        add_362 = torch.ops.aten.add.Tensor(relu_139, getitem_558);  getitem_558 = None
        convolution_147 = torch.ops.aten.convolution.default(add_362, arg311_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_362 = arg311_1 = None
        add_363 = torch.ops.aten.add.Tensor(arg313_1, 1e-05);  arg313_1 = None
        sqrt_147 = torch.ops.aten.sqrt.default(add_363);  add_363 = None
        reciprocal_147 = torch.ops.aten.reciprocal.default(sqrt_147);  sqrt_147 = None
        mul_441 = torch.ops.aten.mul.Tensor(reciprocal_147, 1);  reciprocal_147 = None
        unsqueeze_1176 = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_1177 = torch.ops.aten.unsqueeze.default(unsqueeze_1176, -1);  unsqueeze_1176 = None
        unsqueeze_1178 = torch.ops.aten.unsqueeze.default(mul_441, -1);  mul_441 = None
        unsqueeze_1179 = torch.ops.aten.unsqueeze.default(unsqueeze_1178, -1);  unsqueeze_1178 = None
        sub_147 = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_1177);  convolution_147 = unsqueeze_1177 = None
        mul_442 = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_1179);  sub_147 = unsqueeze_1179 = None
        unsqueeze_1180 = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_1181 = torch.ops.aten.unsqueeze.default(unsqueeze_1180, -1);  unsqueeze_1180 = None
        mul_443 = torch.ops.aten.mul.Tensor(mul_442, unsqueeze_1181);  mul_442 = unsqueeze_1181 = None
        unsqueeze_1182 = torch.ops.aten.unsqueeze.default(arg315_1, -1);  arg315_1 = None
        unsqueeze_1183 = torch.ops.aten.unsqueeze.default(unsqueeze_1182, -1);  unsqueeze_1182 = None
        add_364 = torch.ops.aten.add.Tensor(mul_443, unsqueeze_1183);  mul_443 = unsqueeze_1183 = None
        relu_140 = torch.ops.aten.relu.default(add_364);  add_364 = None
        split_139 = torch.ops.aten.split.Tensor(relu_137, 128, 1);  relu_137 = None
        getitem_563 = split_139[3];  split_139 = None
        cat_27 = torch.ops.aten.cat.default([relu_138, relu_139, relu_140, getitem_563], 1);  relu_138 = relu_139 = relu_140 = getitem_563 = None
        convolution_148 = torch.ops.aten.convolution.default(cat_27, arg316_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_27 = arg316_1 = None
        add_365 = torch.ops.aten.add.Tensor(arg318_1, 1e-05);  arg318_1 = None
        sqrt_148 = torch.ops.aten.sqrt.default(add_365);  add_365 = None
        reciprocal_148 = torch.ops.aten.reciprocal.default(sqrt_148);  sqrt_148 = None
        mul_444 = torch.ops.aten.mul.Tensor(reciprocal_148, 1);  reciprocal_148 = None
        unsqueeze_1184 = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
        unsqueeze_1185 = torch.ops.aten.unsqueeze.default(unsqueeze_1184, -1);  unsqueeze_1184 = None
        unsqueeze_1186 = torch.ops.aten.unsqueeze.default(mul_444, -1);  mul_444 = None
        unsqueeze_1187 = torch.ops.aten.unsqueeze.default(unsqueeze_1186, -1);  unsqueeze_1186 = None
        sub_148 = torch.ops.aten.sub.Tensor(convolution_148, unsqueeze_1185);  convolution_148 = unsqueeze_1185 = None
        mul_445 = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_1187);  sub_148 = unsqueeze_1187 = None
        unsqueeze_1188 = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_1189 = torch.ops.aten.unsqueeze.default(unsqueeze_1188, -1);  unsqueeze_1188 = None
        mul_446 = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_1189);  mul_445 = unsqueeze_1189 = None
        unsqueeze_1190 = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_1191 = torch.ops.aten.unsqueeze.default(unsqueeze_1190, -1);  unsqueeze_1190 = None
        add_366 = torch.ops.aten.add.Tensor(mul_446, unsqueeze_1191);  mul_446 = unsqueeze_1191 = None
        add_367 = torch.ops.aten.add.Tensor(add_366, relu_136);  add_366 = relu_136 = None
        relu_141 = torch.ops.aten.relu.default(add_367);  add_367 = None
        convolution_149 = torch.ops.aten.convolution.default(relu_141, arg321_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg321_1 = None
        add_368 = torch.ops.aten.add.Tensor(arg323_1, 1e-05);  arg323_1 = None
        sqrt_149 = torch.ops.aten.sqrt.default(add_368);  add_368 = None
        reciprocal_149 = torch.ops.aten.reciprocal.default(sqrt_149);  sqrt_149 = None
        mul_447 = torch.ops.aten.mul.Tensor(reciprocal_149, 1);  reciprocal_149 = None
        unsqueeze_1192 = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_1193 = torch.ops.aten.unsqueeze.default(unsqueeze_1192, -1);  unsqueeze_1192 = None
        unsqueeze_1194 = torch.ops.aten.unsqueeze.default(mul_447, -1);  mul_447 = None
        unsqueeze_1195 = torch.ops.aten.unsqueeze.default(unsqueeze_1194, -1);  unsqueeze_1194 = None
        sub_149 = torch.ops.aten.sub.Tensor(convolution_149, unsqueeze_1193);  convolution_149 = unsqueeze_1193 = None
        mul_448 = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_1195);  sub_149 = unsqueeze_1195 = None
        unsqueeze_1196 = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_1197 = torch.ops.aten.unsqueeze.default(unsqueeze_1196, -1);  unsqueeze_1196 = None
        mul_449 = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_1197);  mul_448 = unsqueeze_1197 = None
        unsqueeze_1198 = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_1199 = torch.ops.aten.unsqueeze.default(unsqueeze_1198, -1);  unsqueeze_1198 = None
        add_369 = torch.ops.aten.add.Tensor(mul_449, unsqueeze_1199);  mul_449 = unsqueeze_1199 = None
        relu_142 = torch.ops.aten.relu.default(add_369);  add_369 = None
        split_141 = torch.ops.aten.split.Tensor(relu_142, 128, 1)
        getitem_568 = split_141[0];  split_141 = None
        convolution_150 = torch.ops.aten.convolution.default(getitem_568, arg326_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_568 = arg326_1 = None
        add_370 = torch.ops.aten.add.Tensor(arg328_1, 1e-05);  arg328_1 = None
        sqrt_150 = torch.ops.aten.sqrt.default(add_370);  add_370 = None
        reciprocal_150 = torch.ops.aten.reciprocal.default(sqrt_150);  sqrt_150 = None
        mul_450 = torch.ops.aten.mul.Tensor(reciprocal_150, 1);  reciprocal_150 = None
        unsqueeze_1200 = torch.ops.aten.unsqueeze.default(arg327_1, -1);  arg327_1 = None
        unsqueeze_1201 = torch.ops.aten.unsqueeze.default(unsqueeze_1200, -1);  unsqueeze_1200 = None
        unsqueeze_1202 = torch.ops.aten.unsqueeze.default(mul_450, -1);  mul_450 = None
        unsqueeze_1203 = torch.ops.aten.unsqueeze.default(unsqueeze_1202, -1);  unsqueeze_1202 = None
        sub_150 = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_1201);  convolution_150 = unsqueeze_1201 = None
        mul_451 = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_1203);  sub_150 = unsqueeze_1203 = None
        unsqueeze_1204 = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_1205 = torch.ops.aten.unsqueeze.default(unsqueeze_1204, -1);  unsqueeze_1204 = None
        mul_452 = torch.ops.aten.mul.Tensor(mul_451, unsqueeze_1205);  mul_451 = unsqueeze_1205 = None
        unsqueeze_1206 = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_1207 = torch.ops.aten.unsqueeze.default(unsqueeze_1206, -1);  unsqueeze_1206 = None
        add_371 = torch.ops.aten.add.Tensor(mul_452, unsqueeze_1207);  mul_452 = unsqueeze_1207 = None
        relu_143 = torch.ops.aten.relu.default(add_371);  add_371 = None
        split_142 = torch.ops.aten.split.Tensor(relu_142, 128, 1)
        getitem_573 = split_142[1];  split_142 = None
        add_372 = torch.ops.aten.add.Tensor(relu_143, getitem_573);  getitem_573 = None
        convolution_151 = torch.ops.aten.convolution.default(add_372, arg331_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_372 = arg331_1 = None
        add_373 = torch.ops.aten.add.Tensor(arg333_1, 1e-05);  arg333_1 = None
        sqrt_151 = torch.ops.aten.sqrt.default(add_373);  add_373 = None
        reciprocal_151 = torch.ops.aten.reciprocal.default(sqrt_151);  sqrt_151 = None
        mul_453 = torch.ops.aten.mul.Tensor(reciprocal_151, 1);  reciprocal_151 = None
        unsqueeze_1208 = torch.ops.aten.unsqueeze.default(arg332_1, -1);  arg332_1 = None
        unsqueeze_1209 = torch.ops.aten.unsqueeze.default(unsqueeze_1208, -1);  unsqueeze_1208 = None
        unsqueeze_1210 = torch.ops.aten.unsqueeze.default(mul_453, -1);  mul_453 = None
        unsqueeze_1211 = torch.ops.aten.unsqueeze.default(unsqueeze_1210, -1);  unsqueeze_1210 = None
        sub_151 = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_1209);  convolution_151 = unsqueeze_1209 = None
        mul_454 = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_1211);  sub_151 = unsqueeze_1211 = None
        unsqueeze_1212 = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_1213 = torch.ops.aten.unsqueeze.default(unsqueeze_1212, -1);  unsqueeze_1212 = None
        mul_455 = torch.ops.aten.mul.Tensor(mul_454, unsqueeze_1213);  mul_454 = unsqueeze_1213 = None
        unsqueeze_1214 = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
        unsqueeze_1215 = torch.ops.aten.unsqueeze.default(unsqueeze_1214, -1);  unsqueeze_1214 = None
        add_374 = torch.ops.aten.add.Tensor(mul_455, unsqueeze_1215);  mul_455 = unsqueeze_1215 = None
        relu_144 = torch.ops.aten.relu.default(add_374);  add_374 = None
        split_143 = torch.ops.aten.split.Tensor(relu_142, 128, 1)
        getitem_578 = split_143[2];  split_143 = None
        add_375 = torch.ops.aten.add.Tensor(relu_144, getitem_578);  getitem_578 = None
        convolution_152 = torch.ops.aten.convolution.default(add_375, arg336_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_375 = arg336_1 = None
        add_376 = torch.ops.aten.add.Tensor(arg338_1, 1e-05);  arg338_1 = None
        sqrt_152 = torch.ops.aten.sqrt.default(add_376);  add_376 = None
        reciprocal_152 = torch.ops.aten.reciprocal.default(sqrt_152);  sqrt_152 = None
        mul_456 = torch.ops.aten.mul.Tensor(reciprocal_152, 1);  reciprocal_152 = None
        unsqueeze_1216 = torch.ops.aten.unsqueeze.default(arg337_1, -1);  arg337_1 = None
        unsqueeze_1217 = torch.ops.aten.unsqueeze.default(unsqueeze_1216, -1);  unsqueeze_1216 = None
        unsqueeze_1218 = torch.ops.aten.unsqueeze.default(mul_456, -1);  mul_456 = None
        unsqueeze_1219 = torch.ops.aten.unsqueeze.default(unsqueeze_1218, -1);  unsqueeze_1218 = None
        sub_152 = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_1217);  convolution_152 = unsqueeze_1217 = None
        mul_457 = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_1219);  sub_152 = unsqueeze_1219 = None
        unsqueeze_1220 = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_1221 = torch.ops.aten.unsqueeze.default(unsqueeze_1220, -1);  unsqueeze_1220 = None
        mul_458 = torch.ops.aten.mul.Tensor(mul_457, unsqueeze_1221);  mul_457 = unsqueeze_1221 = None
        unsqueeze_1222 = torch.ops.aten.unsqueeze.default(arg340_1, -1);  arg340_1 = None
        unsqueeze_1223 = torch.ops.aten.unsqueeze.default(unsqueeze_1222, -1);  unsqueeze_1222 = None
        add_377 = torch.ops.aten.add.Tensor(mul_458, unsqueeze_1223);  mul_458 = unsqueeze_1223 = None
        relu_145 = torch.ops.aten.relu.default(add_377);  add_377 = None
        split_144 = torch.ops.aten.split.Tensor(relu_142, 128, 1);  relu_142 = None
        getitem_583 = split_144[3];  split_144 = None
        cat_28 = torch.ops.aten.cat.default([relu_143, relu_144, relu_145, getitem_583], 1);  relu_143 = relu_144 = relu_145 = getitem_583 = None
        convolution_153 = torch.ops.aten.convolution.default(cat_28, arg341_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_28 = arg341_1 = None
        add_378 = torch.ops.aten.add.Tensor(arg343_1, 1e-05);  arg343_1 = None
        sqrt_153 = torch.ops.aten.sqrt.default(add_378);  add_378 = None
        reciprocal_153 = torch.ops.aten.reciprocal.default(sqrt_153);  sqrt_153 = None
        mul_459 = torch.ops.aten.mul.Tensor(reciprocal_153, 1);  reciprocal_153 = None
        unsqueeze_1224 = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_1225 = torch.ops.aten.unsqueeze.default(unsqueeze_1224, -1);  unsqueeze_1224 = None
        unsqueeze_1226 = torch.ops.aten.unsqueeze.default(mul_459, -1);  mul_459 = None
        unsqueeze_1227 = torch.ops.aten.unsqueeze.default(unsqueeze_1226, -1);  unsqueeze_1226 = None
        sub_153 = torch.ops.aten.sub.Tensor(convolution_153, unsqueeze_1225);  convolution_153 = unsqueeze_1225 = None
        mul_460 = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_1227);  sub_153 = unsqueeze_1227 = None
        unsqueeze_1228 = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_1229 = torch.ops.aten.unsqueeze.default(unsqueeze_1228, -1);  unsqueeze_1228 = None
        mul_461 = torch.ops.aten.mul.Tensor(mul_460, unsqueeze_1229);  mul_460 = unsqueeze_1229 = None
        unsqueeze_1230 = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_1231 = torch.ops.aten.unsqueeze.default(unsqueeze_1230, -1);  unsqueeze_1230 = None
        add_379 = torch.ops.aten.add.Tensor(mul_461, unsqueeze_1231);  mul_461 = unsqueeze_1231 = None
        add_380 = torch.ops.aten.add.Tensor(add_379, relu_141);  add_379 = relu_141 = None
        relu_146 = torch.ops.aten.relu.default(add_380);  add_380 = None
        convolution_154 = torch.ops.aten.convolution.default(relu_146, arg346_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg346_1 = None
        add_381 = torch.ops.aten.add.Tensor(arg348_1, 1e-05);  arg348_1 = None
        sqrt_154 = torch.ops.aten.sqrt.default(add_381);  add_381 = None
        reciprocal_154 = torch.ops.aten.reciprocal.default(sqrt_154);  sqrt_154 = None
        mul_462 = torch.ops.aten.mul.Tensor(reciprocal_154, 1);  reciprocal_154 = None
        unsqueeze_1232 = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
        unsqueeze_1233 = torch.ops.aten.unsqueeze.default(unsqueeze_1232, -1);  unsqueeze_1232 = None
        unsqueeze_1234 = torch.ops.aten.unsqueeze.default(mul_462, -1);  mul_462 = None
        unsqueeze_1235 = torch.ops.aten.unsqueeze.default(unsqueeze_1234, -1);  unsqueeze_1234 = None
        sub_154 = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_1233);  convolution_154 = unsqueeze_1233 = None
        mul_463 = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_1235);  sub_154 = unsqueeze_1235 = None
        unsqueeze_1236 = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_1237 = torch.ops.aten.unsqueeze.default(unsqueeze_1236, -1);  unsqueeze_1236 = None
        mul_464 = torch.ops.aten.mul.Tensor(mul_463, unsqueeze_1237);  mul_463 = unsqueeze_1237 = None
        unsqueeze_1238 = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
        unsqueeze_1239 = torch.ops.aten.unsqueeze.default(unsqueeze_1238, -1);  unsqueeze_1238 = None
        add_382 = torch.ops.aten.add.Tensor(mul_464, unsqueeze_1239);  mul_464 = unsqueeze_1239 = None
        relu_147 = torch.ops.aten.relu.default(add_382);  add_382 = None
        split_146 = torch.ops.aten.split.Tensor(relu_147, 256, 1)
        getitem_588 = split_146[0];  split_146 = None
        convolution_155 = torch.ops.aten.convolution.default(getitem_588, arg351_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8);  getitem_588 = arg351_1 = None
        add_383 = torch.ops.aten.add.Tensor(arg353_1, 1e-05);  arg353_1 = None
        sqrt_155 = torch.ops.aten.sqrt.default(add_383);  add_383 = None
        reciprocal_155 = torch.ops.aten.reciprocal.default(sqrt_155);  sqrt_155 = None
        mul_465 = torch.ops.aten.mul.Tensor(reciprocal_155, 1);  reciprocal_155 = None
        unsqueeze_1240 = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
        unsqueeze_1241 = torch.ops.aten.unsqueeze.default(unsqueeze_1240, -1);  unsqueeze_1240 = None
        unsqueeze_1242 = torch.ops.aten.unsqueeze.default(mul_465, -1);  mul_465 = None
        unsqueeze_1243 = torch.ops.aten.unsqueeze.default(unsqueeze_1242, -1);  unsqueeze_1242 = None
        sub_155 = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_1241);  convolution_155 = unsqueeze_1241 = None
        mul_466 = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_1243);  sub_155 = unsqueeze_1243 = None
        unsqueeze_1244 = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_1245 = torch.ops.aten.unsqueeze.default(unsqueeze_1244, -1);  unsqueeze_1244 = None
        mul_467 = torch.ops.aten.mul.Tensor(mul_466, unsqueeze_1245);  mul_466 = unsqueeze_1245 = None
        unsqueeze_1246 = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
        unsqueeze_1247 = torch.ops.aten.unsqueeze.default(unsqueeze_1246, -1);  unsqueeze_1246 = None
        add_384 = torch.ops.aten.add.Tensor(mul_467, unsqueeze_1247);  mul_467 = unsqueeze_1247 = None
        relu_148 = torch.ops.aten.relu.default(add_384);  add_384 = None
        split_147 = torch.ops.aten.split.Tensor(relu_147, 256, 1)
        getitem_593 = split_147[1];  split_147 = None
        convolution_156 = torch.ops.aten.convolution.default(getitem_593, arg356_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8);  getitem_593 = arg356_1 = None
        add_385 = torch.ops.aten.add.Tensor(arg358_1, 1e-05);  arg358_1 = None
        sqrt_156 = torch.ops.aten.sqrt.default(add_385);  add_385 = None
        reciprocal_156 = torch.ops.aten.reciprocal.default(sqrt_156);  sqrt_156 = None
        mul_468 = torch.ops.aten.mul.Tensor(reciprocal_156, 1);  reciprocal_156 = None
        unsqueeze_1248 = torch.ops.aten.unsqueeze.default(arg357_1, -1);  arg357_1 = None
        unsqueeze_1249 = torch.ops.aten.unsqueeze.default(unsqueeze_1248, -1);  unsqueeze_1248 = None
        unsqueeze_1250 = torch.ops.aten.unsqueeze.default(mul_468, -1);  mul_468 = None
        unsqueeze_1251 = torch.ops.aten.unsqueeze.default(unsqueeze_1250, -1);  unsqueeze_1250 = None
        sub_156 = torch.ops.aten.sub.Tensor(convolution_156, unsqueeze_1249);  convolution_156 = unsqueeze_1249 = None
        mul_469 = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_1251);  sub_156 = unsqueeze_1251 = None
        unsqueeze_1252 = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_1253 = torch.ops.aten.unsqueeze.default(unsqueeze_1252, -1);  unsqueeze_1252 = None
        mul_470 = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_1253);  mul_469 = unsqueeze_1253 = None
        unsqueeze_1254 = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_1255 = torch.ops.aten.unsqueeze.default(unsqueeze_1254, -1);  unsqueeze_1254 = None
        add_386 = torch.ops.aten.add.Tensor(mul_470, unsqueeze_1255);  mul_470 = unsqueeze_1255 = None
        relu_149 = torch.ops.aten.relu.default(add_386);  add_386 = None
        split_148 = torch.ops.aten.split.Tensor(relu_147, 256, 1)
        getitem_598 = split_148[2];  split_148 = None
        convolution_157 = torch.ops.aten.convolution.default(getitem_598, arg361_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8);  getitem_598 = arg361_1 = None
        add_387 = torch.ops.aten.add.Tensor(arg363_1, 1e-05);  arg363_1 = None
        sqrt_157 = torch.ops.aten.sqrt.default(add_387);  add_387 = None
        reciprocal_157 = torch.ops.aten.reciprocal.default(sqrt_157);  sqrt_157 = None
        mul_471 = torch.ops.aten.mul.Tensor(reciprocal_157, 1);  reciprocal_157 = None
        unsqueeze_1256 = torch.ops.aten.unsqueeze.default(arg362_1, -1);  arg362_1 = None
        unsqueeze_1257 = torch.ops.aten.unsqueeze.default(unsqueeze_1256, -1);  unsqueeze_1256 = None
        unsqueeze_1258 = torch.ops.aten.unsqueeze.default(mul_471, -1);  mul_471 = None
        unsqueeze_1259 = torch.ops.aten.unsqueeze.default(unsqueeze_1258, -1);  unsqueeze_1258 = None
        sub_157 = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_1257);  convolution_157 = unsqueeze_1257 = None
        mul_472 = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_1259);  sub_157 = unsqueeze_1259 = None
        unsqueeze_1260 = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
        unsqueeze_1261 = torch.ops.aten.unsqueeze.default(unsqueeze_1260, -1);  unsqueeze_1260 = None
        mul_473 = torch.ops.aten.mul.Tensor(mul_472, unsqueeze_1261);  mul_472 = unsqueeze_1261 = None
        unsqueeze_1262 = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
        unsqueeze_1263 = torch.ops.aten.unsqueeze.default(unsqueeze_1262, -1);  unsqueeze_1262 = None
        add_388 = torch.ops.aten.add.Tensor(mul_473, unsqueeze_1263);  mul_473 = unsqueeze_1263 = None
        relu_150 = torch.ops.aten.relu.default(add_388);  add_388 = None
        split_149 = torch.ops.aten.split.Tensor(relu_147, 256, 1);  relu_147 = None
        getitem_603 = split_149[3];  split_149 = None
        avg_pool2d_7 = torch.ops.aten.avg_pool2d.default(getitem_603, [3, 3], [2, 2], [1, 1]);  getitem_603 = None
        cat_29 = torch.ops.aten.cat.default([relu_148, relu_149, relu_150, avg_pool2d_7], 1);  relu_148 = relu_149 = relu_150 = avg_pool2d_7 = None
        convolution_158 = torch.ops.aten.convolution.default(cat_29, arg366_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_29 = arg366_1 = None
        add_389 = torch.ops.aten.add.Tensor(arg368_1, 1e-05);  arg368_1 = None
        sqrt_158 = torch.ops.aten.sqrt.default(add_389);  add_389 = None
        reciprocal_158 = torch.ops.aten.reciprocal.default(sqrt_158);  sqrt_158 = None
        mul_474 = torch.ops.aten.mul.Tensor(reciprocal_158, 1);  reciprocal_158 = None
        unsqueeze_1264 = torch.ops.aten.unsqueeze.default(arg367_1, -1);  arg367_1 = None
        unsqueeze_1265 = torch.ops.aten.unsqueeze.default(unsqueeze_1264, -1);  unsqueeze_1264 = None
        unsqueeze_1266 = torch.ops.aten.unsqueeze.default(mul_474, -1);  mul_474 = None
        unsqueeze_1267 = torch.ops.aten.unsqueeze.default(unsqueeze_1266, -1);  unsqueeze_1266 = None
        sub_158 = torch.ops.aten.sub.Tensor(convolution_158, unsqueeze_1265);  convolution_158 = unsqueeze_1265 = None
        mul_475 = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_1267);  sub_158 = unsqueeze_1267 = None
        unsqueeze_1268 = torch.ops.aten.unsqueeze.default(arg369_1, -1);  arg369_1 = None
        unsqueeze_1269 = torch.ops.aten.unsqueeze.default(unsqueeze_1268, -1);  unsqueeze_1268 = None
        mul_476 = torch.ops.aten.mul.Tensor(mul_475, unsqueeze_1269);  mul_475 = unsqueeze_1269 = None
        unsqueeze_1270 = torch.ops.aten.unsqueeze.default(arg370_1, -1);  arg370_1 = None
        unsqueeze_1271 = torch.ops.aten.unsqueeze.default(unsqueeze_1270, -1);  unsqueeze_1270 = None
        add_390 = torch.ops.aten.add.Tensor(mul_476, unsqueeze_1271);  mul_476 = unsqueeze_1271 = None
        convolution_159 = torch.ops.aten.convolution.default(relu_146, arg371_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_146 = arg371_1 = None
        add_391 = torch.ops.aten.add.Tensor(arg373_1, 1e-05);  arg373_1 = None
        sqrt_159 = torch.ops.aten.sqrt.default(add_391);  add_391 = None
        reciprocal_159 = torch.ops.aten.reciprocal.default(sqrt_159);  sqrt_159 = None
        mul_477 = torch.ops.aten.mul.Tensor(reciprocal_159, 1);  reciprocal_159 = None
        unsqueeze_1272 = torch.ops.aten.unsqueeze.default(arg372_1, -1);  arg372_1 = None
        unsqueeze_1273 = torch.ops.aten.unsqueeze.default(unsqueeze_1272, -1);  unsqueeze_1272 = None
        unsqueeze_1274 = torch.ops.aten.unsqueeze.default(mul_477, -1);  mul_477 = None
        unsqueeze_1275 = torch.ops.aten.unsqueeze.default(unsqueeze_1274, -1);  unsqueeze_1274 = None
        sub_159 = torch.ops.aten.sub.Tensor(convolution_159, unsqueeze_1273);  convolution_159 = unsqueeze_1273 = None
        mul_478 = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_1275);  sub_159 = unsqueeze_1275 = None
        unsqueeze_1276 = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
        unsqueeze_1277 = torch.ops.aten.unsqueeze.default(unsqueeze_1276, -1);  unsqueeze_1276 = None
        mul_479 = torch.ops.aten.mul.Tensor(mul_478, unsqueeze_1277);  mul_478 = unsqueeze_1277 = None
        unsqueeze_1278 = torch.ops.aten.unsqueeze.default(arg375_1, -1);  arg375_1 = None
        unsqueeze_1279 = torch.ops.aten.unsqueeze.default(unsqueeze_1278, -1);  unsqueeze_1278 = None
        add_392 = torch.ops.aten.add.Tensor(mul_479, unsqueeze_1279);  mul_479 = unsqueeze_1279 = None
        add_393 = torch.ops.aten.add.Tensor(add_390, add_392);  add_390 = add_392 = None
        relu_151 = torch.ops.aten.relu.default(add_393);  add_393 = None
        convolution_160 = torch.ops.aten.convolution.default(relu_151, arg376_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg376_1 = None
        add_394 = torch.ops.aten.add.Tensor(arg378_1, 1e-05);  arg378_1 = None
        sqrt_160 = torch.ops.aten.sqrt.default(add_394);  add_394 = None
        reciprocal_160 = torch.ops.aten.reciprocal.default(sqrt_160);  sqrt_160 = None
        mul_480 = torch.ops.aten.mul.Tensor(reciprocal_160, 1);  reciprocal_160 = None
        unsqueeze_1280 = torch.ops.aten.unsqueeze.default(arg377_1, -1);  arg377_1 = None
        unsqueeze_1281 = torch.ops.aten.unsqueeze.default(unsqueeze_1280, -1);  unsqueeze_1280 = None
        unsqueeze_1282 = torch.ops.aten.unsqueeze.default(mul_480, -1);  mul_480 = None
        unsqueeze_1283 = torch.ops.aten.unsqueeze.default(unsqueeze_1282, -1);  unsqueeze_1282 = None
        sub_160 = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_1281);  convolution_160 = unsqueeze_1281 = None
        mul_481 = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_1283);  sub_160 = unsqueeze_1283 = None
        unsqueeze_1284 = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_1285 = torch.ops.aten.unsqueeze.default(unsqueeze_1284, -1);  unsqueeze_1284 = None
        mul_482 = torch.ops.aten.mul.Tensor(mul_481, unsqueeze_1285);  mul_481 = unsqueeze_1285 = None
        unsqueeze_1286 = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
        unsqueeze_1287 = torch.ops.aten.unsqueeze.default(unsqueeze_1286, -1);  unsqueeze_1286 = None
        add_395 = torch.ops.aten.add.Tensor(mul_482, unsqueeze_1287);  mul_482 = unsqueeze_1287 = None
        relu_152 = torch.ops.aten.relu.default(add_395);  add_395 = None
        split_151 = torch.ops.aten.split.Tensor(relu_152, 256, 1)
        getitem_608 = split_151[0];  split_151 = None
        convolution_161 = torch.ops.aten.convolution.default(getitem_608, arg381_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_608 = arg381_1 = None
        add_396 = torch.ops.aten.add.Tensor(arg383_1, 1e-05);  arg383_1 = None
        sqrt_161 = torch.ops.aten.sqrt.default(add_396);  add_396 = None
        reciprocal_161 = torch.ops.aten.reciprocal.default(sqrt_161);  sqrt_161 = None
        mul_483 = torch.ops.aten.mul.Tensor(reciprocal_161, 1);  reciprocal_161 = None
        unsqueeze_1288 = torch.ops.aten.unsqueeze.default(arg382_1, -1);  arg382_1 = None
        unsqueeze_1289 = torch.ops.aten.unsqueeze.default(unsqueeze_1288, -1);  unsqueeze_1288 = None
        unsqueeze_1290 = torch.ops.aten.unsqueeze.default(mul_483, -1);  mul_483 = None
        unsqueeze_1291 = torch.ops.aten.unsqueeze.default(unsqueeze_1290, -1);  unsqueeze_1290 = None
        sub_161 = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_1289);  convolution_161 = unsqueeze_1289 = None
        mul_484 = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_1291);  sub_161 = unsqueeze_1291 = None
        unsqueeze_1292 = torch.ops.aten.unsqueeze.default(arg384_1, -1);  arg384_1 = None
        unsqueeze_1293 = torch.ops.aten.unsqueeze.default(unsqueeze_1292, -1);  unsqueeze_1292 = None
        mul_485 = torch.ops.aten.mul.Tensor(mul_484, unsqueeze_1293);  mul_484 = unsqueeze_1293 = None
        unsqueeze_1294 = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
        unsqueeze_1295 = torch.ops.aten.unsqueeze.default(unsqueeze_1294, -1);  unsqueeze_1294 = None
        add_397 = torch.ops.aten.add.Tensor(mul_485, unsqueeze_1295);  mul_485 = unsqueeze_1295 = None
        relu_153 = torch.ops.aten.relu.default(add_397);  add_397 = None
        split_152 = torch.ops.aten.split.Tensor(relu_152, 256, 1)
        getitem_613 = split_152[1];  split_152 = None
        add_398 = torch.ops.aten.add.Tensor(relu_153, getitem_613);  getitem_613 = None
        convolution_162 = torch.ops.aten.convolution.default(add_398, arg386_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_398 = arg386_1 = None
        add_399 = torch.ops.aten.add.Tensor(arg388_1, 1e-05);  arg388_1 = None
        sqrt_162 = torch.ops.aten.sqrt.default(add_399);  add_399 = None
        reciprocal_162 = torch.ops.aten.reciprocal.default(sqrt_162);  sqrt_162 = None
        mul_486 = torch.ops.aten.mul.Tensor(reciprocal_162, 1);  reciprocal_162 = None
        unsqueeze_1296 = torch.ops.aten.unsqueeze.default(arg387_1, -1);  arg387_1 = None
        unsqueeze_1297 = torch.ops.aten.unsqueeze.default(unsqueeze_1296, -1);  unsqueeze_1296 = None
        unsqueeze_1298 = torch.ops.aten.unsqueeze.default(mul_486, -1);  mul_486 = None
        unsqueeze_1299 = torch.ops.aten.unsqueeze.default(unsqueeze_1298, -1);  unsqueeze_1298 = None
        sub_162 = torch.ops.aten.sub.Tensor(convolution_162, unsqueeze_1297);  convolution_162 = unsqueeze_1297 = None
        mul_487 = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_1299);  sub_162 = unsqueeze_1299 = None
        unsqueeze_1300 = torch.ops.aten.unsqueeze.default(arg389_1, -1);  arg389_1 = None
        unsqueeze_1301 = torch.ops.aten.unsqueeze.default(unsqueeze_1300, -1);  unsqueeze_1300 = None
        mul_488 = torch.ops.aten.mul.Tensor(mul_487, unsqueeze_1301);  mul_487 = unsqueeze_1301 = None
        unsqueeze_1302 = torch.ops.aten.unsqueeze.default(arg390_1, -1);  arg390_1 = None
        unsqueeze_1303 = torch.ops.aten.unsqueeze.default(unsqueeze_1302, -1);  unsqueeze_1302 = None
        add_400 = torch.ops.aten.add.Tensor(mul_488, unsqueeze_1303);  mul_488 = unsqueeze_1303 = None
        relu_154 = torch.ops.aten.relu.default(add_400);  add_400 = None
        split_153 = torch.ops.aten.split.Tensor(relu_152, 256, 1)
        getitem_618 = split_153[2];  split_153 = None
        add_401 = torch.ops.aten.add.Tensor(relu_154, getitem_618);  getitem_618 = None
        convolution_163 = torch.ops.aten.convolution.default(add_401, arg391_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_401 = arg391_1 = None
        add_402 = torch.ops.aten.add.Tensor(arg393_1, 1e-05);  arg393_1 = None
        sqrt_163 = torch.ops.aten.sqrt.default(add_402);  add_402 = None
        reciprocal_163 = torch.ops.aten.reciprocal.default(sqrt_163);  sqrt_163 = None
        mul_489 = torch.ops.aten.mul.Tensor(reciprocal_163, 1);  reciprocal_163 = None
        unsqueeze_1304 = torch.ops.aten.unsqueeze.default(arg392_1, -1);  arg392_1 = None
        unsqueeze_1305 = torch.ops.aten.unsqueeze.default(unsqueeze_1304, -1);  unsqueeze_1304 = None
        unsqueeze_1306 = torch.ops.aten.unsqueeze.default(mul_489, -1);  mul_489 = None
        unsqueeze_1307 = torch.ops.aten.unsqueeze.default(unsqueeze_1306, -1);  unsqueeze_1306 = None
        sub_163 = torch.ops.aten.sub.Tensor(convolution_163, unsqueeze_1305);  convolution_163 = unsqueeze_1305 = None
        mul_490 = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_1307);  sub_163 = unsqueeze_1307 = None
        unsqueeze_1308 = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
        unsqueeze_1309 = torch.ops.aten.unsqueeze.default(unsqueeze_1308, -1);  unsqueeze_1308 = None
        mul_491 = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_1309);  mul_490 = unsqueeze_1309 = None
        unsqueeze_1310 = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
        unsqueeze_1311 = torch.ops.aten.unsqueeze.default(unsqueeze_1310, -1);  unsqueeze_1310 = None
        add_403 = torch.ops.aten.add.Tensor(mul_491, unsqueeze_1311);  mul_491 = unsqueeze_1311 = None
        relu_155 = torch.ops.aten.relu.default(add_403);  add_403 = None
        split_154 = torch.ops.aten.split.Tensor(relu_152, 256, 1);  relu_152 = None
        getitem_623 = split_154[3];  split_154 = None
        cat_30 = torch.ops.aten.cat.default([relu_153, relu_154, relu_155, getitem_623], 1);  relu_153 = relu_154 = relu_155 = getitem_623 = None
        convolution_164 = torch.ops.aten.convolution.default(cat_30, arg396_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_30 = arg396_1 = None
        add_404 = torch.ops.aten.add.Tensor(arg398_1, 1e-05);  arg398_1 = None
        sqrt_164 = torch.ops.aten.sqrt.default(add_404);  add_404 = None
        reciprocal_164 = torch.ops.aten.reciprocal.default(sqrt_164);  sqrt_164 = None
        mul_492 = torch.ops.aten.mul.Tensor(reciprocal_164, 1);  reciprocal_164 = None
        unsqueeze_1312 = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
        unsqueeze_1313 = torch.ops.aten.unsqueeze.default(unsqueeze_1312, -1);  unsqueeze_1312 = None
        unsqueeze_1314 = torch.ops.aten.unsqueeze.default(mul_492, -1);  mul_492 = None
        unsqueeze_1315 = torch.ops.aten.unsqueeze.default(unsqueeze_1314, -1);  unsqueeze_1314 = None
        sub_164 = torch.ops.aten.sub.Tensor(convolution_164, unsqueeze_1313);  convolution_164 = unsqueeze_1313 = None
        mul_493 = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_1315);  sub_164 = unsqueeze_1315 = None
        unsqueeze_1316 = torch.ops.aten.unsqueeze.default(arg399_1, -1);  arg399_1 = None
        unsqueeze_1317 = torch.ops.aten.unsqueeze.default(unsqueeze_1316, -1);  unsqueeze_1316 = None
        mul_494 = torch.ops.aten.mul.Tensor(mul_493, unsqueeze_1317);  mul_493 = unsqueeze_1317 = None
        unsqueeze_1318 = torch.ops.aten.unsqueeze.default(arg400_1, -1);  arg400_1 = None
        unsqueeze_1319 = torch.ops.aten.unsqueeze.default(unsqueeze_1318, -1);  unsqueeze_1318 = None
        add_405 = torch.ops.aten.add.Tensor(mul_494, unsqueeze_1319);  mul_494 = unsqueeze_1319 = None
        add_406 = torch.ops.aten.add.Tensor(add_405, relu_151);  add_405 = relu_151 = None
        relu_156 = torch.ops.aten.relu.default(add_406);  add_406 = None
        convolution_165 = torch.ops.aten.convolution.default(relu_156, arg401_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg401_1 = None
        add_407 = torch.ops.aten.add.Tensor(arg403_1, 1e-05);  arg403_1 = None
        sqrt_165 = torch.ops.aten.sqrt.default(add_407);  add_407 = None
        reciprocal_165 = torch.ops.aten.reciprocal.default(sqrt_165);  sqrt_165 = None
        mul_495 = torch.ops.aten.mul.Tensor(reciprocal_165, 1);  reciprocal_165 = None
        unsqueeze_1320 = torch.ops.aten.unsqueeze.default(arg402_1, -1);  arg402_1 = None
        unsqueeze_1321 = torch.ops.aten.unsqueeze.default(unsqueeze_1320, -1);  unsqueeze_1320 = None
        unsqueeze_1322 = torch.ops.aten.unsqueeze.default(mul_495, -1);  mul_495 = None
        unsqueeze_1323 = torch.ops.aten.unsqueeze.default(unsqueeze_1322, -1);  unsqueeze_1322 = None
        sub_165 = torch.ops.aten.sub.Tensor(convolution_165, unsqueeze_1321);  convolution_165 = unsqueeze_1321 = None
        mul_496 = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_1323);  sub_165 = unsqueeze_1323 = None
        unsqueeze_1324 = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
        unsqueeze_1325 = torch.ops.aten.unsqueeze.default(unsqueeze_1324, -1);  unsqueeze_1324 = None
        mul_497 = torch.ops.aten.mul.Tensor(mul_496, unsqueeze_1325);  mul_496 = unsqueeze_1325 = None
        unsqueeze_1326 = torch.ops.aten.unsqueeze.default(arg405_1, -1);  arg405_1 = None
        unsqueeze_1327 = torch.ops.aten.unsqueeze.default(unsqueeze_1326, -1);  unsqueeze_1326 = None
        add_408 = torch.ops.aten.add.Tensor(mul_497, unsqueeze_1327);  mul_497 = unsqueeze_1327 = None
        relu_157 = torch.ops.aten.relu.default(add_408);  add_408 = None
        split_156 = torch.ops.aten.split.Tensor(relu_157, 256, 1)
        getitem_628 = split_156[0];  split_156 = None
        convolution_166 = torch.ops.aten.convolution.default(getitem_628, arg406_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  getitem_628 = arg406_1 = None
        add_409 = torch.ops.aten.add.Tensor(arg408_1, 1e-05);  arg408_1 = None
        sqrt_166 = torch.ops.aten.sqrt.default(add_409);  add_409 = None
        reciprocal_166 = torch.ops.aten.reciprocal.default(sqrt_166);  sqrt_166 = None
        mul_498 = torch.ops.aten.mul.Tensor(reciprocal_166, 1);  reciprocal_166 = None
        unsqueeze_1328 = torch.ops.aten.unsqueeze.default(arg407_1, -1);  arg407_1 = None
        unsqueeze_1329 = torch.ops.aten.unsqueeze.default(unsqueeze_1328, -1);  unsqueeze_1328 = None
        unsqueeze_1330 = torch.ops.aten.unsqueeze.default(mul_498, -1);  mul_498 = None
        unsqueeze_1331 = torch.ops.aten.unsqueeze.default(unsqueeze_1330, -1);  unsqueeze_1330 = None
        sub_166 = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_1329);  convolution_166 = unsqueeze_1329 = None
        mul_499 = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_1331);  sub_166 = unsqueeze_1331 = None
        unsqueeze_1332 = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
        unsqueeze_1333 = torch.ops.aten.unsqueeze.default(unsqueeze_1332, -1);  unsqueeze_1332 = None
        mul_500 = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_1333);  mul_499 = unsqueeze_1333 = None
        unsqueeze_1334 = torch.ops.aten.unsqueeze.default(arg410_1, -1);  arg410_1 = None
        unsqueeze_1335 = torch.ops.aten.unsqueeze.default(unsqueeze_1334, -1);  unsqueeze_1334 = None
        add_410 = torch.ops.aten.add.Tensor(mul_500, unsqueeze_1335);  mul_500 = unsqueeze_1335 = None
        relu_158 = torch.ops.aten.relu.default(add_410);  add_410 = None
        split_157 = torch.ops.aten.split.Tensor(relu_157, 256, 1)
        getitem_633 = split_157[1];  split_157 = None
        add_411 = torch.ops.aten.add.Tensor(relu_158, getitem_633);  getitem_633 = None
        convolution_167 = torch.ops.aten.convolution.default(add_411, arg411_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_411 = arg411_1 = None
        add_412 = torch.ops.aten.add.Tensor(arg413_1, 1e-05);  arg413_1 = None
        sqrt_167 = torch.ops.aten.sqrt.default(add_412);  add_412 = None
        reciprocal_167 = torch.ops.aten.reciprocal.default(sqrt_167);  sqrt_167 = None
        mul_501 = torch.ops.aten.mul.Tensor(reciprocal_167, 1);  reciprocal_167 = None
        unsqueeze_1336 = torch.ops.aten.unsqueeze.default(arg412_1, -1);  arg412_1 = None
        unsqueeze_1337 = torch.ops.aten.unsqueeze.default(unsqueeze_1336, -1);  unsqueeze_1336 = None
        unsqueeze_1338 = torch.ops.aten.unsqueeze.default(mul_501, -1);  mul_501 = None
        unsqueeze_1339 = torch.ops.aten.unsqueeze.default(unsqueeze_1338, -1);  unsqueeze_1338 = None
        sub_167 = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_1337);  convolution_167 = unsqueeze_1337 = None
        mul_502 = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_1339);  sub_167 = unsqueeze_1339 = None
        unsqueeze_1340 = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_1341 = torch.ops.aten.unsqueeze.default(unsqueeze_1340, -1);  unsqueeze_1340 = None
        mul_503 = torch.ops.aten.mul.Tensor(mul_502, unsqueeze_1341);  mul_502 = unsqueeze_1341 = None
        unsqueeze_1342 = torch.ops.aten.unsqueeze.default(arg415_1, -1);  arg415_1 = None
        unsqueeze_1343 = torch.ops.aten.unsqueeze.default(unsqueeze_1342, -1);  unsqueeze_1342 = None
        add_413 = torch.ops.aten.add.Tensor(mul_503, unsqueeze_1343);  mul_503 = unsqueeze_1343 = None
        relu_159 = torch.ops.aten.relu.default(add_413);  add_413 = None
        split_158 = torch.ops.aten.split.Tensor(relu_157, 256, 1)
        getitem_638 = split_158[2];  split_158 = None
        add_414 = torch.ops.aten.add.Tensor(relu_159, getitem_638);  getitem_638 = None
        convolution_168 = torch.ops.aten.convolution.default(add_414, arg416_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  add_414 = arg416_1 = None
        add_415 = torch.ops.aten.add.Tensor(arg418_1, 1e-05);  arg418_1 = None
        sqrt_168 = torch.ops.aten.sqrt.default(add_415);  add_415 = None
        reciprocal_168 = torch.ops.aten.reciprocal.default(sqrt_168);  sqrt_168 = None
        mul_504 = torch.ops.aten.mul.Tensor(reciprocal_168, 1);  reciprocal_168 = None
        unsqueeze_1344 = torch.ops.aten.unsqueeze.default(arg417_1, -1);  arg417_1 = None
        unsqueeze_1345 = torch.ops.aten.unsqueeze.default(unsqueeze_1344, -1);  unsqueeze_1344 = None
        unsqueeze_1346 = torch.ops.aten.unsqueeze.default(mul_504, -1);  mul_504 = None
        unsqueeze_1347 = torch.ops.aten.unsqueeze.default(unsqueeze_1346, -1);  unsqueeze_1346 = None
        sub_168 = torch.ops.aten.sub.Tensor(convolution_168, unsqueeze_1345);  convolution_168 = unsqueeze_1345 = None
        mul_505 = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_1347);  sub_168 = unsqueeze_1347 = None
        unsqueeze_1348 = torch.ops.aten.unsqueeze.default(arg419_1, -1);  arg419_1 = None
        unsqueeze_1349 = torch.ops.aten.unsqueeze.default(unsqueeze_1348, -1);  unsqueeze_1348 = None
        mul_506 = torch.ops.aten.mul.Tensor(mul_505, unsqueeze_1349);  mul_505 = unsqueeze_1349 = None
        unsqueeze_1350 = torch.ops.aten.unsqueeze.default(arg420_1, -1);  arg420_1 = None
        unsqueeze_1351 = torch.ops.aten.unsqueeze.default(unsqueeze_1350, -1);  unsqueeze_1350 = None
        add_416 = torch.ops.aten.add.Tensor(mul_506, unsqueeze_1351);  mul_506 = unsqueeze_1351 = None
        relu_160 = torch.ops.aten.relu.default(add_416);  add_416 = None
        split_159 = torch.ops.aten.split.Tensor(relu_157, 256, 1);  relu_157 = None
        getitem_643 = split_159[3];  split_159 = None
        cat_31 = torch.ops.aten.cat.default([relu_158, relu_159, relu_160, getitem_643], 1);  relu_158 = relu_159 = relu_160 = getitem_643 = None
        convolution_169 = torch.ops.aten.convolution.default(cat_31, arg421_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_31 = arg421_1 = None
        add_417 = torch.ops.aten.add.Tensor(arg423_1, 1e-05);  arg423_1 = None
        sqrt_169 = torch.ops.aten.sqrt.default(add_417);  add_417 = None
        reciprocal_169 = torch.ops.aten.reciprocal.default(sqrt_169);  sqrt_169 = None
        mul_507 = torch.ops.aten.mul.Tensor(reciprocal_169, 1);  reciprocal_169 = None
        unsqueeze_1352 = torch.ops.aten.unsqueeze.default(arg422_1, -1);  arg422_1 = None
        unsqueeze_1353 = torch.ops.aten.unsqueeze.default(unsqueeze_1352, -1);  unsqueeze_1352 = None
        unsqueeze_1354 = torch.ops.aten.unsqueeze.default(mul_507, -1);  mul_507 = None
        unsqueeze_1355 = torch.ops.aten.unsqueeze.default(unsqueeze_1354, -1);  unsqueeze_1354 = None
        sub_169 = torch.ops.aten.sub.Tensor(convolution_169, unsqueeze_1353);  convolution_169 = unsqueeze_1353 = None
        mul_508 = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_1355);  sub_169 = unsqueeze_1355 = None
        unsqueeze_1356 = torch.ops.aten.unsqueeze.default(arg424_1, -1);  arg424_1 = None
        unsqueeze_1357 = torch.ops.aten.unsqueeze.default(unsqueeze_1356, -1);  unsqueeze_1356 = None
        mul_509 = torch.ops.aten.mul.Tensor(mul_508, unsqueeze_1357);  mul_508 = unsqueeze_1357 = None
        unsqueeze_1358 = torch.ops.aten.unsqueeze.default(arg425_1, -1);  arg425_1 = None
        unsqueeze_1359 = torch.ops.aten.unsqueeze.default(unsqueeze_1358, -1);  unsqueeze_1358 = None
        add_418 = torch.ops.aten.add.Tensor(mul_509, unsqueeze_1359);  mul_509 = unsqueeze_1359 = None
        add_419 = torch.ops.aten.add.Tensor(add_418, relu_156);  add_418 = relu_156 = None
        relu_161 = torch.ops.aten.relu.default(add_419);  add_419 = None
        mean_1 = torch.ops.aten.mean.dim(relu_161, [-1, -2], True);  relu_161 = None
        view_1 = torch.ops.aten.view.default(mean_1, [8, 2048]);  mean_1 = None
        permute_1 = torch.ops.aten.permute.default(arg426_1, [1, 0]);  arg426_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg427_1, view_1, permute_1);  arg427_1 = view_1 = permute_1 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 37632, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64, 3, 7, 7), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 224, 224), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf3, (64,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf4, (64,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf5, (64,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf6, (128, 64, 1, 1), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf7, (128,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf9, (128,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf10, (128,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf11, (32, 4, 3, 3), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf12, (32,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf13, (32,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf14, (32,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf15, (32,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf16, (32, 4, 3, 3), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf17, (32,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf18, (32,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf19, (32,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf20, (32,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf21, (32, 4, 3, 3), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf22, (32,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf23, (32,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf24, (32,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf25, (32,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf26, (256, 128, 1, 1), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf27, (256,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf28, (256,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf29, (256,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf30, (256,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf31, (256, 64, 1, 1), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf32, (256,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf33, (256,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf35, (256,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf36, (128, 256, 1, 1), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf37, (128,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf38, (128,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf39, (128,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf40, (128,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf41, (32, 4, 3, 3), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf42, (32,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf43, (32,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf44, (32,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf45, (32,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf46, (32, 4, 3, 3), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf47, (32,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf48, (32,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf49, (32,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf50, (32,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf51, (32, 4, 3, 3), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf52, (32,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf53, (32,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf54, (32,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf55, (32,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf56, (256, 128, 1, 1), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf57, (256,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf58, (256,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf59, (256,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf60, (256,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128, 256, 1, 1), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf63, (128,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf64, (128,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf65, (128,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf66, (32, 4, 3, 3), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf67, (32,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf68, (32,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf69, (32,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf70, (32,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf71, (32, 4, 3, 3), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf72, (32,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf73, (32,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf74, (32,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf75, (32,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf76, (32, 4, 3, 3), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf77, (32,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf78, (32,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf79, (32,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf80, (32,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (256, 128, 1, 1), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf82, (256,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf83, (256,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf84, (256,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf85, (256,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf86, (256, 256, 1, 1), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf87, (256,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf88, (256,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf89, (256,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf90, (256,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf91, (64, 8, 3, 3), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf92, (64,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf93, (64,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf94, (64,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf95, (64,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf96, (64, 8, 3, 3), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf97, (64,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf98, (64,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf99, (64,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf100, (64,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf101, (64, 8, 3, 3), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf102, (64,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf103, (64,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf104, (64,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf105, (64,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf106, (512, 256, 1, 1), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf107, (512,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf108, (512,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf109, (512,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf110, (512,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf111, (512, 256, 1, 1), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf112, (512,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf113, (512,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf114, (512,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf115, (512,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf116, (256, 512, 1, 1), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf117, (256,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf118, (256,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf119, (256,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf120, (256,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf121, (64, 8, 3, 3), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf122, (64,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf123, (64,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf124, (64,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf125, (64,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf126, (64, 8, 3, 3), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf127, (64,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf128, (64,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf129, (64,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf130, (64,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf131, (64, 8, 3, 3), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf132, (64,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf133, (64,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf134, (64,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf135, (64,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf136, (512, 256, 1, 1), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf137, (512,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf138, (512,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf139, (512,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf140, (512,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf141, (256, 512, 1, 1), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf142, (256,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf143, (256,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf144, (256,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf145, (256,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf146, (64, 8, 3, 3), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf147, (64,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf148, (64,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf149, (64,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf150, (64,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf151, (64, 8, 3, 3), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf152, (64,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf153, (64,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf154, (64,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf155, (64,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf156, (64, 8, 3, 3), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf157, (64,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf158, (64,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf159, (64,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf160, (64,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf161, (512, 256, 1, 1), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf162, (512,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf163, (512,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf164, (512,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf165, (512,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf166, (256, 512, 1, 1), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf167, (256,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf168, (256,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf169, (256,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf170, (256,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf171, (64, 8, 3, 3), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf172, (64,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf173, (64,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf174, (64,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf175, (64,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf176, (64, 8, 3, 3), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf177, (64,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf178, (64,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf179, (64,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf180, (64,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf181, (64, 8, 3, 3), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf182, (64,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf183, (64,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf184, (64,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf185, (64,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf186, (512, 256, 1, 1), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf187, (512,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf188, (512,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf189, (512,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf190, (512,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf191, (512, 512, 1, 1), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf192, (512,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf193, (512,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf194, (512,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf195, (512,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf196, (128, 16, 3, 3), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf197, (128,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf198, (128,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf199, (128,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf200, (128,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf201, (128, 16, 3, 3), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf202, (128,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf203, (128,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf204, (128,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf205, (128,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf206, (128, 16, 3, 3), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf207, (128,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf208, (128,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf209, (128,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf210, (128,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf211, (1024, 512, 1, 1), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf212, (1024,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf213, (1024,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf214, (1024,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf215, (1024,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf216, (1024, 512, 1, 1), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf217, (1024,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf218, (1024,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf219, (1024,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf220, (1024,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf221, (512, 1024, 1, 1), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf222, (512,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf223, (512,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf224, (512,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf225, (512,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf226, (128, 16, 3, 3), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf227, (128,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf228, (128,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf229, (128,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf230, (128,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf231, (128, 16, 3, 3), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf232, (128,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf233, (128,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf234, (128,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf235, (128,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf236, (128, 16, 3, 3), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf237, (128,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf238, (128,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf239, (128,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf240, (128,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf241, (1024, 512, 1, 1), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf242, (1024,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf243, (1024,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf244, (1024,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf245, (1024,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf246, (512, 1024, 1, 1), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf247, (512,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf248, (512,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf249, (512,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf250, (512,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf251, (128, 16, 3, 3), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf252, (128,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf253, (128,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf254, (128,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf255, (128,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf256, (128, 16, 3, 3), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf257, (128,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf258, (128,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf259, (128,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf260, (128,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf261, (128, 16, 3, 3), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf262, (128,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf263, (128,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf264, (128,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf265, (128,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf266, (1024, 512, 1, 1), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf267, (1024,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf268, (1024,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf269, (1024,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf270, (1024,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf271, (512, 1024, 1, 1), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf272, (512,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf273, (512,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf274, (512,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf275, (512,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf276, (128, 16, 3, 3), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf277, (128,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf278, (128,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf279, (128,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf280, (128,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf281, (128, 16, 3, 3), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf282, (128,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf283, (128,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf284, (128,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf285, (128,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf286, (128, 16, 3, 3), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf287, (128,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf288, (128,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf289, (128,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf290, (128,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf291, (1024, 512, 1, 1), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf292, (1024,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf293, (1024,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf294, (1024,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf295, (1024,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf296, (512, 1024, 1, 1), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf297, (512,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf298, (512,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf299, (512,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf300, (512,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf301, (128, 16, 3, 3), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf302, (128,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf303, (128,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf304, (128,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf305, (128,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf306, (128, 16, 3, 3), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf307, (128,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf308, (128,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf309, (128,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf310, (128,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf311, (128, 16, 3, 3), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf312, (128,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf313, (128,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf314, (128,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf315, (128,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf316, (1024, 512, 1, 1), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf317, (1024,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf318, (1024,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf319, (1024,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf320, (1024,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf321, (512, 1024, 1, 1), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf322, (512,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf323, (512,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf324, (512,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf325, (512,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf326, (128, 16, 3, 3), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf327, (128,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf328, (128,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf329, (128,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf330, (128,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf331, (128, 16, 3, 3), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf332, (128,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf333, (128,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf334, (128,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf335, (128,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf336, (128, 16, 3, 3), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf337, (128,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf338, (128,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf339, (128,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf340, (128,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf341, (1024, 512, 1, 1), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf342, (1024,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf343, (1024,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf344, (1024,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf345, (1024,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf346, (1024, 1024, 1, 1), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf347, (1024,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf348, (1024,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf349, (1024,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf350, (1024,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf351, (256, 32, 3, 3), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf352, (256,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf353, (256,), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf354, (256,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf355, (256,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf356, (256, 32, 3, 3), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf357, (256,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf358, (256,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf359, (256,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf360, (256,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf361, (256, 32, 3, 3), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf362, (256,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf363, (256,), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf364, (256,), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf365, (256,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf366, (2048, 1024, 1, 1), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf367, (2048,), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf368, (2048,), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf369, (2048,), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf370, (2048,), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf371, (2048, 1024, 1, 1), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf372, (2048,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf373, (2048,), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf374, (2048,), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf375, (2048,), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf376, (1024, 2048, 1, 1), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf377, (1024,), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf378, (1024,), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf379, (1024,), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf380, (1024,), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf381, (256, 32, 3, 3), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf382, (256,), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf383, (256,), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf384, (256,), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf385, (256,), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf386, (256, 32, 3, 3), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf387, (256,), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf388, (256,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf389, (256,), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf390, (256,), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf391, (256, 32, 3, 3), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf392, (256,), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf393, (256,), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf394, (256,), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf395, (256,), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf396, (2048, 1024, 1, 1), is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf397, (2048,), is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf398, (2048,), is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf399, (2048,), is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf400, (2048,), is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf401, (1024, 2048, 1, 1), is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf402, (1024,), is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf403, (1024,), is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf404, (1024,), is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf405, (1024,), is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf406, (256, 32, 3, 3), is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf407, (256,), is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf408, (256,), is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf409, (256,), is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf410, (256,), is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf411, (256, 32, 3, 3), is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf412, (256,), is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf413, (256,), is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf414, (256,), is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf415, (256,), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf416, (256, 32, 3, 3), is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf417, (256,), is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf418, (256,), is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf419, (256,), is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf420, (256,), is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf421, (2048, 1024, 1, 1), is_leaf=True)  # arg421_1
    buf422 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf422, (2048,), is_leaf=True)  # arg422_1
    buf423 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf423, (2048,), is_leaf=True)  # arg423_1
    buf424 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf424, (2048,), is_leaf=True)  # arg424_1
    buf425 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf425, (2048,), is_leaf=True)  # arg425_1
    buf426 = reader.storage(None, 8192000, device=device(type='cuda', index=0))
    reader.tensor(buf426, (1000, 2048), is_leaf=True)  # arg426_1
    buf427 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf427, (1000,), is_leaf=True)  # arg427_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)