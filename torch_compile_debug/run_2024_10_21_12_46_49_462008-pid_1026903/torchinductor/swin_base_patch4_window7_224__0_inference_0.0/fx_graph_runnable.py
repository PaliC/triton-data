
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1):
        convolution_1 = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        permute_248 = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None
        clone_265 = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        var_mean_53 = torch.ops.aten.var_mean.correction(clone_265, [3], correction = 0, keepdim = True)
        getitem_178 = var_mean_53[0]
        getitem_179 = var_mean_53[1];  var_mean_53 = None
        add_257 = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
        sub_77 = torch.ops.aten.sub.Tensor(clone_265, getitem_179);  clone_265 = getitem_179 = None
        mul_202 = torch.ops.aten.mul.Tensor(sub_77, rsqrt_53);  sub_77 = rsqrt_53 = None
        mul_203 = torch.ops.aten.mul.Tensor(mul_202, arg3_1);  mul_202 = arg3_1 = None
        add_258 = torch.ops.aten.add.Tensor(mul_203, arg4_1);  mul_203 = arg4_1 = None
        var_mean_54 = torch.ops.aten.var_mean.correction(add_258, [3], correction = 0, keepdim = True)
        getitem_180 = var_mean_54[0]
        getitem_181 = var_mean_54[1];  var_mean_54 = None
        add_259 = torch.ops.aten.add.Tensor(getitem_180, 1e-05);  getitem_180 = None
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_259);  add_259 = None
        sub_78 = torch.ops.aten.sub.Tensor(add_258, getitem_181);  getitem_181 = None
        mul_204 = torch.ops.aten.mul.Tensor(sub_78, rsqrt_54);  sub_78 = rsqrt_54 = None
        mul_205 = torch.ops.aten.mul.Tensor(mul_204, arg5_1);  mul_204 = arg5_1 = None
        add_260 = torch.ops.aten.add.Tensor(mul_205, arg6_1);  mul_205 = arg6_1 = None
        view_658 = torch.ops.aten.view.default(add_260, [8, 8, 7, 8, 7, 128]);  add_260 = None
        permute_249 = torch.ops.aten.permute.default(view_658, [0, 1, 3, 2, 4, 5]);  view_658 = None
        clone_266 = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_659 = torch.ops.aten.view.default(clone_266, [-1, 7, 7, 128]);  clone_266 = None
        view_660 = torch.ops.aten.view.default(view_659, [-1, 49, 128]);  view_659 = None
        view_661 = torch.ops.aten.view.default(view_660, [25088, 128]);  view_660 = None
        permute_250 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg8_1, view_661, permute_250);  arg8_1 = view_661 = permute_250 = None
        view_662 = torch.ops.aten.view.default(addmm_97, [512, 49, 384]);  addmm_97 = None
        view_663 = torch.ops.aten.view.default(view_662, [512, 49, 3, 4, -1]);  view_662 = None
        permute_251 = torch.ops.aten.permute.default(view_663, [2, 0, 3, 1, 4]);  view_663 = None
        unbind_24 = torch.ops.aten.unbind.int(permute_251);  permute_251 = None
        getitem_182 = unbind_24[0]
        getitem_183 = unbind_24[1]
        getitem_184 = unbind_24[2];  unbind_24 = None
        mul_206 = torch.ops.aten.mul.Tensor(getitem_182, 0.1767766952966369);  getitem_182 = None
        permute_252 = torch.ops.aten.permute.default(getitem_183, [0, 1, 3, 2]);  getitem_183 = None
        expand_96 = torch.ops.aten.expand.default(mul_206, [512, 4, 49, 32]);  mul_206 = None
        clone_267 = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_664 = torch.ops.aten.view.default(clone_267, [2048, 49, 32]);  clone_267 = None
        expand_97 = torch.ops.aten.expand.default(permute_252, [512, 4, 32, 49]);  permute_252 = None
        clone_268 = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_665 = torch.ops.aten.view.default(clone_268, [2048, 32, 49]);  clone_268 = None
        bmm_48 = torch.ops.aten.bmm.default(view_664, view_665);  view_664 = view_665 = None
        view_666 = torch.ops.aten.view.default(bmm_48, [512, 4, 49, 49]);  bmm_48 = None
        view_667 = torch.ops.aten.view.default(arg10_1, [-1]);  arg10_1 = None
        index_68 = torch.ops.aten.index.Tensor(arg9_1, [view_667]);  arg9_1 = view_667 = None
        view_668 = torch.ops.aten.view.default(index_68, [49, 49, -1]);  index_68 = None
        permute_253 = torch.ops.aten.permute.default(view_668, [2, 0, 1]);  view_668 = None
        clone_269 = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(clone_269, 0);  clone_269 = None
        add_261 = torch.ops.aten.add.Tensor(view_666, unsqueeze_46);  view_666 = unsqueeze_46 = None
        amax_24 = torch.ops.aten.amax.default(add_261, [-1], True)
        sub_79 = torch.ops.aten.sub.Tensor(add_261, amax_24);  add_261 = amax_24 = None
        exp_24 = torch.ops.aten.exp.default(sub_79);  sub_79 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_24 = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        expand_98 = torch.ops.aten.expand.default(div_24, [512, 4, 49, 49]);  div_24 = None
        view_669 = torch.ops.aten.view.default(expand_98, [2048, 49, 49]);  expand_98 = None
        expand_99 = torch.ops.aten.expand.default(getitem_184, [512, 4, 49, 32]);  getitem_184 = None
        clone_271 = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
        view_670 = torch.ops.aten.view.default(clone_271, [2048, 49, 32]);  clone_271 = None
        bmm_49 = torch.ops.aten.bmm.default(view_669, view_670);  view_669 = view_670 = None
        view_671 = torch.ops.aten.view.default(bmm_49, [512, 4, 49, 32]);  bmm_49 = None
        permute_254 = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None
        clone_272 = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_672 = torch.ops.aten.view.default(clone_272, [512, 49, 128]);  clone_272 = None
        view_673 = torch.ops.aten.view.default(view_672, [25088, 128]);  view_672 = None
        permute_255 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg12_1, view_673, permute_255);  arg12_1 = view_673 = permute_255 = None
        view_674 = torch.ops.aten.view.default(addmm_98, [512, 49, 128]);  addmm_98 = None
        view_675 = torch.ops.aten.view.default(view_674, [-1, 7, 7, 128]);  view_674 = None
        view_676 = torch.ops.aten.view.default(view_675, [-1, 8, 8, 7, 7, 128]);  view_675 = None
        permute_256 = torch.ops.aten.permute.default(view_676, [0, 1, 3, 2, 4, 5]);  view_676 = None
        clone_274 = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_677 = torch.ops.aten.view.default(clone_274, [-1, 56, 56, 128]);  clone_274 = None
        add_262 = torch.ops.aten.add.Tensor(add_258, view_677);  add_258 = view_677 = None
        view_678 = torch.ops.aten.view.default(add_262, [8, -1, 128]);  add_262 = None
        var_mean_55 = torch.ops.aten.var_mean.correction(view_678, [2], correction = 0, keepdim = True)
        getitem_185 = var_mean_55[0]
        getitem_186 = var_mean_55[1];  var_mean_55 = None
        add_263 = torch.ops.aten.add.Tensor(getitem_185, 1e-05);  getitem_185 = None
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
        sub_80 = torch.ops.aten.sub.Tensor(view_678, getitem_186);  getitem_186 = None
        mul_207 = torch.ops.aten.mul.Tensor(sub_80, rsqrt_55);  sub_80 = rsqrt_55 = None
        mul_208 = torch.ops.aten.mul.Tensor(mul_207, arg13_1);  mul_207 = arg13_1 = None
        add_264 = torch.ops.aten.add.Tensor(mul_208, arg14_1);  mul_208 = arg14_1 = None
        view_679 = torch.ops.aten.view.default(add_264, [25088, 128]);  add_264 = None
        permute_257 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg16_1, view_679, permute_257);  arg16_1 = view_679 = permute_257 = None
        view_680 = torch.ops.aten.view.default(addmm_99, [8, 3136, 512]);  addmm_99 = None
        mul_209 = torch.ops.aten.mul.Tensor(view_680, 0.5)
        mul_210 = torch.ops.aten.mul.Tensor(view_680, 0.7071067811865476);  view_680 = None
        erf_24 = torch.ops.aten.erf.default(mul_210);  mul_210 = None
        add_265 = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_211 = torch.ops.aten.mul.Tensor(mul_209, add_265);  mul_209 = add_265 = None
        view_681 = torch.ops.aten.view.default(mul_211, [25088, 512]);  mul_211 = None
        permute_258 = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg18_1, view_681, permute_258);  arg18_1 = view_681 = permute_258 = None
        view_682 = torch.ops.aten.view.default(addmm_100, [8, 3136, 128]);  addmm_100 = None
        add_266 = torch.ops.aten.add.Tensor(view_678, view_682);  view_678 = view_682 = None
        view_683 = torch.ops.aten.view.default(add_266, [8, 56, 56, 128]);  add_266 = None
        var_mean_56 = torch.ops.aten.var_mean.correction(view_683, [3], correction = 0, keepdim = True)
        getitem_187 = var_mean_56[0]
        getitem_188 = var_mean_56[1];  var_mean_56 = None
        add_267 = torch.ops.aten.add.Tensor(getitem_187, 1e-05);  getitem_187 = None
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
        sub_81 = torch.ops.aten.sub.Tensor(view_683, getitem_188);  getitem_188 = None
        mul_212 = torch.ops.aten.mul.Tensor(sub_81, rsqrt_56);  sub_81 = rsqrt_56 = None
        mul_213 = torch.ops.aten.mul.Tensor(mul_212, arg19_1);  mul_212 = arg19_1 = None
        add_268 = torch.ops.aten.add.Tensor(mul_213, arg20_1);  mul_213 = arg20_1 = None
        iota_44 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_269 = torch.ops.aten.add.Tensor(iota_44, 3);  iota_44 = None
        fmod_44 = torch.ops.aten.fmod.Scalar(add_269, 56);  add_269 = None
        index_69 = torch.ops.aten.index.Tensor(add_268, [None, fmod_44]);  add_268 = fmod_44 = None
        iota_45 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_270 = torch.ops.aten.add.Tensor(iota_45, 3);  iota_45 = None
        fmod_45 = torch.ops.aten.fmod.Scalar(add_270, 56);  add_270 = None
        index_70 = torch.ops.aten.index.Tensor(index_69, [None, None, fmod_45]);  index_69 = fmod_45 = None
        view_684 = torch.ops.aten.view.default(index_70, [8, 8, 7, 8, 7, 128]);  index_70 = None
        permute_259 = torch.ops.aten.permute.default(view_684, [0, 1, 3, 2, 4, 5]);  view_684 = None
        clone_277 = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_685 = torch.ops.aten.view.default(clone_277, [-1, 7, 7, 128]);  clone_277 = None
        view_686 = torch.ops.aten.view.default(view_685, [-1, 49, 128]);  view_685 = None
        view_687 = torch.ops.aten.view.default(view_686, [25088, 128]);  view_686 = None
        permute_260 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg23_1, view_687, permute_260);  arg23_1 = view_687 = permute_260 = None
        view_688 = torch.ops.aten.view.default(addmm_101, [512, 49, 384]);  addmm_101 = None
        view_689 = torch.ops.aten.view.default(view_688, [512, 49, 3, 4, -1]);  view_688 = None
        permute_261 = torch.ops.aten.permute.default(view_689, [2, 0, 3, 1, 4]);  view_689 = None
        unbind_25 = torch.ops.aten.unbind.int(permute_261);  permute_261 = None
        getitem_189 = unbind_25[0]
        getitem_190 = unbind_25[1]
        getitem_191 = unbind_25[2];  unbind_25 = None
        mul_214 = torch.ops.aten.mul.Tensor(getitem_189, 0.1767766952966369);  getitem_189 = None
        permute_262 = torch.ops.aten.permute.default(getitem_190, [0, 1, 3, 2]);  getitem_190 = None
        expand_100 = torch.ops.aten.expand.default(mul_214, [512, 4, 49, 32]);  mul_214 = None
        clone_278 = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
        view_690 = torch.ops.aten.view.default(clone_278, [2048, 49, 32]);  clone_278 = None
        expand_101 = torch.ops.aten.expand.default(permute_262, [512, 4, 32, 49]);  permute_262 = None
        clone_279 = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
        view_691 = torch.ops.aten.view.default(clone_279, [2048, 32, 49]);  clone_279 = None
        bmm_50 = torch.ops.aten.bmm.default(view_690, view_691);  view_690 = view_691 = None
        view_692 = torch.ops.aten.view.default(bmm_50, [512, 4, 49, 49]);  bmm_50 = None
        view_693 = torch.ops.aten.view.default(arg25_1, [-1]);  arg25_1 = None
        index_71 = torch.ops.aten.index.Tensor(arg24_1, [view_693]);  arg24_1 = view_693 = None
        view_694 = torch.ops.aten.view.default(index_71, [49, 49, -1]);  index_71 = None
        permute_263 = torch.ops.aten.permute.default(view_694, [2, 0, 1]);  view_694 = None
        clone_280 = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(clone_280, 0);  clone_280 = None
        add_271 = torch.ops.aten.add.Tensor(view_692, unsqueeze_47);  view_692 = unsqueeze_47 = None
        view_695 = torch.ops.aten.view.default(add_271, [-1, 64, 4, 49, 49]);  add_271 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(arg21_1, 1);  arg21_1 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(unsqueeze_48, 0);  unsqueeze_48 = None
        add_272 = torch.ops.aten.add.Tensor(view_695, unsqueeze_49);  view_695 = unsqueeze_49 = None
        view_696 = torch.ops.aten.view.default(add_272, [-1, 4, 49, 49]);  add_272 = None
        amax_25 = torch.ops.aten.amax.default(view_696, [-1], True)
        sub_82 = torch.ops.aten.sub.Tensor(view_696, amax_25);  view_696 = amax_25 = None
        exp_25 = torch.ops.aten.exp.default(sub_82);  sub_82 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_25 = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
        expand_102 = torch.ops.aten.expand.default(div_25, [512, 4, 49, 49]);  div_25 = None
        view_697 = torch.ops.aten.view.default(expand_102, [2048, 49, 49]);  expand_102 = None
        expand_103 = torch.ops.aten.expand.default(getitem_191, [512, 4, 49, 32]);  getitem_191 = None
        clone_282 = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
        view_698 = torch.ops.aten.view.default(clone_282, [2048, 49, 32]);  clone_282 = None
        bmm_51 = torch.ops.aten.bmm.default(view_697, view_698);  view_697 = view_698 = None
        view_699 = torch.ops.aten.view.default(bmm_51, [512, 4, 49, 32]);  bmm_51 = None
        permute_264 = torch.ops.aten.permute.default(view_699, [0, 2, 1, 3]);  view_699 = None
        clone_283 = torch.ops.aten.clone.default(permute_264, memory_format = torch.contiguous_format);  permute_264 = None
        view_700 = torch.ops.aten.view.default(clone_283, [512, 49, 128]);  clone_283 = None
        view_701 = torch.ops.aten.view.default(view_700, [25088, 128]);  view_700 = None
        permute_265 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg27_1, view_701, permute_265);  arg27_1 = view_701 = permute_265 = None
        view_702 = torch.ops.aten.view.default(addmm_102, [512, 49, 128]);  addmm_102 = None
        view_703 = torch.ops.aten.view.default(view_702, [-1, 7, 7, 128]);  view_702 = None
        view_704 = torch.ops.aten.view.default(view_703, [-1, 8, 8, 7, 7, 128]);  view_703 = None
        permute_266 = torch.ops.aten.permute.default(view_704, [0, 1, 3, 2, 4, 5]);  view_704 = None
        clone_285 = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
        view_705 = torch.ops.aten.view.default(clone_285, [-1, 56, 56, 128]);  clone_285 = None
        iota_46 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_273 = torch.ops.aten.add.Tensor(iota_46, 53);  iota_46 = None
        fmod_46 = torch.ops.aten.fmod.Scalar(add_273, 56);  add_273 = None
        index_72 = torch.ops.aten.index.Tensor(view_705, [None, fmod_46]);  view_705 = fmod_46 = None
        iota_47 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_274 = torch.ops.aten.add.Tensor(iota_47, 53);  iota_47 = None
        fmod_47 = torch.ops.aten.fmod.Scalar(add_274, 56);  add_274 = None
        index_73 = torch.ops.aten.index.Tensor(index_72, [None, None, fmod_47]);  index_72 = fmod_47 = None
        add_275 = torch.ops.aten.add.Tensor(view_683, index_73);  view_683 = index_73 = None
        view_706 = torch.ops.aten.view.default(add_275, [8, -1, 128]);  add_275 = None
        var_mean_57 = torch.ops.aten.var_mean.correction(view_706, [2], correction = 0, keepdim = True)
        getitem_192 = var_mean_57[0]
        getitem_193 = var_mean_57[1];  var_mean_57 = None
        add_276 = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_83 = torch.ops.aten.sub.Tensor(view_706, getitem_193);  getitem_193 = None
        mul_215 = torch.ops.aten.mul.Tensor(sub_83, rsqrt_57);  sub_83 = rsqrt_57 = None
        mul_216 = torch.ops.aten.mul.Tensor(mul_215, arg28_1);  mul_215 = arg28_1 = None
        add_277 = torch.ops.aten.add.Tensor(mul_216, arg29_1);  mul_216 = arg29_1 = None
        view_707 = torch.ops.aten.view.default(add_277, [25088, 128]);  add_277 = None
        permute_267 = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg31_1, view_707, permute_267);  arg31_1 = view_707 = permute_267 = None
        view_708 = torch.ops.aten.view.default(addmm_103, [8, 3136, 512]);  addmm_103 = None
        mul_217 = torch.ops.aten.mul.Tensor(view_708, 0.5)
        mul_218 = torch.ops.aten.mul.Tensor(view_708, 0.7071067811865476);  view_708 = None
        erf_25 = torch.ops.aten.erf.default(mul_218);  mul_218 = None
        add_278 = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_219 = torch.ops.aten.mul.Tensor(mul_217, add_278);  mul_217 = add_278 = None
        view_709 = torch.ops.aten.view.default(mul_219, [25088, 512]);  mul_219 = None
        permute_268 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg33_1, view_709, permute_268);  arg33_1 = view_709 = permute_268 = None
        view_710 = torch.ops.aten.view.default(addmm_104, [8, 3136, 128]);  addmm_104 = None
        add_279 = torch.ops.aten.add.Tensor(view_706, view_710);  view_706 = view_710 = None
        view_711 = torch.ops.aten.view.default(add_279, [8, 56, 56, 128]);  add_279 = None
        view_712 = torch.ops.aten.view.default(view_711, [8, 28, 2, 28, 2, 128]);  view_711 = None
        permute_269 = torch.ops.aten.permute.default(view_712, [0, 1, 3, 4, 2, 5]);  view_712 = None
        clone_288 = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_713 = torch.ops.aten.view.default(clone_288, [8, 28, 28, 512]);  clone_288 = None
        var_mean_58 = torch.ops.aten.var_mean.correction(view_713, [3], correction = 0, keepdim = True)
        getitem_194 = var_mean_58[0]
        getitem_195 = var_mean_58[1];  var_mean_58 = None
        add_280 = torch.ops.aten.add.Tensor(getitem_194, 1e-05);  getitem_194 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
        sub_84 = torch.ops.aten.sub.Tensor(view_713, getitem_195);  view_713 = getitem_195 = None
        mul_220 = torch.ops.aten.mul.Tensor(sub_84, rsqrt_58);  sub_84 = rsqrt_58 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, arg34_1);  mul_220 = arg34_1 = None
        add_281 = torch.ops.aten.add.Tensor(mul_221, arg35_1);  mul_221 = arg35_1 = None
        permute_270 = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        view_714 = torch.ops.aten.view.default(add_281, [6272, 512]);  add_281 = None
        mm_3 = torch.ops.aten.mm.default(view_714, permute_270);  view_714 = permute_270 = None
        view_715 = torch.ops.aten.view.default(mm_3, [8, 28, 28, 256]);  mm_3 = None
        var_mean_59 = torch.ops.aten.var_mean.correction(view_715, [3], correction = 0, keepdim = True)
        getitem_196 = var_mean_59[0]
        getitem_197 = var_mean_59[1];  var_mean_59 = None
        add_282 = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        sub_85 = torch.ops.aten.sub.Tensor(view_715, getitem_197);  getitem_197 = None
        mul_222 = torch.ops.aten.mul.Tensor(sub_85, rsqrt_59);  sub_85 = rsqrt_59 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_222, arg37_1);  mul_222 = arg37_1 = None
        add_283 = torch.ops.aten.add.Tensor(mul_223, arg38_1);  mul_223 = arg38_1 = None
        view_716 = torch.ops.aten.view.default(add_283, [8, 4, 7, 4, 7, 256]);  add_283 = None
        permute_271 = torch.ops.aten.permute.default(view_716, [0, 1, 3, 2, 4, 5]);  view_716 = None
        clone_289 = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None
        view_717 = torch.ops.aten.view.default(clone_289, [-1, 7, 7, 256]);  clone_289 = None
        view_718 = torch.ops.aten.view.default(view_717, [-1, 49, 256]);  view_717 = None
        view_719 = torch.ops.aten.view.default(view_718, [6272, 256]);  view_718 = None
        permute_272 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg40_1, view_719, permute_272);  arg40_1 = view_719 = permute_272 = None
        view_720 = torch.ops.aten.view.default(addmm_105, [128, 49, 768]);  addmm_105 = None
        view_721 = torch.ops.aten.view.default(view_720, [128, 49, 3, 8, -1]);  view_720 = None
        permute_273 = torch.ops.aten.permute.default(view_721, [2, 0, 3, 1, 4]);  view_721 = None
        unbind_26 = torch.ops.aten.unbind.int(permute_273);  permute_273 = None
        getitem_198 = unbind_26[0]
        getitem_199 = unbind_26[1]
        getitem_200 = unbind_26[2];  unbind_26 = None
        mul_224 = torch.ops.aten.mul.Tensor(getitem_198, 0.1767766952966369);  getitem_198 = None
        permute_274 = torch.ops.aten.permute.default(getitem_199, [0, 1, 3, 2]);  getitem_199 = None
        expand_104 = torch.ops.aten.expand.default(mul_224, [128, 8, 49, 32]);  mul_224 = None
        clone_290 = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
        view_722 = torch.ops.aten.view.default(clone_290, [1024, 49, 32]);  clone_290 = None
        expand_105 = torch.ops.aten.expand.default(permute_274, [128, 8, 32, 49]);  permute_274 = None
        clone_291 = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_723 = torch.ops.aten.view.default(clone_291, [1024, 32, 49]);  clone_291 = None
        bmm_52 = torch.ops.aten.bmm.default(view_722, view_723);  view_722 = view_723 = None
        view_724 = torch.ops.aten.view.default(bmm_52, [128, 8, 49, 49]);  bmm_52 = None
        view_725 = torch.ops.aten.view.default(arg42_1, [-1]);  arg42_1 = None
        index_74 = torch.ops.aten.index.Tensor(arg41_1, [view_725]);  arg41_1 = view_725 = None
        view_726 = torch.ops.aten.view.default(index_74, [49, 49, -1]);  index_74 = None
        permute_275 = torch.ops.aten.permute.default(view_726, [2, 0, 1]);  view_726 = None
        clone_292 = torch.ops.aten.clone.default(permute_275, memory_format = torch.contiguous_format);  permute_275 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(clone_292, 0);  clone_292 = None
        add_284 = torch.ops.aten.add.Tensor(view_724, unsqueeze_50);  view_724 = unsqueeze_50 = None
        amax_26 = torch.ops.aten.amax.default(add_284, [-1], True)
        sub_86 = torch.ops.aten.sub.Tensor(add_284, amax_26);  add_284 = amax_26 = None
        exp_26 = torch.ops.aten.exp.default(sub_86);  sub_86 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_26 = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
        expand_106 = torch.ops.aten.expand.default(div_26, [128, 8, 49, 49]);  div_26 = None
        view_727 = torch.ops.aten.view.default(expand_106, [1024, 49, 49]);  expand_106 = None
        expand_107 = torch.ops.aten.expand.default(getitem_200, [128, 8, 49, 32]);  getitem_200 = None
        clone_294 = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
        view_728 = torch.ops.aten.view.default(clone_294, [1024, 49, 32]);  clone_294 = None
        bmm_53 = torch.ops.aten.bmm.default(view_727, view_728);  view_727 = view_728 = None
        view_729 = torch.ops.aten.view.default(bmm_53, [128, 8, 49, 32]);  bmm_53 = None
        permute_276 = torch.ops.aten.permute.default(view_729, [0, 2, 1, 3]);  view_729 = None
        clone_295 = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
        view_730 = torch.ops.aten.view.default(clone_295, [128, 49, 256]);  clone_295 = None
        view_731 = torch.ops.aten.view.default(view_730, [6272, 256]);  view_730 = None
        permute_277 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg44_1, view_731, permute_277);  arg44_1 = view_731 = permute_277 = None
        view_732 = torch.ops.aten.view.default(addmm_106, [128, 49, 256]);  addmm_106 = None
        view_733 = torch.ops.aten.view.default(view_732, [-1, 7, 7, 256]);  view_732 = None
        view_734 = torch.ops.aten.view.default(view_733, [-1, 4, 4, 7, 7, 256]);  view_733 = None
        permute_278 = torch.ops.aten.permute.default(view_734, [0, 1, 3, 2, 4, 5]);  view_734 = None
        clone_297 = torch.ops.aten.clone.default(permute_278, memory_format = torch.contiguous_format);  permute_278 = None
        view_735 = torch.ops.aten.view.default(clone_297, [-1, 28, 28, 256]);  clone_297 = None
        add_285 = torch.ops.aten.add.Tensor(view_715, view_735);  view_715 = view_735 = None
        view_736 = torch.ops.aten.view.default(add_285, [8, -1, 256]);  add_285 = None
        var_mean_60 = torch.ops.aten.var_mean.correction(view_736, [2], correction = 0, keepdim = True)
        getitem_201 = var_mean_60[0]
        getitem_202 = var_mean_60[1];  var_mean_60 = None
        add_286 = torch.ops.aten.add.Tensor(getitem_201, 1e-05);  getitem_201 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
        sub_87 = torch.ops.aten.sub.Tensor(view_736, getitem_202);  getitem_202 = None
        mul_225 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_60);  sub_87 = rsqrt_60 = None
        mul_226 = torch.ops.aten.mul.Tensor(mul_225, arg45_1);  mul_225 = arg45_1 = None
        add_287 = torch.ops.aten.add.Tensor(mul_226, arg46_1);  mul_226 = arg46_1 = None
        view_737 = torch.ops.aten.view.default(add_287, [6272, 256]);  add_287 = None
        permute_279 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg48_1, view_737, permute_279);  arg48_1 = view_737 = permute_279 = None
        view_738 = torch.ops.aten.view.default(addmm_107, [8, 784, 1024]);  addmm_107 = None
        mul_227 = torch.ops.aten.mul.Tensor(view_738, 0.5)
        mul_228 = torch.ops.aten.mul.Tensor(view_738, 0.7071067811865476);  view_738 = None
        erf_26 = torch.ops.aten.erf.default(mul_228);  mul_228 = None
        add_288 = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_229 = torch.ops.aten.mul.Tensor(mul_227, add_288);  mul_227 = add_288 = None
        view_739 = torch.ops.aten.view.default(mul_229, [6272, 1024]);  mul_229 = None
        permute_280 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg50_1, view_739, permute_280);  arg50_1 = view_739 = permute_280 = None
        view_740 = torch.ops.aten.view.default(addmm_108, [8, 784, 256]);  addmm_108 = None
        add_289 = torch.ops.aten.add.Tensor(view_736, view_740);  view_736 = view_740 = None
        view_741 = torch.ops.aten.view.default(add_289, [8, 28, 28, 256]);  add_289 = None
        var_mean_61 = torch.ops.aten.var_mean.correction(view_741, [3], correction = 0, keepdim = True)
        getitem_203 = var_mean_61[0]
        getitem_204 = var_mean_61[1];  var_mean_61 = None
        add_290 = torch.ops.aten.add.Tensor(getitem_203, 1e-05);  getitem_203 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
        sub_88 = torch.ops.aten.sub.Tensor(view_741, getitem_204);  getitem_204 = None
        mul_230 = torch.ops.aten.mul.Tensor(sub_88, rsqrt_61);  sub_88 = rsqrt_61 = None
        mul_231 = torch.ops.aten.mul.Tensor(mul_230, arg51_1);  mul_230 = arg51_1 = None
        add_291 = torch.ops.aten.add.Tensor(mul_231, arg52_1);  mul_231 = arg52_1 = None
        iota_48 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_292 = torch.ops.aten.add.Tensor(iota_48, 3);  iota_48 = None
        fmod_48 = torch.ops.aten.fmod.Scalar(add_292, 28);  add_292 = None
        index_75 = torch.ops.aten.index.Tensor(add_291, [None, fmod_48]);  add_291 = fmod_48 = None
        iota_49 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_293 = torch.ops.aten.add.Tensor(iota_49, 3);  iota_49 = None
        fmod_49 = torch.ops.aten.fmod.Scalar(add_293, 28);  add_293 = None
        index_76 = torch.ops.aten.index.Tensor(index_75, [None, None, fmod_49]);  index_75 = fmod_49 = None
        view_742 = torch.ops.aten.view.default(index_76, [8, 4, 7, 4, 7, 256]);  index_76 = None
        permute_281 = torch.ops.aten.permute.default(view_742, [0, 1, 3, 2, 4, 5]);  view_742 = None
        clone_300 = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None
        view_743 = torch.ops.aten.view.default(clone_300, [-1, 7, 7, 256]);  clone_300 = None
        view_744 = torch.ops.aten.view.default(view_743, [-1, 49, 256]);  view_743 = None
        view_745 = torch.ops.aten.view.default(view_744, [6272, 256]);  view_744 = None
        permute_282 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg55_1, view_745, permute_282);  arg55_1 = view_745 = permute_282 = None
        view_746 = torch.ops.aten.view.default(addmm_109, [128, 49, 768]);  addmm_109 = None
        view_747 = torch.ops.aten.view.default(view_746, [128, 49, 3, 8, -1]);  view_746 = None
        permute_283 = torch.ops.aten.permute.default(view_747, [2, 0, 3, 1, 4]);  view_747 = None
        unbind_27 = torch.ops.aten.unbind.int(permute_283);  permute_283 = None
        getitem_205 = unbind_27[0]
        getitem_206 = unbind_27[1]
        getitem_207 = unbind_27[2];  unbind_27 = None
        mul_232 = torch.ops.aten.mul.Tensor(getitem_205, 0.1767766952966369);  getitem_205 = None
        permute_284 = torch.ops.aten.permute.default(getitem_206, [0, 1, 3, 2]);  getitem_206 = None
        expand_108 = torch.ops.aten.expand.default(mul_232, [128, 8, 49, 32]);  mul_232 = None
        clone_301 = torch.ops.aten.clone.default(expand_108, memory_format = torch.contiguous_format);  expand_108 = None
        view_748 = torch.ops.aten.view.default(clone_301, [1024, 49, 32]);  clone_301 = None
        expand_109 = torch.ops.aten.expand.default(permute_284, [128, 8, 32, 49]);  permute_284 = None
        clone_302 = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
        view_749 = torch.ops.aten.view.default(clone_302, [1024, 32, 49]);  clone_302 = None
        bmm_54 = torch.ops.aten.bmm.default(view_748, view_749);  view_748 = view_749 = None
        view_750 = torch.ops.aten.view.default(bmm_54, [128, 8, 49, 49]);  bmm_54 = None
        view_751 = torch.ops.aten.view.default(arg57_1, [-1]);  arg57_1 = None
        index_77 = torch.ops.aten.index.Tensor(arg56_1, [view_751]);  arg56_1 = view_751 = None
        view_752 = torch.ops.aten.view.default(index_77, [49, 49, -1]);  index_77 = None
        permute_285 = torch.ops.aten.permute.default(view_752, [2, 0, 1]);  view_752 = None
        clone_303 = torch.ops.aten.clone.default(permute_285, memory_format = torch.contiguous_format);  permute_285 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(clone_303, 0);  clone_303 = None
        add_294 = torch.ops.aten.add.Tensor(view_750, unsqueeze_51);  view_750 = unsqueeze_51 = None
        view_753 = torch.ops.aten.view.default(add_294, [-1, 16, 8, 49, 49]);  add_294 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(arg53_1, 1);  arg53_1 = None
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(unsqueeze_52, 0);  unsqueeze_52 = None
        add_295 = torch.ops.aten.add.Tensor(view_753, unsqueeze_53);  view_753 = unsqueeze_53 = None
        view_754 = torch.ops.aten.view.default(add_295, [-1, 8, 49, 49]);  add_295 = None
        amax_27 = torch.ops.aten.amax.default(view_754, [-1], True)
        sub_89 = torch.ops.aten.sub.Tensor(view_754, amax_27);  view_754 = amax_27 = None
        exp_27 = torch.ops.aten.exp.default(sub_89);  sub_89 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_27 = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
        expand_110 = torch.ops.aten.expand.default(div_27, [128, 8, 49, 49]);  div_27 = None
        view_755 = torch.ops.aten.view.default(expand_110, [1024, 49, 49]);  expand_110 = None
        expand_111 = torch.ops.aten.expand.default(getitem_207, [128, 8, 49, 32]);  getitem_207 = None
        clone_305 = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
        view_756 = torch.ops.aten.view.default(clone_305, [1024, 49, 32]);  clone_305 = None
        bmm_55 = torch.ops.aten.bmm.default(view_755, view_756);  view_755 = view_756 = None
        view_757 = torch.ops.aten.view.default(bmm_55, [128, 8, 49, 32]);  bmm_55 = None
        permute_286 = torch.ops.aten.permute.default(view_757, [0, 2, 1, 3]);  view_757 = None
        clone_306 = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        view_758 = torch.ops.aten.view.default(clone_306, [128, 49, 256]);  clone_306 = None
        view_759 = torch.ops.aten.view.default(view_758, [6272, 256]);  view_758 = None
        permute_287 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg59_1, view_759, permute_287);  arg59_1 = view_759 = permute_287 = None
        view_760 = torch.ops.aten.view.default(addmm_110, [128, 49, 256]);  addmm_110 = None
        view_761 = torch.ops.aten.view.default(view_760, [-1, 7, 7, 256]);  view_760 = None
        view_762 = torch.ops.aten.view.default(view_761, [-1, 4, 4, 7, 7, 256]);  view_761 = None
        permute_288 = torch.ops.aten.permute.default(view_762, [0, 1, 3, 2, 4, 5]);  view_762 = None
        clone_308 = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
        view_763 = torch.ops.aten.view.default(clone_308, [-1, 28, 28, 256]);  clone_308 = None
        iota_50 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_296 = torch.ops.aten.add.Tensor(iota_50, 25);  iota_50 = None
        fmod_50 = torch.ops.aten.fmod.Scalar(add_296, 28);  add_296 = None
        index_78 = torch.ops.aten.index.Tensor(view_763, [None, fmod_50]);  view_763 = fmod_50 = None
        iota_51 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_297 = torch.ops.aten.add.Tensor(iota_51, 25);  iota_51 = None
        fmod_51 = torch.ops.aten.fmod.Scalar(add_297, 28);  add_297 = None
        index_79 = torch.ops.aten.index.Tensor(index_78, [None, None, fmod_51]);  index_78 = fmod_51 = None
        add_298 = torch.ops.aten.add.Tensor(view_741, index_79);  view_741 = index_79 = None
        view_764 = torch.ops.aten.view.default(add_298, [8, -1, 256]);  add_298 = None
        var_mean_62 = torch.ops.aten.var_mean.correction(view_764, [2], correction = 0, keepdim = True)
        getitem_208 = var_mean_62[0]
        getitem_209 = var_mean_62[1];  var_mean_62 = None
        add_299 = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_62 = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
        sub_90 = torch.ops.aten.sub.Tensor(view_764, getitem_209);  getitem_209 = None
        mul_233 = torch.ops.aten.mul.Tensor(sub_90, rsqrt_62);  sub_90 = rsqrt_62 = None
        mul_234 = torch.ops.aten.mul.Tensor(mul_233, arg60_1);  mul_233 = arg60_1 = None
        add_300 = torch.ops.aten.add.Tensor(mul_234, arg61_1);  mul_234 = arg61_1 = None
        view_765 = torch.ops.aten.view.default(add_300, [6272, 256]);  add_300 = None
        permute_289 = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg63_1, view_765, permute_289);  arg63_1 = view_765 = permute_289 = None
        view_766 = torch.ops.aten.view.default(addmm_111, [8, 784, 1024]);  addmm_111 = None
        mul_235 = torch.ops.aten.mul.Tensor(view_766, 0.5)
        mul_236 = torch.ops.aten.mul.Tensor(view_766, 0.7071067811865476);  view_766 = None
        erf_27 = torch.ops.aten.erf.default(mul_236);  mul_236 = None
        add_301 = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_237 = torch.ops.aten.mul.Tensor(mul_235, add_301);  mul_235 = add_301 = None
        view_767 = torch.ops.aten.view.default(mul_237, [6272, 1024]);  mul_237 = None
        permute_290 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg65_1, view_767, permute_290);  arg65_1 = view_767 = permute_290 = None
        view_768 = torch.ops.aten.view.default(addmm_112, [8, 784, 256]);  addmm_112 = None
        add_302 = torch.ops.aten.add.Tensor(view_764, view_768);  view_764 = view_768 = None
        view_769 = torch.ops.aten.view.default(add_302, [8, 28, 28, 256]);  add_302 = None
        view_770 = torch.ops.aten.view.default(view_769, [8, 14, 2, 14, 2, 256]);  view_769 = None
        permute_291 = torch.ops.aten.permute.default(view_770, [0, 1, 3, 4, 2, 5]);  view_770 = None
        clone_311 = torch.ops.aten.clone.default(permute_291, memory_format = torch.contiguous_format);  permute_291 = None
        view_771 = torch.ops.aten.view.default(clone_311, [8, 14, 14, 1024]);  clone_311 = None
        var_mean_63 = torch.ops.aten.var_mean.correction(view_771, [3], correction = 0, keepdim = True)
        getitem_210 = var_mean_63[0]
        getitem_211 = var_mean_63[1];  var_mean_63 = None
        add_303 = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        sub_91 = torch.ops.aten.sub.Tensor(view_771, getitem_211);  view_771 = getitem_211 = None
        mul_238 = torch.ops.aten.mul.Tensor(sub_91, rsqrt_63);  sub_91 = rsqrt_63 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_238, arg66_1);  mul_238 = arg66_1 = None
        add_304 = torch.ops.aten.add.Tensor(mul_239, arg67_1);  mul_239 = arg67_1 = None
        permute_292 = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        view_772 = torch.ops.aten.view.default(add_304, [1568, 1024]);  add_304 = None
        mm_4 = torch.ops.aten.mm.default(view_772, permute_292);  view_772 = permute_292 = None
        view_773 = torch.ops.aten.view.default(mm_4, [8, 14, 14, 512]);  mm_4 = None
        var_mean_64 = torch.ops.aten.var_mean.correction(view_773, [3], correction = 0, keepdim = True)
        getitem_212 = var_mean_64[0]
        getitem_213 = var_mean_64[1];  var_mean_64 = None
        add_305 = torch.ops.aten.add.Tensor(getitem_212, 1e-05);  getitem_212 = None
        rsqrt_64 = torch.ops.aten.rsqrt.default(add_305);  add_305 = None
        sub_92 = torch.ops.aten.sub.Tensor(view_773, getitem_213);  getitem_213 = None
        mul_240 = torch.ops.aten.mul.Tensor(sub_92, rsqrt_64);  sub_92 = rsqrt_64 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_240, arg69_1);  mul_240 = arg69_1 = None
        add_306 = torch.ops.aten.add.Tensor(mul_241, arg70_1);  mul_241 = arg70_1 = None
        view_774 = torch.ops.aten.view.default(add_306, [8, 2, 7, 2, 7, 512]);  add_306 = None
        permute_293 = torch.ops.aten.permute.default(view_774, [0, 1, 3, 2, 4, 5]);  view_774 = None
        clone_312 = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        view_775 = torch.ops.aten.view.default(clone_312, [-1, 7, 7, 512]);  clone_312 = None
        view_776 = torch.ops.aten.view.default(view_775, [-1, 49, 512]);  view_775 = None
        view_777 = torch.ops.aten.view.default(view_776, [1568, 512]);  view_776 = None
        permute_294 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg72_1, view_777, permute_294);  arg72_1 = view_777 = permute_294 = None
        view_778 = torch.ops.aten.view.default(addmm_113, [32, 49, 1536]);  addmm_113 = None
        view_779 = torch.ops.aten.view.default(view_778, [32, 49, 3, 16, -1]);  view_778 = None
        permute_295 = torch.ops.aten.permute.default(view_779, [2, 0, 3, 1, 4]);  view_779 = None
        unbind_28 = torch.ops.aten.unbind.int(permute_295);  permute_295 = None
        getitem_214 = unbind_28[0]
        getitem_215 = unbind_28[1]
        getitem_216 = unbind_28[2];  unbind_28 = None
        mul_242 = torch.ops.aten.mul.Tensor(getitem_214, 0.1767766952966369);  getitem_214 = None
        permute_296 = torch.ops.aten.permute.default(getitem_215, [0, 1, 3, 2]);  getitem_215 = None
        expand_112 = torch.ops.aten.expand.default(mul_242, [32, 16, 49, 32]);  mul_242 = None
        clone_313 = torch.ops.aten.clone.default(expand_112, memory_format = torch.contiguous_format);  expand_112 = None
        view_780 = torch.ops.aten.view.default(clone_313, [512, 49, 32]);  clone_313 = None
        expand_113 = torch.ops.aten.expand.default(permute_296, [32, 16, 32, 49]);  permute_296 = None
        clone_314 = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_781 = torch.ops.aten.view.default(clone_314, [512, 32, 49]);  clone_314 = None
        bmm_56 = torch.ops.aten.bmm.default(view_780, view_781);  view_780 = view_781 = None
        view_782 = torch.ops.aten.view.default(bmm_56, [32, 16, 49, 49]);  bmm_56 = None
        view_783 = torch.ops.aten.view.default(arg74_1, [-1]);  arg74_1 = None
        index_80 = torch.ops.aten.index.Tensor(arg73_1, [view_783]);  arg73_1 = view_783 = None
        view_784 = torch.ops.aten.view.default(index_80, [49, 49, -1]);  index_80 = None
        permute_297 = torch.ops.aten.permute.default(view_784, [2, 0, 1]);  view_784 = None
        clone_315 = torch.ops.aten.clone.default(permute_297, memory_format = torch.contiguous_format);  permute_297 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(clone_315, 0);  clone_315 = None
        add_307 = torch.ops.aten.add.Tensor(view_782, unsqueeze_54);  view_782 = unsqueeze_54 = None
        amax_28 = torch.ops.aten.amax.default(add_307, [-1], True)
        sub_93 = torch.ops.aten.sub.Tensor(add_307, amax_28);  add_307 = amax_28 = None
        exp_28 = torch.ops.aten.exp.default(sub_93);  sub_93 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
        div_28 = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
        expand_114 = torch.ops.aten.expand.default(div_28, [32, 16, 49, 49]);  div_28 = None
        view_785 = torch.ops.aten.view.default(expand_114, [512, 49, 49]);  expand_114 = None
        expand_115 = torch.ops.aten.expand.default(getitem_216, [32, 16, 49, 32]);  getitem_216 = None
        clone_317 = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
        view_786 = torch.ops.aten.view.default(clone_317, [512, 49, 32]);  clone_317 = None
        bmm_57 = torch.ops.aten.bmm.default(view_785, view_786);  view_785 = view_786 = None
        view_787 = torch.ops.aten.view.default(bmm_57, [32, 16, 49, 32]);  bmm_57 = None
        permute_298 = torch.ops.aten.permute.default(view_787, [0, 2, 1, 3]);  view_787 = None
        clone_318 = torch.ops.aten.clone.default(permute_298, memory_format = torch.contiguous_format);  permute_298 = None
        view_788 = torch.ops.aten.view.default(clone_318, [32, 49, 512]);  clone_318 = None
        view_789 = torch.ops.aten.view.default(view_788, [1568, 512]);  view_788 = None
        permute_299 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg76_1, view_789, permute_299);  arg76_1 = view_789 = permute_299 = None
        view_790 = torch.ops.aten.view.default(addmm_114, [32, 49, 512]);  addmm_114 = None
        view_791 = torch.ops.aten.view.default(view_790, [-1, 7, 7, 512]);  view_790 = None
        view_792 = torch.ops.aten.view.default(view_791, [-1, 2, 2, 7, 7, 512]);  view_791 = None
        permute_300 = torch.ops.aten.permute.default(view_792, [0, 1, 3, 2, 4, 5]);  view_792 = None
        clone_320 = torch.ops.aten.clone.default(permute_300, memory_format = torch.contiguous_format);  permute_300 = None
        view_793 = torch.ops.aten.view.default(clone_320, [-1, 14, 14, 512]);  clone_320 = None
        add_308 = torch.ops.aten.add.Tensor(view_773, view_793);  view_773 = view_793 = None
        view_794 = torch.ops.aten.view.default(add_308, [8, -1, 512]);  add_308 = None
        var_mean_65 = torch.ops.aten.var_mean.correction(view_794, [2], correction = 0, keepdim = True)
        getitem_217 = var_mean_65[0]
        getitem_218 = var_mean_65[1];  var_mean_65 = None
        add_309 = torch.ops.aten.add.Tensor(getitem_217, 1e-05);  getitem_217 = None
        rsqrt_65 = torch.ops.aten.rsqrt.default(add_309);  add_309 = None
        sub_94 = torch.ops.aten.sub.Tensor(view_794, getitem_218);  getitem_218 = None
        mul_243 = torch.ops.aten.mul.Tensor(sub_94, rsqrt_65);  sub_94 = rsqrt_65 = None
        mul_244 = torch.ops.aten.mul.Tensor(mul_243, arg77_1);  mul_243 = arg77_1 = None
        add_310 = torch.ops.aten.add.Tensor(mul_244, arg78_1);  mul_244 = arg78_1 = None
        view_795 = torch.ops.aten.view.default(add_310, [1568, 512]);  add_310 = None
        permute_301 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg80_1, view_795, permute_301);  arg80_1 = view_795 = permute_301 = None
        view_796 = torch.ops.aten.view.default(addmm_115, [8, 196, 2048]);  addmm_115 = None
        mul_245 = torch.ops.aten.mul.Tensor(view_796, 0.5)
        mul_246 = torch.ops.aten.mul.Tensor(view_796, 0.7071067811865476);  view_796 = None
        erf_28 = torch.ops.aten.erf.default(mul_246);  mul_246 = None
        add_311 = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_247 = torch.ops.aten.mul.Tensor(mul_245, add_311);  mul_245 = add_311 = None
        view_797 = torch.ops.aten.view.default(mul_247, [1568, 2048]);  mul_247 = None
        permute_302 = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg82_1, view_797, permute_302);  arg82_1 = view_797 = permute_302 = None
        view_798 = torch.ops.aten.view.default(addmm_116, [8, 196, 512]);  addmm_116 = None
        add_312 = torch.ops.aten.add.Tensor(view_794, view_798);  view_794 = view_798 = None
        view_799 = torch.ops.aten.view.default(add_312, [8, 14, 14, 512]);  add_312 = None
        var_mean_66 = torch.ops.aten.var_mean.correction(view_799, [3], correction = 0, keepdim = True)
        getitem_219 = var_mean_66[0]
        getitem_220 = var_mean_66[1];  var_mean_66 = None
        add_313 = torch.ops.aten.add.Tensor(getitem_219, 1e-05);  getitem_219 = None
        rsqrt_66 = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        sub_95 = torch.ops.aten.sub.Tensor(view_799, getitem_220);  getitem_220 = None
        mul_248 = torch.ops.aten.mul.Tensor(sub_95, rsqrt_66);  sub_95 = rsqrt_66 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, arg83_1);  mul_248 = arg83_1 = None
        add_314 = torch.ops.aten.add.Tensor(mul_249, arg84_1);  mul_249 = arg84_1 = None
        iota_52 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_315 = torch.ops.aten.add.Tensor(iota_52, 3);  iota_52 = None
        fmod_52 = torch.ops.aten.fmod.Scalar(add_315, 14);  add_315 = None
        index_81 = torch.ops.aten.index.Tensor(add_314, [None, fmod_52]);  add_314 = fmod_52 = None
        iota_53 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_316 = torch.ops.aten.add.Tensor(iota_53, 3);  iota_53 = None
        fmod_53 = torch.ops.aten.fmod.Scalar(add_316, 14);  add_316 = None
        index_82 = torch.ops.aten.index.Tensor(index_81, [None, None, fmod_53]);  index_81 = fmod_53 = None
        view_800 = torch.ops.aten.view.default(index_82, [8, 2, 7, 2, 7, 512]);  index_82 = None
        permute_303 = torch.ops.aten.permute.default(view_800, [0, 1, 3, 2, 4, 5]);  view_800 = None
        clone_323 = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
        view_801 = torch.ops.aten.view.default(clone_323, [-1, 7, 7, 512]);  clone_323 = None
        view_802 = torch.ops.aten.view.default(view_801, [-1, 49, 512]);  view_801 = None
        view_803 = torch.ops.aten.view.default(view_802, [1568, 512]);  view_802 = None
        permute_304 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg87_1, view_803, permute_304);  arg87_1 = view_803 = permute_304 = None
        view_804 = torch.ops.aten.view.default(addmm_117, [32, 49, 1536]);  addmm_117 = None
        view_805 = torch.ops.aten.view.default(view_804, [32, 49, 3, 16, -1]);  view_804 = None
        permute_305 = torch.ops.aten.permute.default(view_805, [2, 0, 3, 1, 4]);  view_805 = None
        unbind_29 = torch.ops.aten.unbind.int(permute_305);  permute_305 = None
        getitem_221 = unbind_29[0]
        getitem_222 = unbind_29[1]
        getitem_223 = unbind_29[2];  unbind_29 = None
        mul_250 = torch.ops.aten.mul.Tensor(getitem_221, 0.1767766952966369);  getitem_221 = None
        permute_306 = torch.ops.aten.permute.default(getitem_222, [0, 1, 3, 2]);  getitem_222 = None
        expand_116 = torch.ops.aten.expand.default(mul_250, [32, 16, 49, 32]);  mul_250 = None
        clone_324 = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
        view_806 = torch.ops.aten.view.default(clone_324, [512, 49, 32]);  clone_324 = None
        expand_117 = torch.ops.aten.expand.default(permute_306, [32, 16, 32, 49]);  permute_306 = None
        clone_325 = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
        view_807 = torch.ops.aten.view.default(clone_325, [512, 32, 49]);  clone_325 = None
        bmm_58 = torch.ops.aten.bmm.default(view_806, view_807);  view_806 = view_807 = None
        view_808 = torch.ops.aten.view.default(bmm_58, [32, 16, 49, 49]);  bmm_58 = None
        view_809 = torch.ops.aten.view.default(arg89_1, [-1]);  arg89_1 = None
        index_83 = torch.ops.aten.index.Tensor(arg88_1, [view_809]);  arg88_1 = view_809 = None
        view_810 = torch.ops.aten.view.default(index_83, [49, 49, -1]);  index_83 = None
        permute_307 = torch.ops.aten.permute.default(view_810, [2, 0, 1]);  view_810 = None
        clone_326 = torch.ops.aten.clone.default(permute_307, memory_format = torch.contiguous_format);  permute_307 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(clone_326, 0);  clone_326 = None
        add_317 = torch.ops.aten.add.Tensor(view_808, unsqueeze_55);  view_808 = unsqueeze_55 = None
        view_811 = torch.ops.aten.view.default(add_317, [-1, 4, 16, 49, 49]);  add_317 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(arg85_1, 1);  arg85_1 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(unsqueeze_56, 0);  unsqueeze_56 = None
        add_318 = torch.ops.aten.add.Tensor(view_811, unsqueeze_57);  view_811 = unsqueeze_57 = None
        view_812 = torch.ops.aten.view.default(add_318, [-1, 16, 49, 49]);  add_318 = None
        amax_29 = torch.ops.aten.amax.default(view_812, [-1], True)
        sub_96 = torch.ops.aten.sub.Tensor(view_812, amax_29);  view_812 = amax_29 = None
        exp_29 = torch.ops.aten.exp.default(sub_96);  sub_96 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
        div_29 = torch.ops.aten.div.Tensor(exp_29, sum_30);  exp_29 = sum_30 = None
        expand_118 = torch.ops.aten.expand.default(div_29, [32, 16, 49, 49]);  div_29 = None
        view_813 = torch.ops.aten.view.default(expand_118, [512, 49, 49]);  expand_118 = None
        expand_119 = torch.ops.aten.expand.default(getitem_223, [32, 16, 49, 32]);  getitem_223 = None
        clone_328 = torch.ops.aten.clone.default(expand_119, memory_format = torch.contiguous_format);  expand_119 = None
        view_814 = torch.ops.aten.view.default(clone_328, [512, 49, 32]);  clone_328 = None
        bmm_59 = torch.ops.aten.bmm.default(view_813, view_814);  view_813 = view_814 = None
        view_815 = torch.ops.aten.view.default(bmm_59, [32, 16, 49, 32]);  bmm_59 = None
        permute_308 = torch.ops.aten.permute.default(view_815, [0, 2, 1, 3]);  view_815 = None
        clone_329 = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
        view_816 = torch.ops.aten.view.default(clone_329, [32, 49, 512]);  clone_329 = None
        view_817 = torch.ops.aten.view.default(view_816, [1568, 512]);  view_816 = None
        permute_309 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg91_1, view_817, permute_309);  arg91_1 = view_817 = permute_309 = None
        view_818 = torch.ops.aten.view.default(addmm_118, [32, 49, 512]);  addmm_118 = None
        view_819 = torch.ops.aten.view.default(view_818, [-1, 7, 7, 512]);  view_818 = None
        view_820 = torch.ops.aten.view.default(view_819, [-1, 2, 2, 7, 7, 512]);  view_819 = None
        permute_310 = torch.ops.aten.permute.default(view_820, [0, 1, 3, 2, 4, 5]);  view_820 = None
        clone_331 = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None
        view_821 = torch.ops.aten.view.default(clone_331, [-1, 14, 14, 512]);  clone_331 = None
        iota_54 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_319 = torch.ops.aten.add.Tensor(iota_54, 11);  iota_54 = None
        fmod_54 = torch.ops.aten.fmod.Scalar(add_319, 14);  add_319 = None
        index_84 = torch.ops.aten.index.Tensor(view_821, [None, fmod_54]);  view_821 = fmod_54 = None
        iota_55 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_320 = torch.ops.aten.add.Tensor(iota_55, 11);  iota_55 = None
        fmod_55 = torch.ops.aten.fmod.Scalar(add_320, 14);  add_320 = None
        index_85 = torch.ops.aten.index.Tensor(index_84, [None, None, fmod_55]);  index_84 = fmod_55 = None
        add_321 = torch.ops.aten.add.Tensor(view_799, index_85);  view_799 = index_85 = None
        view_822 = torch.ops.aten.view.default(add_321, [8, -1, 512]);  add_321 = None
        var_mean_67 = torch.ops.aten.var_mean.correction(view_822, [2], correction = 0, keepdim = True)
        getitem_224 = var_mean_67[0]
        getitem_225 = var_mean_67[1];  var_mean_67 = None
        add_322 = torch.ops.aten.add.Tensor(getitem_224, 1e-05);  getitem_224 = None
        rsqrt_67 = torch.ops.aten.rsqrt.default(add_322);  add_322 = None
        sub_97 = torch.ops.aten.sub.Tensor(view_822, getitem_225);  getitem_225 = None
        mul_251 = torch.ops.aten.mul.Tensor(sub_97, rsqrt_67);  sub_97 = rsqrt_67 = None
        mul_252 = torch.ops.aten.mul.Tensor(mul_251, arg92_1);  mul_251 = arg92_1 = None
        add_323 = torch.ops.aten.add.Tensor(mul_252, arg93_1);  mul_252 = arg93_1 = None
        view_823 = torch.ops.aten.view.default(add_323, [1568, 512]);  add_323 = None
        permute_311 = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg95_1, view_823, permute_311);  arg95_1 = view_823 = permute_311 = None
        view_824 = torch.ops.aten.view.default(addmm_119, [8, 196, 2048]);  addmm_119 = None
        mul_253 = torch.ops.aten.mul.Tensor(view_824, 0.5)
        mul_254 = torch.ops.aten.mul.Tensor(view_824, 0.7071067811865476);  view_824 = None
        erf_29 = torch.ops.aten.erf.default(mul_254);  mul_254 = None
        add_324 = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_255 = torch.ops.aten.mul.Tensor(mul_253, add_324);  mul_253 = add_324 = None
        view_825 = torch.ops.aten.view.default(mul_255, [1568, 2048]);  mul_255 = None
        permute_312 = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg97_1, view_825, permute_312);  arg97_1 = view_825 = permute_312 = None
        view_826 = torch.ops.aten.view.default(addmm_120, [8, 196, 512]);  addmm_120 = None
        add_325 = torch.ops.aten.add.Tensor(view_822, view_826);  view_822 = view_826 = None
        view_827 = torch.ops.aten.view.default(add_325, [8, 14, 14, 512]);  add_325 = None
        var_mean_68 = torch.ops.aten.var_mean.correction(view_827, [3], correction = 0, keepdim = True)
        getitem_226 = var_mean_68[0]
        getitem_227 = var_mean_68[1];  var_mean_68 = None
        add_326 = torch.ops.aten.add.Tensor(getitem_226, 1e-05);  getitem_226 = None
        rsqrt_68 = torch.ops.aten.rsqrt.default(add_326);  add_326 = None
        sub_98 = torch.ops.aten.sub.Tensor(view_827, getitem_227);  getitem_227 = None
        mul_256 = torch.ops.aten.mul.Tensor(sub_98, rsqrt_68);  sub_98 = rsqrt_68 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_256, arg98_1);  mul_256 = arg98_1 = None
        add_327 = torch.ops.aten.add.Tensor(mul_257, arg99_1);  mul_257 = arg99_1 = None
        view_828 = torch.ops.aten.view.default(add_327, [8, 2, 7, 2, 7, 512]);  add_327 = None
        permute_313 = torch.ops.aten.permute.default(view_828, [0, 1, 3, 2, 4, 5]);  view_828 = None
        clone_334 = torch.ops.aten.clone.default(permute_313, memory_format = torch.contiguous_format);  permute_313 = None
        view_829 = torch.ops.aten.view.default(clone_334, [-1, 7, 7, 512]);  clone_334 = None
        view_830 = torch.ops.aten.view.default(view_829, [-1, 49, 512]);  view_829 = None
        view_831 = torch.ops.aten.view.default(view_830, [1568, 512]);  view_830 = None
        permute_314 = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg101_1, view_831, permute_314);  arg101_1 = view_831 = permute_314 = None
        view_832 = torch.ops.aten.view.default(addmm_121, [32, 49, 1536]);  addmm_121 = None
        view_833 = torch.ops.aten.view.default(view_832, [32, 49, 3, 16, -1]);  view_832 = None
        permute_315 = torch.ops.aten.permute.default(view_833, [2, 0, 3, 1, 4]);  view_833 = None
        unbind_30 = torch.ops.aten.unbind.int(permute_315);  permute_315 = None
        getitem_228 = unbind_30[0]
        getitem_229 = unbind_30[1]
        getitem_230 = unbind_30[2];  unbind_30 = None
        mul_258 = torch.ops.aten.mul.Tensor(getitem_228, 0.1767766952966369);  getitem_228 = None
        permute_316 = torch.ops.aten.permute.default(getitem_229, [0, 1, 3, 2]);  getitem_229 = None
        expand_120 = torch.ops.aten.expand.default(mul_258, [32, 16, 49, 32]);  mul_258 = None
        clone_335 = torch.ops.aten.clone.default(expand_120, memory_format = torch.contiguous_format);  expand_120 = None
        view_834 = torch.ops.aten.view.default(clone_335, [512, 49, 32]);  clone_335 = None
        expand_121 = torch.ops.aten.expand.default(permute_316, [32, 16, 32, 49]);  permute_316 = None
        clone_336 = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_835 = torch.ops.aten.view.default(clone_336, [512, 32, 49]);  clone_336 = None
        bmm_60 = torch.ops.aten.bmm.default(view_834, view_835);  view_834 = view_835 = None
        view_836 = torch.ops.aten.view.default(bmm_60, [32, 16, 49, 49]);  bmm_60 = None
        view_837 = torch.ops.aten.view.default(arg103_1, [-1]);  arg103_1 = None
        index_86 = torch.ops.aten.index.Tensor(arg102_1, [view_837]);  arg102_1 = view_837 = None
        view_838 = torch.ops.aten.view.default(index_86, [49, 49, -1]);  index_86 = None
        permute_317 = torch.ops.aten.permute.default(view_838, [2, 0, 1]);  view_838 = None
        clone_337 = torch.ops.aten.clone.default(permute_317, memory_format = torch.contiguous_format);  permute_317 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(clone_337, 0);  clone_337 = None
        add_328 = torch.ops.aten.add.Tensor(view_836, unsqueeze_58);  view_836 = unsqueeze_58 = None
        amax_30 = torch.ops.aten.amax.default(add_328, [-1], True)
        sub_99 = torch.ops.aten.sub.Tensor(add_328, amax_30);  add_328 = amax_30 = None
        exp_30 = torch.ops.aten.exp.default(sub_99);  sub_99 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
        div_30 = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
        expand_122 = torch.ops.aten.expand.default(div_30, [32, 16, 49, 49]);  div_30 = None
        view_839 = torch.ops.aten.view.default(expand_122, [512, 49, 49]);  expand_122 = None
        expand_123 = torch.ops.aten.expand.default(getitem_230, [32, 16, 49, 32]);  getitem_230 = None
        clone_339 = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
        view_840 = torch.ops.aten.view.default(clone_339, [512, 49, 32]);  clone_339 = None
        bmm_61 = torch.ops.aten.bmm.default(view_839, view_840);  view_839 = view_840 = None
        view_841 = torch.ops.aten.view.default(bmm_61, [32, 16, 49, 32]);  bmm_61 = None
        permute_318 = torch.ops.aten.permute.default(view_841, [0, 2, 1, 3]);  view_841 = None
        clone_340 = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
        view_842 = torch.ops.aten.view.default(clone_340, [32, 49, 512]);  clone_340 = None
        view_843 = torch.ops.aten.view.default(view_842, [1568, 512]);  view_842 = None
        permute_319 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg105_1, view_843, permute_319);  arg105_1 = view_843 = permute_319 = None
        view_844 = torch.ops.aten.view.default(addmm_122, [32, 49, 512]);  addmm_122 = None
        view_845 = torch.ops.aten.view.default(view_844, [-1, 7, 7, 512]);  view_844 = None
        view_846 = torch.ops.aten.view.default(view_845, [-1, 2, 2, 7, 7, 512]);  view_845 = None
        permute_320 = torch.ops.aten.permute.default(view_846, [0, 1, 3, 2, 4, 5]);  view_846 = None
        clone_342 = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
        view_847 = torch.ops.aten.view.default(clone_342, [-1, 14, 14, 512]);  clone_342 = None
        add_329 = torch.ops.aten.add.Tensor(view_827, view_847);  view_827 = view_847 = None
        view_848 = torch.ops.aten.view.default(add_329, [8, -1, 512]);  add_329 = None
        var_mean_69 = torch.ops.aten.var_mean.correction(view_848, [2], correction = 0, keepdim = True)
        getitem_231 = var_mean_69[0]
        getitem_232 = var_mean_69[1];  var_mean_69 = None
        add_330 = torch.ops.aten.add.Tensor(getitem_231, 1e-05);  getitem_231 = None
        rsqrt_69 = torch.ops.aten.rsqrt.default(add_330);  add_330 = None
        sub_100 = torch.ops.aten.sub.Tensor(view_848, getitem_232);  getitem_232 = None
        mul_259 = torch.ops.aten.mul.Tensor(sub_100, rsqrt_69);  sub_100 = rsqrt_69 = None
        mul_260 = torch.ops.aten.mul.Tensor(mul_259, arg106_1);  mul_259 = arg106_1 = None
        add_331 = torch.ops.aten.add.Tensor(mul_260, arg107_1);  mul_260 = arg107_1 = None
        view_849 = torch.ops.aten.view.default(add_331, [1568, 512]);  add_331 = None
        permute_321 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg109_1, view_849, permute_321);  arg109_1 = view_849 = permute_321 = None
        view_850 = torch.ops.aten.view.default(addmm_123, [8, 196, 2048]);  addmm_123 = None
        mul_261 = torch.ops.aten.mul.Tensor(view_850, 0.5)
        mul_262 = torch.ops.aten.mul.Tensor(view_850, 0.7071067811865476);  view_850 = None
        erf_30 = torch.ops.aten.erf.default(mul_262);  mul_262 = None
        add_332 = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_261, add_332);  mul_261 = add_332 = None
        view_851 = torch.ops.aten.view.default(mul_263, [1568, 2048]);  mul_263 = None
        permute_322 = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg111_1, view_851, permute_322);  arg111_1 = view_851 = permute_322 = None
        view_852 = torch.ops.aten.view.default(addmm_124, [8, 196, 512]);  addmm_124 = None
        add_333 = torch.ops.aten.add.Tensor(view_848, view_852);  view_848 = view_852 = None
        view_853 = torch.ops.aten.view.default(add_333, [8, 14, 14, 512]);  add_333 = None
        var_mean_70 = torch.ops.aten.var_mean.correction(view_853, [3], correction = 0, keepdim = True)
        getitem_233 = var_mean_70[0]
        getitem_234 = var_mean_70[1];  var_mean_70 = None
        add_334 = torch.ops.aten.add.Tensor(getitem_233, 1e-05);  getitem_233 = None
        rsqrt_70 = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        sub_101 = torch.ops.aten.sub.Tensor(view_853, getitem_234);  getitem_234 = None
        mul_264 = torch.ops.aten.mul.Tensor(sub_101, rsqrt_70);  sub_101 = rsqrt_70 = None
        mul_265 = torch.ops.aten.mul.Tensor(mul_264, arg112_1);  mul_264 = arg112_1 = None
        add_335 = torch.ops.aten.add.Tensor(mul_265, arg113_1);  mul_265 = arg113_1 = None
        iota_56 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_336 = torch.ops.aten.add.Tensor(iota_56, 3);  iota_56 = None
        fmod_56 = torch.ops.aten.fmod.Scalar(add_336, 14);  add_336 = None
        index_87 = torch.ops.aten.index.Tensor(add_335, [None, fmod_56]);  add_335 = fmod_56 = None
        iota_57 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_337 = torch.ops.aten.add.Tensor(iota_57, 3);  iota_57 = None
        fmod_57 = torch.ops.aten.fmod.Scalar(add_337, 14);  add_337 = None
        index_88 = torch.ops.aten.index.Tensor(index_87, [None, None, fmod_57]);  index_87 = fmod_57 = None
        view_854 = torch.ops.aten.view.default(index_88, [8, 2, 7, 2, 7, 512]);  index_88 = None
        permute_323 = torch.ops.aten.permute.default(view_854, [0, 1, 3, 2, 4, 5]);  view_854 = None
        clone_345 = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
        view_855 = torch.ops.aten.view.default(clone_345, [-1, 7, 7, 512]);  clone_345 = None
        view_856 = torch.ops.aten.view.default(view_855, [-1, 49, 512]);  view_855 = None
        view_857 = torch.ops.aten.view.default(view_856, [1568, 512]);  view_856 = None
        permute_324 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg116_1, view_857, permute_324);  arg116_1 = view_857 = permute_324 = None
        view_858 = torch.ops.aten.view.default(addmm_125, [32, 49, 1536]);  addmm_125 = None
        view_859 = torch.ops.aten.view.default(view_858, [32, 49, 3, 16, -1]);  view_858 = None
        permute_325 = torch.ops.aten.permute.default(view_859, [2, 0, 3, 1, 4]);  view_859 = None
        unbind_31 = torch.ops.aten.unbind.int(permute_325);  permute_325 = None
        getitem_235 = unbind_31[0]
        getitem_236 = unbind_31[1]
        getitem_237 = unbind_31[2];  unbind_31 = None
        mul_266 = torch.ops.aten.mul.Tensor(getitem_235, 0.1767766952966369);  getitem_235 = None
        permute_326 = torch.ops.aten.permute.default(getitem_236, [0, 1, 3, 2]);  getitem_236 = None
        expand_124 = torch.ops.aten.expand.default(mul_266, [32, 16, 49, 32]);  mul_266 = None
        clone_346 = torch.ops.aten.clone.default(expand_124, memory_format = torch.contiguous_format);  expand_124 = None
        view_860 = torch.ops.aten.view.default(clone_346, [512, 49, 32]);  clone_346 = None
        expand_125 = torch.ops.aten.expand.default(permute_326, [32, 16, 32, 49]);  permute_326 = None
        clone_347 = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
        view_861 = torch.ops.aten.view.default(clone_347, [512, 32, 49]);  clone_347 = None
        bmm_62 = torch.ops.aten.bmm.default(view_860, view_861);  view_860 = view_861 = None
        view_862 = torch.ops.aten.view.default(bmm_62, [32, 16, 49, 49]);  bmm_62 = None
        view_863 = torch.ops.aten.view.default(arg118_1, [-1]);  arg118_1 = None
        index_89 = torch.ops.aten.index.Tensor(arg117_1, [view_863]);  arg117_1 = view_863 = None
        view_864 = torch.ops.aten.view.default(index_89, [49, 49, -1]);  index_89 = None
        permute_327 = torch.ops.aten.permute.default(view_864, [2, 0, 1]);  view_864 = None
        clone_348 = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(clone_348, 0);  clone_348 = None
        add_338 = torch.ops.aten.add.Tensor(view_862, unsqueeze_59);  view_862 = unsqueeze_59 = None
        view_865 = torch.ops.aten.view.default(add_338, [-1, 4, 16, 49, 49]);  add_338 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(arg114_1, 1);  arg114_1 = None
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(unsqueeze_60, 0);  unsqueeze_60 = None
        add_339 = torch.ops.aten.add.Tensor(view_865, unsqueeze_61);  view_865 = unsqueeze_61 = None
        view_866 = torch.ops.aten.view.default(add_339, [-1, 16, 49, 49]);  add_339 = None
        amax_31 = torch.ops.aten.amax.default(view_866, [-1], True)
        sub_102 = torch.ops.aten.sub.Tensor(view_866, amax_31);  view_866 = amax_31 = None
        exp_31 = torch.ops.aten.exp.default(sub_102);  sub_102 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
        div_31 = torch.ops.aten.div.Tensor(exp_31, sum_32);  exp_31 = sum_32 = None
        expand_126 = torch.ops.aten.expand.default(div_31, [32, 16, 49, 49]);  div_31 = None
        view_867 = torch.ops.aten.view.default(expand_126, [512, 49, 49]);  expand_126 = None
        expand_127 = torch.ops.aten.expand.default(getitem_237, [32, 16, 49, 32]);  getitem_237 = None
        clone_350 = torch.ops.aten.clone.default(expand_127, memory_format = torch.contiguous_format);  expand_127 = None
        view_868 = torch.ops.aten.view.default(clone_350, [512, 49, 32]);  clone_350 = None
        bmm_63 = torch.ops.aten.bmm.default(view_867, view_868);  view_867 = view_868 = None
        view_869 = torch.ops.aten.view.default(bmm_63, [32, 16, 49, 32]);  bmm_63 = None
        permute_328 = torch.ops.aten.permute.default(view_869, [0, 2, 1, 3]);  view_869 = None
        clone_351 = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
        view_870 = torch.ops.aten.view.default(clone_351, [32, 49, 512]);  clone_351 = None
        view_871 = torch.ops.aten.view.default(view_870, [1568, 512]);  view_870 = None
        permute_329 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg120_1, view_871, permute_329);  arg120_1 = view_871 = permute_329 = None
        view_872 = torch.ops.aten.view.default(addmm_126, [32, 49, 512]);  addmm_126 = None
        view_873 = torch.ops.aten.view.default(view_872, [-1, 7, 7, 512]);  view_872 = None
        view_874 = torch.ops.aten.view.default(view_873, [-1, 2, 2, 7, 7, 512]);  view_873 = None
        permute_330 = torch.ops.aten.permute.default(view_874, [0, 1, 3, 2, 4, 5]);  view_874 = None
        clone_353 = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
        view_875 = torch.ops.aten.view.default(clone_353, [-1, 14, 14, 512]);  clone_353 = None
        iota_58 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_340 = torch.ops.aten.add.Tensor(iota_58, 11);  iota_58 = None
        fmod_58 = torch.ops.aten.fmod.Scalar(add_340, 14);  add_340 = None
        index_90 = torch.ops.aten.index.Tensor(view_875, [None, fmod_58]);  view_875 = fmod_58 = None
        iota_59 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_341 = torch.ops.aten.add.Tensor(iota_59, 11);  iota_59 = None
        fmod_59 = torch.ops.aten.fmod.Scalar(add_341, 14);  add_341 = None
        index_91 = torch.ops.aten.index.Tensor(index_90, [None, None, fmod_59]);  index_90 = fmod_59 = None
        add_342 = torch.ops.aten.add.Tensor(view_853, index_91);  view_853 = index_91 = None
        view_876 = torch.ops.aten.view.default(add_342, [8, -1, 512]);  add_342 = None
        var_mean_71 = torch.ops.aten.var_mean.correction(view_876, [2], correction = 0, keepdim = True)
        getitem_238 = var_mean_71[0]
        getitem_239 = var_mean_71[1];  var_mean_71 = None
        add_343 = torch.ops.aten.add.Tensor(getitem_238, 1e-05);  getitem_238 = None
        rsqrt_71 = torch.ops.aten.rsqrt.default(add_343);  add_343 = None
        sub_103 = torch.ops.aten.sub.Tensor(view_876, getitem_239);  getitem_239 = None
        mul_267 = torch.ops.aten.mul.Tensor(sub_103, rsqrt_71);  sub_103 = rsqrt_71 = None
        mul_268 = torch.ops.aten.mul.Tensor(mul_267, arg121_1);  mul_267 = arg121_1 = None
        add_344 = torch.ops.aten.add.Tensor(mul_268, arg122_1);  mul_268 = arg122_1 = None
        view_877 = torch.ops.aten.view.default(add_344, [1568, 512]);  add_344 = None
        permute_331 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg124_1, view_877, permute_331);  arg124_1 = view_877 = permute_331 = None
        view_878 = torch.ops.aten.view.default(addmm_127, [8, 196, 2048]);  addmm_127 = None
        mul_269 = torch.ops.aten.mul.Tensor(view_878, 0.5)
        mul_270 = torch.ops.aten.mul.Tensor(view_878, 0.7071067811865476);  view_878 = None
        erf_31 = torch.ops.aten.erf.default(mul_270);  mul_270 = None
        add_345 = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_271 = torch.ops.aten.mul.Tensor(mul_269, add_345);  mul_269 = add_345 = None
        view_879 = torch.ops.aten.view.default(mul_271, [1568, 2048]);  mul_271 = None
        permute_332 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_128 = torch.ops.aten.addmm.default(arg126_1, view_879, permute_332);  arg126_1 = view_879 = permute_332 = None
        view_880 = torch.ops.aten.view.default(addmm_128, [8, 196, 512]);  addmm_128 = None
        add_346 = torch.ops.aten.add.Tensor(view_876, view_880);  view_876 = view_880 = None
        view_881 = torch.ops.aten.view.default(add_346, [8, 14, 14, 512]);  add_346 = None
        var_mean_72 = torch.ops.aten.var_mean.correction(view_881, [3], correction = 0, keepdim = True)
        getitem_240 = var_mean_72[0]
        getitem_241 = var_mean_72[1];  var_mean_72 = None
        add_347 = torch.ops.aten.add.Tensor(getitem_240, 1e-05);  getitem_240 = None
        rsqrt_72 = torch.ops.aten.rsqrt.default(add_347);  add_347 = None
        sub_104 = torch.ops.aten.sub.Tensor(view_881, getitem_241);  getitem_241 = None
        mul_272 = torch.ops.aten.mul.Tensor(sub_104, rsqrt_72);  sub_104 = rsqrt_72 = None
        mul_273 = torch.ops.aten.mul.Tensor(mul_272, arg127_1);  mul_272 = arg127_1 = None
        add_348 = torch.ops.aten.add.Tensor(mul_273, arg128_1);  mul_273 = arg128_1 = None
        view_882 = torch.ops.aten.view.default(add_348, [8, 2, 7, 2, 7, 512]);  add_348 = None
        permute_333 = torch.ops.aten.permute.default(view_882, [0, 1, 3, 2, 4, 5]);  view_882 = None
        clone_356 = torch.ops.aten.clone.default(permute_333, memory_format = torch.contiguous_format);  permute_333 = None
        view_883 = torch.ops.aten.view.default(clone_356, [-1, 7, 7, 512]);  clone_356 = None
        view_884 = torch.ops.aten.view.default(view_883, [-1, 49, 512]);  view_883 = None
        view_885 = torch.ops.aten.view.default(view_884, [1568, 512]);  view_884 = None
        permute_334 = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_129 = torch.ops.aten.addmm.default(arg130_1, view_885, permute_334);  arg130_1 = view_885 = permute_334 = None
        view_886 = torch.ops.aten.view.default(addmm_129, [32, 49, 1536]);  addmm_129 = None
        view_887 = torch.ops.aten.view.default(view_886, [32, 49, 3, 16, -1]);  view_886 = None
        permute_335 = torch.ops.aten.permute.default(view_887, [2, 0, 3, 1, 4]);  view_887 = None
        unbind_32 = torch.ops.aten.unbind.int(permute_335);  permute_335 = None
        getitem_242 = unbind_32[0]
        getitem_243 = unbind_32[1]
        getitem_244 = unbind_32[2];  unbind_32 = None
        mul_274 = torch.ops.aten.mul.Tensor(getitem_242, 0.1767766952966369);  getitem_242 = None
        permute_336 = torch.ops.aten.permute.default(getitem_243, [0, 1, 3, 2]);  getitem_243 = None
        expand_128 = torch.ops.aten.expand.default(mul_274, [32, 16, 49, 32]);  mul_274 = None
        clone_357 = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
        view_888 = torch.ops.aten.view.default(clone_357, [512, 49, 32]);  clone_357 = None
        expand_129 = torch.ops.aten.expand.default(permute_336, [32, 16, 32, 49]);  permute_336 = None
        clone_358 = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_889 = torch.ops.aten.view.default(clone_358, [512, 32, 49]);  clone_358 = None
        bmm_64 = torch.ops.aten.bmm.default(view_888, view_889);  view_888 = view_889 = None
        view_890 = torch.ops.aten.view.default(bmm_64, [32, 16, 49, 49]);  bmm_64 = None
        view_891 = torch.ops.aten.view.default(arg132_1, [-1]);  arg132_1 = None
        index_92 = torch.ops.aten.index.Tensor(arg131_1, [view_891]);  arg131_1 = view_891 = None
        view_892 = torch.ops.aten.view.default(index_92, [49, 49, -1]);  index_92 = None
        permute_337 = torch.ops.aten.permute.default(view_892, [2, 0, 1]);  view_892 = None
        clone_359 = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(clone_359, 0);  clone_359 = None
        add_349 = torch.ops.aten.add.Tensor(view_890, unsqueeze_62);  view_890 = unsqueeze_62 = None
        amax_32 = torch.ops.aten.amax.default(add_349, [-1], True)
        sub_105 = torch.ops.aten.sub.Tensor(add_349, amax_32);  add_349 = amax_32 = None
        exp_32 = torch.ops.aten.exp.default(sub_105);  sub_105 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
        div_32 = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
        expand_130 = torch.ops.aten.expand.default(div_32, [32, 16, 49, 49]);  div_32 = None
        view_893 = torch.ops.aten.view.default(expand_130, [512, 49, 49]);  expand_130 = None
        expand_131 = torch.ops.aten.expand.default(getitem_244, [32, 16, 49, 32]);  getitem_244 = None
        clone_361 = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
        view_894 = torch.ops.aten.view.default(clone_361, [512, 49, 32]);  clone_361 = None
        bmm_65 = torch.ops.aten.bmm.default(view_893, view_894);  view_893 = view_894 = None
        view_895 = torch.ops.aten.view.default(bmm_65, [32, 16, 49, 32]);  bmm_65 = None
        permute_338 = torch.ops.aten.permute.default(view_895, [0, 2, 1, 3]);  view_895 = None
        clone_362 = torch.ops.aten.clone.default(permute_338, memory_format = torch.contiguous_format);  permute_338 = None
        view_896 = torch.ops.aten.view.default(clone_362, [32, 49, 512]);  clone_362 = None
        view_897 = torch.ops.aten.view.default(view_896, [1568, 512]);  view_896 = None
        permute_339 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_130 = torch.ops.aten.addmm.default(arg134_1, view_897, permute_339);  arg134_1 = view_897 = permute_339 = None
        view_898 = torch.ops.aten.view.default(addmm_130, [32, 49, 512]);  addmm_130 = None
        view_899 = torch.ops.aten.view.default(view_898, [-1, 7, 7, 512]);  view_898 = None
        view_900 = torch.ops.aten.view.default(view_899, [-1, 2, 2, 7, 7, 512]);  view_899 = None
        permute_340 = torch.ops.aten.permute.default(view_900, [0, 1, 3, 2, 4, 5]);  view_900 = None
        clone_364 = torch.ops.aten.clone.default(permute_340, memory_format = torch.contiguous_format);  permute_340 = None
        view_901 = torch.ops.aten.view.default(clone_364, [-1, 14, 14, 512]);  clone_364 = None
        add_350 = torch.ops.aten.add.Tensor(view_881, view_901);  view_881 = view_901 = None
        view_902 = torch.ops.aten.view.default(add_350, [8, -1, 512]);  add_350 = None
        var_mean_73 = torch.ops.aten.var_mean.correction(view_902, [2], correction = 0, keepdim = True)
        getitem_245 = var_mean_73[0]
        getitem_246 = var_mean_73[1];  var_mean_73 = None
        add_351 = torch.ops.aten.add.Tensor(getitem_245, 1e-05);  getitem_245 = None
        rsqrt_73 = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
        sub_106 = torch.ops.aten.sub.Tensor(view_902, getitem_246);  getitem_246 = None
        mul_275 = torch.ops.aten.mul.Tensor(sub_106, rsqrt_73);  sub_106 = rsqrt_73 = None
        mul_276 = torch.ops.aten.mul.Tensor(mul_275, arg135_1);  mul_275 = arg135_1 = None
        add_352 = torch.ops.aten.add.Tensor(mul_276, arg136_1);  mul_276 = arg136_1 = None
        view_903 = torch.ops.aten.view.default(add_352, [1568, 512]);  add_352 = None
        permute_341 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_131 = torch.ops.aten.addmm.default(arg138_1, view_903, permute_341);  arg138_1 = view_903 = permute_341 = None
        view_904 = torch.ops.aten.view.default(addmm_131, [8, 196, 2048]);  addmm_131 = None
        mul_277 = torch.ops.aten.mul.Tensor(view_904, 0.5)
        mul_278 = torch.ops.aten.mul.Tensor(view_904, 0.7071067811865476);  view_904 = None
        erf_32 = torch.ops.aten.erf.default(mul_278);  mul_278 = None
        add_353 = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_277, add_353);  mul_277 = add_353 = None
        view_905 = torch.ops.aten.view.default(mul_279, [1568, 2048]);  mul_279 = None
        permute_342 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_132 = torch.ops.aten.addmm.default(arg140_1, view_905, permute_342);  arg140_1 = view_905 = permute_342 = None
        view_906 = torch.ops.aten.view.default(addmm_132, [8, 196, 512]);  addmm_132 = None
        add_354 = torch.ops.aten.add.Tensor(view_902, view_906);  view_902 = view_906 = None
        view_907 = torch.ops.aten.view.default(add_354, [8, 14, 14, 512]);  add_354 = None
        var_mean_74 = torch.ops.aten.var_mean.correction(view_907, [3], correction = 0, keepdim = True)
        getitem_247 = var_mean_74[0]
        getitem_248 = var_mean_74[1];  var_mean_74 = None
        add_355 = torch.ops.aten.add.Tensor(getitem_247, 1e-05);  getitem_247 = None
        rsqrt_74 = torch.ops.aten.rsqrt.default(add_355);  add_355 = None
        sub_107 = torch.ops.aten.sub.Tensor(view_907, getitem_248);  getitem_248 = None
        mul_280 = torch.ops.aten.mul.Tensor(sub_107, rsqrt_74);  sub_107 = rsqrt_74 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_280, arg141_1);  mul_280 = arg141_1 = None
        add_356 = torch.ops.aten.add.Tensor(mul_281, arg142_1);  mul_281 = arg142_1 = None
        iota_60 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_357 = torch.ops.aten.add.Tensor(iota_60, 3);  iota_60 = None
        fmod_60 = torch.ops.aten.fmod.Scalar(add_357, 14);  add_357 = None
        index_93 = torch.ops.aten.index.Tensor(add_356, [None, fmod_60]);  add_356 = fmod_60 = None
        iota_61 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_358 = torch.ops.aten.add.Tensor(iota_61, 3);  iota_61 = None
        fmod_61 = torch.ops.aten.fmod.Scalar(add_358, 14);  add_358 = None
        index_94 = torch.ops.aten.index.Tensor(index_93, [None, None, fmod_61]);  index_93 = fmod_61 = None
        view_908 = torch.ops.aten.view.default(index_94, [8, 2, 7, 2, 7, 512]);  index_94 = None
        permute_343 = torch.ops.aten.permute.default(view_908, [0, 1, 3, 2, 4, 5]);  view_908 = None
        clone_367 = torch.ops.aten.clone.default(permute_343, memory_format = torch.contiguous_format);  permute_343 = None
        view_909 = torch.ops.aten.view.default(clone_367, [-1, 7, 7, 512]);  clone_367 = None
        view_910 = torch.ops.aten.view.default(view_909, [-1, 49, 512]);  view_909 = None
        view_911 = torch.ops.aten.view.default(view_910, [1568, 512]);  view_910 = None
        permute_344 = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_133 = torch.ops.aten.addmm.default(arg145_1, view_911, permute_344);  arg145_1 = view_911 = permute_344 = None
        view_912 = torch.ops.aten.view.default(addmm_133, [32, 49, 1536]);  addmm_133 = None
        view_913 = torch.ops.aten.view.default(view_912, [32, 49, 3, 16, -1]);  view_912 = None
        permute_345 = torch.ops.aten.permute.default(view_913, [2, 0, 3, 1, 4]);  view_913 = None
        unbind_33 = torch.ops.aten.unbind.int(permute_345);  permute_345 = None
        getitem_249 = unbind_33[0]
        getitem_250 = unbind_33[1]
        getitem_251 = unbind_33[2];  unbind_33 = None
        mul_282 = torch.ops.aten.mul.Tensor(getitem_249, 0.1767766952966369);  getitem_249 = None
        permute_346 = torch.ops.aten.permute.default(getitem_250, [0, 1, 3, 2]);  getitem_250 = None
        expand_132 = torch.ops.aten.expand.default(mul_282, [32, 16, 49, 32]);  mul_282 = None
        clone_368 = torch.ops.aten.clone.default(expand_132, memory_format = torch.contiguous_format);  expand_132 = None
        view_914 = torch.ops.aten.view.default(clone_368, [512, 49, 32]);  clone_368 = None
        expand_133 = torch.ops.aten.expand.default(permute_346, [32, 16, 32, 49]);  permute_346 = None
        clone_369 = torch.ops.aten.clone.default(expand_133, memory_format = torch.contiguous_format);  expand_133 = None
        view_915 = torch.ops.aten.view.default(clone_369, [512, 32, 49]);  clone_369 = None
        bmm_66 = torch.ops.aten.bmm.default(view_914, view_915);  view_914 = view_915 = None
        view_916 = torch.ops.aten.view.default(bmm_66, [32, 16, 49, 49]);  bmm_66 = None
        view_917 = torch.ops.aten.view.default(arg147_1, [-1]);  arg147_1 = None
        index_95 = torch.ops.aten.index.Tensor(arg146_1, [view_917]);  arg146_1 = view_917 = None
        view_918 = torch.ops.aten.view.default(index_95, [49, 49, -1]);  index_95 = None
        permute_347 = torch.ops.aten.permute.default(view_918, [2, 0, 1]);  view_918 = None
        clone_370 = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(clone_370, 0);  clone_370 = None
        add_359 = torch.ops.aten.add.Tensor(view_916, unsqueeze_63);  view_916 = unsqueeze_63 = None
        view_919 = torch.ops.aten.view.default(add_359, [-1, 4, 16, 49, 49]);  add_359 = None
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(arg143_1, 1);  arg143_1 = None
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(unsqueeze_64, 0);  unsqueeze_64 = None
        add_360 = torch.ops.aten.add.Tensor(view_919, unsqueeze_65);  view_919 = unsqueeze_65 = None
        view_920 = torch.ops.aten.view.default(add_360, [-1, 16, 49, 49]);  add_360 = None
        amax_33 = torch.ops.aten.amax.default(view_920, [-1], True)
        sub_108 = torch.ops.aten.sub.Tensor(view_920, amax_33);  view_920 = amax_33 = None
        exp_33 = torch.ops.aten.exp.default(sub_108);  sub_108 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
        div_33 = torch.ops.aten.div.Tensor(exp_33, sum_34);  exp_33 = sum_34 = None
        expand_134 = torch.ops.aten.expand.default(div_33, [32, 16, 49, 49]);  div_33 = None
        view_921 = torch.ops.aten.view.default(expand_134, [512, 49, 49]);  expand_134 = None
        expand_135 = torch.ops.aten.expand.default(getitem_251, [32, 16, 49, 32]);  getitem_251 = None
        clone_372 = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
        view_922 = torch.ops.aten.view.default(clone_372, [512, 49, 32]);  clone_372 = None
        bmm_67 = torch.ops.aten.bmm.default(view_921, view_922);  view_921 = view_922 = None
        view_923 = torch.ops.aten.view.default(bmm_67, [32, 16, 49, 32]);  bmm_67 = None
        permute_348 = torch.ops.aten.permute.default(view_923, [0, 2, 1, 3]);  view_923 = None
        clone_373 = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
        view_924 = torch.ops.aten.view.default(clone_373, [32, 49, 512]);  clone_373 = None
        view_925 = torch.ops.aten.view.default(view_924, [1568, 512]);  view_924 = None
        permute_349 = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_134 = torch.ops.aten.addmm.default(arg149_1, view_925, permute_349);  arg149_1 = view_925 = permute_349 = None
        view_926 = torch.ops.aten.view.default(addmm_134, [32, 49, 512]);  addmm_134 = None
        view_927 = torch.ops.aten.view.default(view_926, [-1, 7, 7, 512]);  view_926 = None
        view_928 = torch.ops.aten.view.default(view_927, [-1, 2, 2, 7, 7, 512]);  view_927 = None
        permute_350 = torch.ops.aten.permute.default(view_928, [0, 1, 3, 2, 4, 5]);  view_928 = None
        clone_375 = torch.ops.aten.clone.default(permute_350, memory_format = torch.contiguous_format);  permute_350 = None
        view_929 = torch.ops.aten.view.default(clone_375, [-1, 14, 14, 512]);  clone_375 = None
        iota_62 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_361 = torch.ops.aten.add.Tensor(iota_62, 11);  iota_62 = None
        fmod_62 = torch.ops.aten.fmod.Scalar(add_361, 14);  add_361 = None
        index_96 = torch.ops.aten.index.Tensor(view_929, [None, fmod_62]);  view_929 = fmod_62 = None
        iota_63 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_362 = torch.ops.aten.add.Tensor(iota_63, 11);  iota_63 = None
        fmod_63 = torch.ops.aten.fmod.Scalar(add_362, 14);  add_362 = None
        index_97 = torch.ops.aten.index.Tensor(index_96, [None, None, fmod_63]);  index_96 = fmod_63 = None
        add_363 = torch.ops.aten.add.Tensor(view_907, index_97);  view_907 = index_97 = None
        view_930 = torch.ops.aten.view.default(add_363, [8, -1, 512]);  add_363 = None
        var_mean_75 = torch.ops.aten.var_mean.correction(view_930, [2], correction = 0, keepdim = True)
        getitem_252 = var_mean_75[0]
        getitem_253 = var_mean_75[1];  var_mean_75 = None
        add_364 = torch.ops.aten.add.Tensor(getitem_252, 1e-05);  getitem_252 = None
        rsqrt_75 = torch.ops.aten.rsqrt.default(add_364);  add_364 = None
        sub_109 = torch.ops.aten.sub.Tensor(view_930, getitem_253);  getitem_253 = None
        mul_283 = torch.ops.aten.mul.Tensor(sub_109, rsqrt_75);  sub_109 = rsqrt_75 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_283, arg150_1);  mul_283 = arg150_1 = None
        add_365 = torch.ops.aten.add.Tensor(mul_284, arg151_1);  mul_284 = arg151_1 = None
        view_931 = torch.ops.aten.view.default(add_365, [1568, 512]);  add_365 = None
        permute_351 = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_135 = torch.ops.aten.addmm.default(arg153_1, view_931, permute_351);  arg153_1 = view_931 = permute_351 = None
        view_932 = torch.ops.aten.view.default(addmm_135, [8, 196, 2048]);  addmm_135 = None
        mul_285 = torch.ops.aten.mul.Tensor(view_932, 0.5)
        mul_286 = torch.ops.aten.mul.Tensor(view_932, 0.7071067811865476);  view_932 = None
        erf_33 = torch.ops.aten.erf.default(mul_286);  mul_286 = None
        add_366 = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_287 = torch.ops.aten.mul.Tensor(mul_285, add_366);  mul_285 = add_366 = None
        view_933 = torch.ops.aten.view.default(mul_287, [1568, 2048]);  mul_287 = None
        permute_352 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_136 = torch.ops.aten.addmm.default(arg155_1, view_933, permute_352);  arg155_1 = view_933 = permute_352 = None
        view_934 = torch.ops.aten.view.default(addmm_136, [8, 196, 512]);  addmm_136 = None
        add_367 = torch.ops.aten.add.Tensor(view_930, view_934);  view_930 = view_934 = None
        view_935 = torch.ops.aten.view.default(add_367, [8, 14, 14, 512]);  add_367 = None
        var_mean_76 = torch.ops.aten.var_mean.correction(view_935, [3], correction = 0, keepdim = True)
        getitem_254 = var_mean_76[0]
        getitem_255 = var_mean_76[1];  var_mean_76 = None
        add_368 = torch.ops.aten.add.Tensor(getitem_254, 1e-05);  getitem_254 = None
        rsqrt_76 = torch.ops.aten.rsqrt.default(add_368);  add_368 = None
        sub_110 = torch.ops.aten.sub.Tensor(view_935, getitem_255);  getitem_255 = None
        mul_288 = torch.ops.aten.mul.Tensor(sub_110, rsqrt_76);  sub_110 = rsqrt_76 = None
        mul_289 = torch.ops.aten.mul.Tensor(mul_288, arg156_1);  mul_288 = arg156_1 = None
        add_369 = torch.ops.aten.add.Tensor(mul_289, arg157_1);  mul_289 = arg157_1 = None
        view_936 = torch.ops.aten.view.default(add_369, [8, 2, 7, 2, 7, 512]);  add_369 = None
        permute_353 = torch.ops.aten.permute.default(view_936, [0, 1, 3, 2, 4, 5]);  view_936 = None
        clone_378 = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
        view_937 = torch.ops.aten.view.default(clone_378, [-1, 7, 7, 512]);  clone_378 = None
        view_938 = torch.ops.aten.view.default(view_937, [-1, 49, 512]);  view_937 = None
        view_939 = torch.ops.aten.view.default(view_938, [1568, 512]);  view_938 = None
        permute_354 = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_137 = torch.ops.aten.addmm.default(arg159_1, view_939, permute_354);  arg159_1 = view_939 = permute_354 = None
        view_940 = torch.ops.aten.view.default(addmm_137, [32, 49, 1536]);  addmm_137 = None
        view_941 = torch.ops.aten.view.default(view_940, [32, 49, 3, 16, -1]);  view_940 = None
        permute_355 = torch.ops.aten.permute.default(view_941, [2, 0, 3, 1, 4]);  view_941 = None
        unbind_34 = torch.ops.aten.unbind.int(permute_355);  permute_355 = None
        getitem_256 = unbind_34[0]
        getitem_257 = unbind_34[1]
        getitem_258 = unbind_34[2];  unbind_34 = None
        mul_290 = torch.ops.aten.mul.Tensor(getitem_256, 0.1767766952966369);  getitem_256 = None
        permute_356 = torch.ops.aten.permute.default(getitem_257, [0, 1, 3, 2]);  getitem_257 = None
        expand_136 = torch.ops.aten.expand.default(mul_290, [32, 16, 49, 32]);  mul_290 = None
        clone_379 = torch.ops.aten.clone.default(expand_136, memory_format = torch.contiguous_format);  expand_136 = None
        view_942 = torch.ops.aten.view.default(clone_379, [512, 49, 32]);  clone_379 = None
        expand_137 = torch.ops.aten.expand.default(permute_356, [32, 16, 32, 49]);  permute_356 = None
        clone_380 = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_943 = torch.ops.aten.view.default(clone_380, [512, 32, 49]);  clone_380 = None
        bmm_68 = torch.ops.aten.bmm.default(view_942, view_943);  view_942 = view_943 = None
        view_944 = torch.ops.aten.view.default(bmm_68, [32, 16, 49, 49]);  bmm_68 = None
        view_945 = torch.ops.aten.view.default(arg161_1, [-1]);  arg161_1 = None
        index_98 = torch.ops.aten.index.Tensor(arg160_1, [view_945]);  arg160_1 = view_945 = None
        view_946 = torch.ops.aten.view.default(index_98, [49, 49, -1]);  index_98 = None
        permute_357 = torch.ops.aten.permute.default(view_946, [2, 0, 1]);  view_946 = None
        clone_381 = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(clone_381, 0);  clone_381 = None
        add_370 = torch.ops.aten.add.Tensor(view_944, unsqueeze_66);  view_944 = unsqueeze_66 = None
        amax_34 = torch.ops.aten.amax.default(add_370, [-1], True)
        sub_111 = torch.ops.aten.sub.Tensor(add_370, amax_34);  add_370 = amax_34 = None
        exp_34 = torch.ops.aten.exp.default(sub_111);  sub_111 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
        div_34 = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
        expand_138 = torch.ops.aten.expand.default(div_34, [32, 16, 49, 49]);  div_34 = None
        view_947 = torch.ops.aten.view.default(expand_138, [512, 49, 49]);  expand_138 = None
        expand_139 = torch.ops.aten.expand.default(getitem_258, [32, 16, 49, 32]);  getitem_258 = None
        clone_383 = torch.ops.aten.clone.default(expand_139, memory_format = torch.contiguous_format);  expand_139 = None
        view_948 = torch.ops.aten.view.default(clone_383, [512, 49, 32]);  clone_383 = None
        bmm_69 = torch.ops.aten.bmm.default(view_947, view_948);  view_947 = view_948 = None
        view_949 = torch.ops.aten.view.default(bmm_69, [32, 16, 49, 32]);  bmm_69 = None
        permute_358 = torch.ops.aten.permute.default(view_949, [0, 2, 1, 3]);  view_949 = None
        clone_384 = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
        view_950 = torch.ops.aten.view.default(clone_384, [32, 49, 512]);  clone_384 = None
        view_951 = torch.ops.aten.view.default(view_950, [1568, 512]);  view_950 = None
        permute_359 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_138 = torch.ops.aten.addmm.default(arg163_1, view_951, permute_359);  arg163_1 = view_951 = permute_359 = None
        view_952 = torch.ops.aten.view.default(addmm_138, [32, 49, 512]);  addmm_138 = None
        view_953 = torch.ops.aten.view.default(view_952, [-1, 7, 7, 512]);  view_952 = None
        view_954 = torch.ops.aten.view.default(view_953, [-1, 2, 2, 7, 7, 512]);  view_953 = None
        permute_360 = torch.ops.aten.permute.default(view_954, [0, 1, 3, 2, 4, 5]);  view_954 = None
        clone_386 = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None
        view_955 = torch.ops.aten.view.default(clone_386, [-1, 14, 14, 512]);  clone_386 = None
        add_371 = torch.ops.aten.add.Tensor(view_935, view_955);  view_935 = view_955 = None
        view_956 = torch.ops.aten.view.default(add_371, [8, -1, 512]);  add_371 = None
        var_mean_77 = torch.ops.aten.var_mean.correction(view_956, [2], correction = 0, keepdim = True)
        getitem_259 = var_mean_77[0]
        getitem_260 = var_mean_77[1];  var_mean_77 = None
        add_372 = torch.ops.aten.add.Tensor(getitem_259, 1e-05);  getitem_259 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_372);  add_372 = None
        sub_112 = torch.ops.aten.sub.Tensor(view_956, getitem_260);  getitem_260 = None
        mul_291 = torch.ops.aten.mul.Tensor(sub_112, rsqrt_77);  sub_112 = rsqrt_77 = None
        mul_292 = torch.ops.aten.mul.Tensor(mul_291, arg164_1);  mul_291 = arg164_1 = None
        add_373 = torch.ops.aten.add.Tensor(mul_292, arg165_1);  mul_292 = arg165_1 = None
        view_957 = torch.ops.aten.view.default(add_373, [1568, 512]);  add_373 = None
        permute_361 = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_139 = torch.ops.aten.addmm.default(arg167_1, view_957, permute_361);  arg167_1 = view_957 = permute_361 = None
        view_958 = torch.ops.aten.view.default(addmm_139, [8, 196, 2048]);  addmm_139 = None
        mul_293 = torch.ops.aten.mul.Tensor(view_958, 0.5)
        mul_294 = torch.ops.aten.mul.Tensor(view_958, 0.7071067811865476);  view_958 = None
        erf_34 = torch.ops.aten.erf.default(mul_294);  mul_294 = None
        add_374 = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_295 = torch.ops.aten.mul.Tensor(mul_293, add_374);  mul_293 = add_374 = None
        view_959 = torch.ops.aten.view.default(mul_295, [1568, 2048]);  mul_295 = None
        permute_362 = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_140 = torch.ops.aten.addmm.default(arg169_1, view_959, permute_362);  arg169_1 = view_959 = permute_362 = None
        view_960 = torch.ops.aten.view.default(addmm_140, [8, 196, 512]);  addmm_140 = None
        add_375 = torch.ops.aten.add.Tensor(view_956, view_960);  view_956 = view_960 = None
        view_961 = torch.ops.aten.view.default(add_375, [8, 14, 14, 512]);  add_375 = None
        var_mean_78 = torch.ops.aten.var_mean.correction(view_961, [3], correction = 0, keepdim = True)
        getitem_261 = var_mean_78[0]
        getitem_262 = var_mean_78[1];  var_mean_78 = None
        add_376 = torch.ops.aten.add.Tensor(getitem_261, 1e-05);  getitem_261 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
        sub_113 = torch.ops.aten.sub.Tensor(view_961, getitem_262);  getitem_262 = None
        mul_296 = torch.ops.aten.mul.Tensor(sub_113, rsqrt_78);  sub_113 = rsqrt_78 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_296, arg170_1);  mul_296 = arg170_1 = None
        add_377 = torch.ops.aten.add.Tensor(mul_297, arg171_1);  mul_297 = arg171_1 = None
        iota_64 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_378 = torch.ops.aten.add.Tensor(iota_64, 3);  iota_64 = None
        fmod_64 = torch.ops.aten.fmod.Scalar(add_378, 14);  add_378 = None
        index_99 = torch.ops.aten.index.Tensor(add_377, [None, fmod_64]);  add_377 = fmod_64 = None
        iota_65 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_379 = torch.ops.aten.add.Tensor(iota_65, 3);  iota_65 = None
        fmod_65 = torch.ops.aten.fmod.Scalar(add_379, 14);  add_379 = None
        index_100 = torch.ops.aten.index.Tensor(index_99, [None, None, fmod_65]);  index_99 = fmod_65 = None
        view_962 = torch.ops.aten.view.default(index_100, [8, 2, 7, 2, 7, 512]);  index_100 = None
        permute_363 = torch.ops.aten.permute.default(view_962, [0, 1, 3, 2, 4, 5]);  view_962 = None
        clone_389 = torch.ops.aten.clone.default(permute_363, memory_format = torch.contiguous_format);  permute_363 = None
        view_963 = torch.ops.aten.view.default(clone_389, [-1, 7, 7, 512]);  clone_389 = None
        view_964 = torch.ops.aten.view.default(view_963, [-1, 49, 512]);  view_963 = None
        view_965 = torch.ops.aten.view.default(view_964, [1568, 512]);  view_964 = None
        permute_364 = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg174_1, view_965, permute_364);  arg174_1 = view_965 = permute_364 = None
        view_966 = torch.ops.aten.view.default(addmm_141, [32, 49, 1536]);  addmm_141 = None
        view_967 = torch.ops.aten.view.default(view_966, [32, 49, 3, 16, -1]);  view_966 = None
        permute_365 = torch.ops.aten.permute.default(view_967, [2, 0, 3, 1, 4]);  view_967 = None
        unbind_35 = torch.ops.aten.unbind.int(permute_365);  permute_365 = None
        getitem_263 = unbind_35[0]
        getitem_264 = unbind_35[1]
        getitem_265 = unbind_35[2];  unbind_35 = None
        mul_298 = torch.ops.aten.mul.Tensor(getitem_263, 0.1767766952966369);  getitem_263 = None
        permute_366 = torch.ops.aten.permute.default(getitem_264, [0, 1, 3, 2]);  getitem_264 = None
        expand_140 = torch.ops.aten.expand.default(mul_298, [32, 16, 49, 32]);  mul_298 = None
        clone_390 = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
        view_968 = torch.ops.aten.view.default(clone_390, [512, 49, 32]);  clone_390 = None
        expand_141 = torch.ops.aten.expand.default(permute_366, [32, 16, 32, 49]);  permute_366 = None
        clone_391 = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
        view_969 = torch.ops.aten.view.default(clone_391, [512, 32, 49]);  clone_391 = None
        bmm_70 = torch.ops.aten.bmm.default(view_968, view_969);  view_968 = view_969 = None
        view_970 = torch.ops.aten.view.default(bmm_70, [32, 16, 49, 49]);  bmm_70 = None
        view_971 = torch.ops.aten.view.default(arg176_1, [-1]);  arg176_1 = None
        index_101 = torch.ops.aten.index.Tensor(arg175_1, [view_971]);  arg175_1 = view_971 = None
        view_972 = torch.ops.aten.view.default(index_101, [49, 49, -1]);  index_101 = None
        permute_367 = torch.ops.aten.permute.default(view_972, [2, 0, 1]);  view_972 = None
        clone_392 = torch.ops.aten.clone.default(permute_367, memory_format = torch.contiguous_format);  permute_367 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(clone_392, 0);  clone_392 = None
        add_380 = torch.ops.aten.add.Tensor(view_970, unsqueeze_67);  view_970 = unsqueeze_67 = None
        view_973 = torch.ops.aten.view.default(add_380, [-1, 4, 16, 49, 49]);  add_380 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(arg172_1, 1);  arg172_1 = None
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(unsqueeze_68, 0);  unsqueeze_68 = None
        add_381 = torch.ops.aten.add.Tensor(view_973, unsqueeze_69);  view_973 = unsqueeze_69 = None
        view_974 = torch.ops.aten.view.default(add_381, [-1, 16, 49, 49]);  add_381 = None
        amax_35 = torch.ops.aten.amax.default(view_974, [-1], True)
        sub_114 = torch.ops.aten.sub.Tensor(view_974, amax_35);  view_974 = amax_35 = None
        exp_35 = torch.ops.aten.exp.default(sub_114);  sub_114 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
        div_35 = torch.ops.aten.div.Tensor(exp_35, sum_36);  exp_35 = sum_36 = None
        expand_142 = torch.ops.aten.expand.default(div_35, [32, 16, 49, 49]);  div_35 = None
        view_975 = torch.ops.aten.view.default(expand_142, [512, 49, 49]);  expand_142 = None
        expand_143 = torch.ops.aten.expand.default(getitem_265, [32, 16, 49, 32]);  getitem_265 = None
        clone_394 = torch.ops.aten.clone.default(expand_143, memory_format = torch.contiguous_format);  expand_143 = None
        view_976 = torch.ops.aten.view.default(clone_394, [512, 49, 32]);  clone_394 = None
        bmm_71 = torch.ops.aten.bmm.default(view_975, view_976);  view_975 = view_976 = None
        view_977 = torch.ops.aten.view.default(bmm_71, [32, 16, 49, 32]);  bmm_71 = None
        permute_368 = torch.ops.aten.permute.default(view_977, [0, 2, 1, 3]);  view_977 = None
        clone_395 = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
        view_978 = torch.ops.aten.view.default(clone_395, [32, 49, 512]);  clone_395 = None
        view_979 = torch.ops.aten.view.default(view_978, [1568, 512]);  view_978 = None
        permute_369 = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg178_1, view_979, permute_369);  arg178_1 = view_979 = permute_369 = None
        view_980 = torch.ops.aten.view.default(addmm_142, [32, 49, 512]);  addmm_142 = None
        view_981 = torch.ops.aten.view.default(view_980, [-1, 7, 7, 512]);  view_980 = None
        view_982 = torch.ops.aten.view.default(view_981, [-1, 2, 2, 7, 7, 512]);  view_981 = None
        permute_370 = torch.ops.aten.permute.default(view_982, [0, 1, 3, 2, 4, 5]);  view_982 = None
        clone_397 = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
        view_983 = torch.ops.aten.view.default(clone_397, [-1, 14, 14, 512]);  clone_397 = None
        iota_66 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_382 = torch.ops.aten.add.Tensor(iota_66, 11);  iota_66 = None
        fmod_66 = torch.ops.aten.fmod.Scalar(add_382, 14);  add_382 = None
        index_102 = torch.ops.aten.index.Tensor(view_983, [None, fmod_66]);  view_983 = fmod_66 = None
        iota_67 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_383 = torch.ops.aten.add.Tensor(iota_67, 11);  iota_67 = None
        fmod_67 = torch.ops.aten.fmod.Scalar(add_383, 14);  add_383 = None
        index_103 = torch.ops.aten.index.Tensor(index_102, [None, None, fmod_67]);  index_102 = fmod_67 = None
        add_384 = torch.ops.aten.add.Tensor(view_961, index_103);  view_961 = index_103 = None
        view_984 = torch.ops.aten.view.default(add_384, [8, -1, 512]);  add_384 = None
        var_mean_79 = torch.ops.aten.var_mean.correction(view_984, [2], correction = 0, keepdim = True)
        getitem_266 = var_mean_79[0]
        getitem_267 = var_mean_79[1];  var_mean_79 = None
        add_385 = torch.ops.aten.add.Tensor(getitem_266, 1e-05);  getitem_266 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_385);  add_385 = None
        sub_115 = torch.ops.aten.sub.Tensor(view_984, getitem_267);  getitem_267 = None
        mul_299 = torch.ops.aten.mul.Tensor(sub_115, rsqrt_79);  sub_115 = rsqrt_79 = None
        mul_300 = torch.ops.aten.mul.Tensor(mul_299, arg179_1);  mul_299 = arg179_1 = None
        add_386 = torch.ops.aten.add.Tensor(mul_300, arg180_1);  mul_300 = arg180_1 = None
        view_985 = torch.ops.aten.view.default(add_386, [1568, 512]);  add_386 = None
        permute_371 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg182_1, view_985, permute_371);  arg182_1 = view_985 = permute_371 = None
        view_986 = torch.ops.aten.view.default(addmm_143, [8, 196, 2048]);  addmm_143 = None
        mul_301 = torch.ops.aten.mul.Tensor(view_986, 0.5)
        mul_302 = torch.ops.aten.mul.Tensor(view_986, 0.7071067811865476);  view_986 = None
        erf_35 = torch.ops.aten.erf.default(mul_302);  mul_302 = None
        add_387 = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_303 = torch.ops.aten.mul.Tensor(mul_301, add_387);  mul_301 = add_387 = None
        view_987 = torch.ops.aten.view.default(mul_303, [1568, 2048]);  mul_303 = None
        permute_372 = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_144 = torch.ops.aten.addmm.default(arg184_1, view_987, permute_372);  arg184_1 = view_987 = permute_372 = None
        view_988 = torch.ops.aten.view.default(addmm_144, [8, 196, 512]);  addmm_144 = None
        add_388 = torch.ops.aten.add.Tensor(view_984, view_988);  view_984 = view_988 = None
        view_989 = torch.ops.aten.view.default(add_388, [8, 14, 14, 512]);  add_388 = None
        var_mean_80 = torch.ops.aten.var_mean.correction(view_989, [3], correction = 0, keepdim = True)
        getitem_268 = var_mean_80[0]
        getitem_269 = var_mean_80[1];  var_mean_80 = None
        add_389 = torch.ops.aten.add.Tensor(getitem_268, 1e-05);  getitem_268 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_389);  add_389 = None
        sub_116 = torch.ops.aten.sub.Tensor(view_989, getitem_269);  getitem_269 = None
        mul_304 = torch.ops.aten.mul.Tensor(sub_116, rsqrt_80);  sub_116 = rsqrt_80 = None
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, arg185_1);  mul_304 = arg185_1 = None
        add_390 = torch.ops.aten.add.Tensor(mul_305, arg186_1);  mul_305 = arg186_1 = None
        view_990 = torch.ops.aten.view.default(add_390, [8, 2, 7, 2, 7, 512]);  add_390 = None
        permute_373 = torch.ops.aten.permute.default(view_990, [0, 1, 3, 2, 4, 5]);  view_990 = None
        clone_400 = torch.ops.aten.clone.default(permute_373, memory_format = torch.contiguous_format);  permute_373 = None
        view_991 = torch.ops.aten.view.default(clone_400, [-1, 7, 7, 512]);  clone_400 = None
        view_992 = torch.ops.aten.view.default(view_991, [-1, 49, 512]);  view_991 = None
        view_993 = torch.ops.aten.view.default(view_992, [1568, 512]);  view_992 = None
        permute_374 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_145 = torch.ops.aten.addmm.default(arg188_1, view_993, permute_374);  arg188_1 = view_993 = permute_374 = None
        view_994 = torch.ops.aten.view.default(addmm_145, [32, 49, 1536]);  addmm_145 = None
        view_995 = torch.ops.aten.view.default(view_994, [32, 49, 3, 16, -1]);  view_994 = None
        permute_375 = torch.ops.aten.permute.default(view_995, [2, 0, 3, 1, 4]);  view_995 = None
        unbind_36 = torch.ops.aten.unbind.int(permute_375);  permute_375 = None
        getitem_270 = unbind_36[0]
        getitem_271 = unbind_36[1]
        getitem_272 = unbind_36[2];  unbind_36 = None
        mul_306 = torch.ops.aten.mul.Tensor(getitem_270, 0.1767766952966369);  getitem_270 = None
        permute_376 = torch.ops.aten.permute.default(getitem_271, [0, 1, 3, 2]);  getitem_271 = None
        expand_144 = torch.ops.aten.expand.default(mul_306, [32, 16, 49, 32]);  mul_306 = None
        clone_401 = torch.ops.aten.clone.default(expand_144, memory_format = torch.contiguous_format);  expand_144 = None
        view_996 = torch.ops.aten.view.default(clone_401, [512, 49, 32]);  clone_401 = None
        expand_145 = torch.ops.aten.expand.default(permute_376, [32, 16, 32, 49]);  permute_376 = None
        clone_402 = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_997 = torch.ops.aten.view.default(clone_402, [512, 32, 49]);  clone_402 = None
        bmm_72 = torch.ops.aten.bmm.default(view_996, view_997);  view_996 = view_997 = None
        view_998 = torch.ops.aten.view.default(bmm_72, [32, 16, 49, 49]);  bmm_72 = None
        view_999 = torch.ops.aten.view.default(arg190_1, [-1]);  arg190_1 = None
        index_104 = torch.ops.aten.index.Tensor(arg189_1, [view_999]);  arg189_1 = view_999 = None
        view_1000 = torch.ops.aten.view.default(index_104, [49, 49, -1]);  index_104 = None
        permute_377 = torch.ops.aten.permute.default(view_1000, [2, 0, 1]);  view_1000 = None
        clone_403 = torch.ops.aten.clone.default(permute_377, memory_format = torch.contiguous_format);  permute_377 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(clone_403, 0);  clone_403 = None
        add_391 = torch.ops.aten.add.Tensor(view_998, unsqueeze_70);  view_998 = unsqueeze_70 = None
        amax_36 = torch.ops.aten.amax.default(add_391, [-1], True)
        sub_117 = torch.ops.aten.sub.Tensor(add_391, amax_36);  add_391 = amax_36 = None
        exp_36 = torch.ops.aten.exp.default(sub_117);  sub_117 = None
        sum_37 = torch.ops.aten.sum.dim_IntList(exp_36, [-1], True)
        div_36 = torch.ops.aten.div.Tensor(exp_36, sum_37);  exp_36 = sum_37 = None
        expand_146 = torch.ops.aten.expand.default(div_36, [32, 16, 49, 49]);  div_36 = None
        view_1001 = torch.ops.aten.view.default(expand_146, [512, 49, 49]);  expand_146 = None
        expand_147 = torch.ops.aten.expand.default(getitem_272, [32, 16, 49, 32]);  getitem_272 = None
        clone_405 = torch.ops.aten.clone.default(expand_147, memory_format = torch.contiguous_format);  expand_147 = None
        view_1002 = torch.ops.aten.view.default(clone_405, [512, 49, 32]);  clone_405 = None
        bmm_73 = torch.ops.aten.bmm.default(view_1001, view_1002);  view_1001 = view_1002 = None
        view_1003 = torch.ops.aten.view.default(bmm_73, [32, 16, 49, 32]);  bmm_73 = None
        permute_378 = torch.ops.aten.permute.default(view_1003, [0, 2, 1, 3]);  view_1003 = None
        clone_406 = torch.ops.aten.clone.default(permute_378, memory_format = torch.contiguous_format);  permute_378 = None
        view_1004 = torch.ops.aten.view.default(clone_406, [32, 49, 512]);  clone_406 = None
        view_1005 = torch.ops.aten.view.default(view_1004, [1568, 512]);  view_1004 = None
        permute_379 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_146 = torch.ops.aten.addmm.default(arg192_1, view_1005, permute_379);  arg192_1 = view_1005 = permute_379 = None
        view_1006 = torch.ops.aten.view.default(addmm_146, [32, 49, 512]);  addmm_146 = None
        view_1007 = torch.ops.aten.view.default(view_1006, [-1, 7, 7, 512]);  view_1006 = None
        view_1008 = torch.ops.aten.view.default(view_1007, [-1, 2, 2, 7, 7, 512]);  view_1007 = None
        permute_380 = torch.ops.aten.permute.default(view_1008, [0, 1, 3, 2, 4, 5]);  view_1008 = None
        clone_408 = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
        view_1009 = torch.ops.aten.view.default(clone_408, [-1, 14, 14, 512]);  clone_408 = None
        add_392 = torch.ops.aten.add.Tensor(view_989, view_1009);  view_989 = view_1009 = None
        view_1010 = torch.ops.aten.view.default(add_392, [8, -1, 512]);  add_392 = None
        var_mean_81 = torch.ops.aten.var_mean.correction(view_1010, [2], correction = 0, keepdim = True)
        getitem_273 = var_mean_81[0]
        getitem_274 = var_mean_81[1];  var_mean_81 = None
        add_393 = torch.ops.aten.add.Tensor(getitem_273, 1e-05);  getitem_273 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_393);  add_393 = None
        sub_118 = torch.ops.aten.sub.Tensor(view_1010, getitem_274);  getitem_274 = None
        mul_307 = torch.ops.aten.mul.Tensor(sub_118, rsqrt_81);  sub_118 = rsqrt_81 = None
        mul_308 = torch.ops.aten.mul.Tensor(mul_307, arg193_1);  mul_307 = arg193_1 = None
        add_394 = torch.ops.aten.add.Tensor(mul_308, arg194_1);  mul_308 = arg194_1 = None
        view_1011 = torch.ops.aten.view.default(add_394, [1568, 512]);  add_394 = None
        permute_381 = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_147 = torch.ops.aten.addmm.default(arg196_1, view_1011, permute_381);  arg196_1 = view_1011 = permute_381 = None
        view_1012 = torch.ops.aten.view.default(addmm_147, [8, 196, 2048]);  addmm_147 = None
        mul_309 = torch.ops.aten.mul.Tensor(view_1012, 0.5)
        mul_310 = torch.ops.aten.mul.Tensor(view_1012, 0.7071067811865476);  view_1012 = None
        erf_36 = torch.ops.aten.erf.default(mul_310);  mul_310 = None
        add_395 = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_309, add_395);  mul_309 = add_395 = None
        view_1013 = torch.ops.aten.view.default(mul_311, [1568, 2048]);  mul_311 = None
        permute_382 = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_148 = torch.ops.aten.addmm.default(arg198_1, view_1013, permute_382);  arg198_1 = view_1013 = permute_382 = None
        view_1014 = torch.ops.aten.view.default(addmm_148, [8, 196, 512]);  addmm_148 = None
        add_396 = torch.ops.aten.add.Tensor(view_1010, view_1014);  view_1010 = view_1014 = None
        view_1015 = torch.ops.aten.view.default(add_396, [8, 14, 14, 512]);  add_396 = None
        var_mean_82 = torch.ops.aten.var_mean.correction(view_1015, [3], correction = 0, keepdim = True)
        getitem_275 = var_mean_82[0]
        getitem_276 = var_mean_82[1];  var_mean_82 = None
        add_397 = torch.ops.aten.add.Tensor(getitem_275, 1e-05);  getitem_275 = None
        rsqrt_82 = torch.ops.aten.rsqrt.default(add_397);  add_397 = None
        sub_119 = torch.ops.aten.sub.Tensor(view_1015, getitem_276);  getitem_276 = None
        mul_312 = torch.ops.aten.mul.Tensor(sub_119, rsqrt_82);  sub_119 = rsqrt_82 = None
        mul_313 = torch.ops.aten.mul.Tensor(mul_312, arg199_1);  mul_312 = arg199_1 = None
        add_398 = torch.ops.aten.add.Tensor(mul_313, arg200_1);  mul_313 = arg200_1 = None
        iota_68 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_399 = torch.ops.aten.add.Tensor(iota_68, 3);  iota_68 = None
        fmod_68 = torch.ops.aten.fmod.Scalar(add_399, 14);  add_399 = None
        index_105 = torch.ops.aten.index.Tensor(add_398, [None, fmod_68]);  add_398 = fmod_68 = None
        iota_69 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_400 = torch.ops.aten.add.Tensor(iota_69, 3);  iota_69 = None
        fmod_69 = torch.ops.aten.fmod.Scalar(add_400, 14);  add_400 = None
        index_106 = torch.ops.aten.index.Tensor(index_105, [None, None, fmod_69]);  index_105 = fmod_69 = None
        view_1016 = torch.ops.aten.view.default(index_106, [8, 2, 7, 2, 7, 512]);  index_106 = None
        permute_383 = torch.ops.aten.permute.default(view_1016, [0, 1, 3, 2, 4, 5]);  view_1016 = None
        clone_411 = torch.ops.aten.clone.default(permute_383, memory_format = torch.contiguous_format);  permute_383 = None
        view_1017 = torch.ops.aten.view.default(clone_411, [-1, 7, 7, 512]);  clone_411 = None
        view_1018 = torch.ops.aten.view.default(view_1017, [-1, 49, 512]);  view_1017 = None
        view_1019 = torch.ops.aten.view.default(view_1018, [1568, 512]);  view_1018 = None
        permute_384 = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        addmm_149 = torch.ops.aten.addmm.default(arg203_1, view_1019, permute_384);  arg203_1 = view_1019 = permute_384 = None
        view_1020 = torch.ops.aten.view.default(addmm_149, [32, 49, 1536]);  addmm_149 = None
        view_1021 = torch.ops.aten.view.default(view_1020, [32, 49, 3, 16, -1]);  view_1020 = None
        permute_385 = torch.ops.aten.permute.default(view_1021, [2, 0, 3, 1, 4]);  view_1021 = None
        unbind_37 = torch.ops.aten.unbind.int(permute_385);  permute_385 = None
        getitem_277 = unbind_37[0]
        getitem_278 = unbind_37[1]
        getitem_279 = unbind_37[2];  unbind_37 = None
        mul_314 = torch.ops.aten.mul.Tensor(getitem_277, 0.1767766952966369);  getitem_277 = None
        permute_386 = torch.ops.aten.permute.default(getitem_278, [0, 1, 3, 2]);  getitem_278 = None
        expand_148 = torch.ops.aten.expand.default(mul_314, [32, 16, 49, 32]);  mul_314 = None
        clone_412 = torch.ops.aten.clone.default(expand_148, memory_format = torch.contiguous_format);  expand_148 = None
        view_1022 = torch.ops.aten.view.default(clone_412, [512, 49, 32]);  clone_412 = None
        expand_149 = torch.ops.aten.expand.default(permute_386, [32, 16, 32, 49]);  permute_386 = None
        clone_413 = torch.ops.aten.clone.default(expand_149, memory_format = torch.contiguous_format);  expand_149 = None
        view_1023 = torch.ops.aten.view.default(clone_413, [512, 32, 49]);  clone_413 = None
        bmm_74 = torch.ops.aten.bmm.default(view_1022, view_1023);  view_1022 = view_1023 = None
        view_1024 = torch.ops.aten.view.default(bmm_74, [32, 16, 49, 49]);  bmm_74 = None
        view_1025 = torch.ops.aten.view.default(arg205_1, [-1]);  arg205_1 = None
        index_107 = torch.ops.aten.index.Tensor(arg204_1, [view_1025]);  arg204_1 = view_1025 = None
        view_1026 = torch.ops.aten.view.default(index_107, [49, 49, -1]);  index_107 = None
        permute_387 = torch.ops.aten.permute.default(view_1026, [2, 0, 1]);  view_1026 = None
        clone_414 = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(clone_414, 0);  clone_414 = None
        add_401 = torch.ops.aten.add.Tensor(view_1024, unsqueeze_71);  view_1024 = unsqueeze_71 = None
        view_1027 = torch.ops.aten.view.default(add_401, [-1, 4, 16, 49, 49]);  add_401 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(arg201_1, 1);  arg201_1 = None
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(unsqueeze_72, 0);  unsqueeze_72 = None
        add_402 = torch.ops.aten.add.Tensor(view_1027, unsqueeze_73);  view_1027 = unsqueeze_73 = None
        view_1028 = torch.ops.aten.view.default(add_402, [-1, 16, 49, 49]);  add_402 = None
        amax_37 = torch.ops.aten.amax.default(view_1028, [-1], True)
        sub_120 = torch.ops.aten.sub.Tensor(view_1028, amax_37);  view_1028 = amax_37 = None
        exp_37 = torch.ops.aten.exp.default(sub_120);  sub_120 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(exp_37, [-1], True)
        div_37 = torch.ops.aten.div.Tensor(exp_37, sum_38);  exp_37 = sum_38 = None
        expand_150 = torch.ops.aten.expand.default(div_37, [32, 16, 49, 49]);  div_37 = None
        view_1029 = torch.ops.aten.view.default(expand_150, [512, 49, 49]);  expand_150 = None
        expand_151 = torch.ops.aten.expand.default(getitem_279, [32, 16, 49, 32]);  getitem_279 = None
        clone_416 = torch.ops.aten.clone.default(expand_151, memory_format = torch.contiguous_format);  expand_151 = None
        view_1030 = torch.ops.aten.view.default(clone_416, [512, 49, 32]);  clone_416 = None
        bmm_75 = torch.ops.aten.bmm.default(view_1029, view_1030);  view_1029 = view_1030 = None
        view_1031 = torch.ops.aten.view.default(bmm_75, [32, 16, 49, 32]);  bmm_75 = None
        permute_388 = torch.ops.aten.permute.default(view_1031, [0, 2, 1, 3]);  view_1031 = None
        clone_417 = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
        view_1032 = torch.ops.aten.view.default(clone_417, [32, 49, 512]);  clone_417 = None
        view_1033 = torch.ops.aten.view.default(view_1032, [1568, 512]);  view_1032 = None
        permute_389 = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        addmm_150 = torch.ops.aten.addmm.default(arg207_1, view_1033, permute_389);  arg207_1 = view_1033 = permute_389 = None
        view_1034 = torch.ops.aten.view.default(addmm_150, [32, 49, 512]);  addmm_150 = None
        view_1035 = torch.ops.aten.view.default(view_1034, [-1, 7, 7, 512]);  view_1034 = None
        view_1036 = torch.ops.aten.view.default(view_1035, [-1, 2, 2, 7, 7, 512]);  view_1035 = None
        permute_390 = torch.ops.aten.permute.default(view_1036, [0, 1, 3, 2, 4, 5]);  view_1036 = None
        clone_419 = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None
        view_1037 = torch.ops.aten.view.default(clone_419, [-1, 14, 14, 512]);  clone_419 = None
        iota_70 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_403 = torch.ops.aten.add.Tensor(iota_70, 11);  iota_70 = None
        fmod_70 = torch.ops.aten.fmod.Scalar(add_403, 14);  add_403 = None
        index_108 = torch.ops.aten.index.Tensor(view_1037, [None, fmod_70]);  view_1037 = fmod_70 = None
        iota_71 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_404 = torch.ops.aten.add.Tensor(iota_71, 11);  iota_71 = None
        fmod_71 = torch.ops.aten.fmod.Scalar(add_404, 14);  add_404 = None
        index_109 = torch.ops.aten.index.Tensor(index_108, [None, None, fmod_71]);  index_108 = fmod_71 = None
        add_405 = torch.ops.aten.add.Tensor(view_1015, index_109);  view_1015 = index_109 = None
        view_1038 = torch.ops.aten.view.default(add_405, [8, -1, 512]);  add_405 = None
        var_mean_83 = torch.ops.aten.var_mean.correction(view_1038, [2], correction = 0, keepdim = True)
        getitem_280 = var_mean_83[0]
        getitem_281 = var_mean_83[1];  var_mean_83 = None
        add_406 = torch.ops.aten.add.Tensor(getitem_280, 1e-05);  getitem_280 = None
        rsqrt_83 = torch.ops.aten.rsqrt.default(add_406);  add_406 = None
        sub_121 = torch.ops.aten.sub.Tensor(view_1038, getitem_281);  getitem_281 = None
        mul_315 = torch.ops.aten.mul.Tensor(sub_121, rsqrt_83);  sub_121 = rsqrt_83 = None
        mul_316 = torch.ops.aten.mul.Tensor(mul_315, arg208_1);  mul_315 = arg208_1 = None
        add_407 = torch.ops.aten.add.Tensor(mul_316, arg209_1);  mul_316 = arg209_1 = None
        view_1039 = torch.ops.aten.view.default(add_407, [1568, 512]);  add_407 = None
        permute_391 = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        addmm_151 = torch.ops.aten.addmm.default(arg211_1, view_1039, permute_391);  arg211_1 = view_1039 = permute_391 = None
        view_1040 = torch.ops.aten.view.default(addmm_151, [8, 196, 2048]);  addmm_151 = None
        mul_317 = torch.ops.aten.mul.Tensor(view_1040, 0.5)
        mul_318 = torch.ops.aten.mul.Tensor(view_1040, 0.7071067811865476);  view_1040 = None
        erf_37 = torch.ops.aten.erf.default(mul_318);  mul_318 = None
        add_408 = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_319 = torch.ops.aten.mul.Tensor(mul_317, add_408);  mul_317 = add_408 = None
        view_1041 = torch.ops.aten.view.default(mul_319, [1568, 2048]);  mul_319 = None
        permute_392 = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        addmm_152 = torch.ops.aten.addmm.default(arg213_1, view_1041, permute_392);  arg213_1 = view_1041 = permute_392 = None
        view_1042 = torch.ops.aten.view.default(addmm_152, [8, 196, 512]);  addmm_152 = None
        add_409 = torch.ops.aten.add.Tensor(view_1038, view_1042);  view_1038 = view_1042 = None
        view_1043 = torch.ops.aten.view.default(add_409, [8, 14, 14, 512]);  add_409 = None
        var_mean_84 = torch.ops.aten.var_mean.correction(view_1043, [3], correction = 0, keepdim = True)
        getitem_282 = var_mean_84[0]
        getitem_283 = var_mean_84[1];  var_mean_84 = None
        add_410 = torch.ops.aten.add.Tensor(getitem_282, 1e-05);  getitem_282 = None
        rsqrt_84 = torch.ops.aten.rsqrt.default(add_410);  add_410 = None
        sub_122 = torch.ops.aten.sub.Tensor(view_1043, getitem_283);  getitem_283 = None
        mul_320 = torch.ops.aten.mul.Tensor(sub_122, rsqrt_84);  sub_122 = rsqrt_84 = None
        mul_321 = torch.ops.aten.mul.Tensor(mul_320, arg214_1);  mul_320 = arg214_1 = None
        add_411 = torch.ops.aten.add.Tensor(mul_321, arg215_1);  mul_321 = arg215_1 = None
        view_1044 = torch.ops.aten.view.default(add_411, [8, 2, 7, 2, 7, 512]);  add_411 = None
        permute_393 = torch.ops.aten.permute.default(view_1044, [0, 1, 3, 2, 4, 5]);  view_1044 = None
        clone_422 = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
        view_1045 = torch.ops.aten.view.default(clone_422, [-1, 7, 7, 512]);  clone_422 = None
        view_1046 = torch.ops.aten.view.default(view_1045, [-1, 49, 512]);  view_1045 = None
        view_1047 = torch.ops.aten.view.default(view_1046, [1568, 512]);  view_1046 = None
        permute_394 = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_153 = torch.ops.aten.addmm.default(arg217_1, view_1047, permute_394);  arg217_1 = view_1047 = permute_394 = None
        view_1048 = torch.ops.aten.view.default(addmm_153, [32, 49, 1536]);  addmm_153 = None
        view_1049 = torch.ops.aten.view.default(view_1048, [32, 49, 3, 16, -1]);  view_1048 = None
        permute_395 = torch.ops.aten.permute.default(view_1049, [2, 0, 3, 1, 4]);  view_1049 = None
        unbind_38 = torch.ops.aten.unbind.int(permute_395);  permute_395 = None
        getitem_284 = unbind_38[0]
        getitem_285 = unbind_38[1]
        getitem_286 = unbind_38[2];  unbind_38 = None
        mul_322 = torch.ops.aten.mul.Tensor(getitem_284, 0.1767766952966369);  getitem_284 = None
        permute_396 = torch.ops.aten.permute.default(getitem_285, [0, 1, 3, 2]);  getitem_285 = None
        expand_152 = torch.ops.aten.expand.default(mul_322, [32, 16, 49, 32]);  mul_322 = None
        clone_423 = torch.ops.aten.clone.default(expand_152, memory_format = torch.contiguous_format);  expand_152 = None
        view_1050 = torch.ops.aten.view.default(clone_423, [512, 49, 32]);  clone_423 = None
        expand_153 = torch.ops.aten.expand.default(permute_396, [32, 16, 32, 49]);  permute_396 = None
        clone_424 = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_1051 = torch.ops.aten.view.default(clone_424, [512, 32, 49]);  clone_424 = None
        bmm_76 = torch.ops.aten.bmm.default(view_1050, view_1051);  view_1050 = view_1051 = None
        view_1052 = torch.ops.aten.view.default(bmm_76, [32, 16, 49, 49]);  bmm_76 = None
        view_1053 = torch.ops.aten.view.default(arg219_1, [-1]);  arg219_1 = None
        index_110 = torch.ops.aten.index.Tensor(arg218_1, [view_1053]);  arg218_1 = view_1053 = None
        view_1054 = torch.ops.aten.view.default(index_110, [49, 49, -1]);  index_110 = None
        permute_397 = torch.ops.aten.permute.default(view_1054, [2, 0, 1]);  view_1054 = None
        clone_425 = torch.ops.aten.clone.default(permute_397, memory_format = torch.contiguous_format);  permute_397 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(clone_425, 0);  clone_425 = None
        add_412 = torch.ops.aten.add.Tensor(view_1052, unsqueeze_74);  view_1052 = unsqueeze_74 = None
        amax_38 = torch.ops.aten.amax.default(add_412, [-1], True)
        sub_123 = torch.ops.aten.sub.Tensor(add_412, amax_38);  add_412 = amax_38 = None
        exp_38 = torch.ops.aten.exp.default(sub_123);  sub_123 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(exp_38, [-1], True)
        div_38 = torch.ops.aten.div.Tensor(exp_38, sum_39);  exp_38 = sum_39 = None
        expand_154 = torch.ops.aten.expand.default(div_38, [32, 16, 49, 49]);  div_38 = None
        view_1055 = torch.ops.aten.view.default(expand_154, [512, 49, 49]);  expand_154 = None
        expand_155 = torch.ops.aten.expand.default(getitem_286, [32, 16, 49, 32]);  getitem_286 = None
        clone_427 = torch.ops.aten.clone.default(expand_155, memory_format = torch.contiguous_format);  expand_155 = None
        view_1056 = torch.ops.aten.view.default(clone_427, [512, 49, 32]);  clone_427 = None
        bmm_77 = torch.ops.aten.bmm.default(view_1055, view_1056);  view_1055 = view_1056 = None
        view_1057 = torch.ops.aten.view.default(bmm_77, [32, 16, 49, 32]);  bmm_77 = None
        permute_398 = torch.ops.aten.permute.default(view_1057, [0, 2, 1, 3]);  view_1057 = None
        clone_428 = torch.ops.aten.clone.default(permute_398, memory_format = torch.contiguous_format);  permute_398 = None
        view_1058 = torch.ops.aten.view.default(clone_428, [32, 49, 512]);  clone_428 = None
        view_1059 = torch.ops.aten.view.default(view_1058, [1568, 512]);  view_1058 = None
        permute_399 = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm_154 = torch.ops.aten.addmm.default(arg221_1, view_1059, permute_399);  arg221_1 = view_1059 = permute_399 = None
        view_1060 = torch.ops.aten.view.default(addmm_154, [32, 49, 512]);  addmm_154 = None
        view_1061 = torch.ops.aten.view.default(view_1060, [-1, 7, 7, 512]);  view_1060 = None
        view_1062 = torch.ops.aten.view.default(view_1061, [-1, 2, 2, 7, 7, 512]);  view_1061 = None
        permute_400 = torch.ops.aten.permute.default(view_1062, [0, 1, 3, 2, 4, 5]);  view_1062 = None
        clone_430 = torch.ops.aten.clone.default(permute_400, memory_format = torch.contiguous_format);  permute_400 = None
        view_1063 = torch.ops.aten.view.default(clone_430, [-1, 14, 14, 512]);  clone_430 = None
        add_413 = torch.ops.aten.add.Tensor(view_1043, view_1063);  view_1043 = view_1063 = None
        view_1064 = torch.ops.aten.view.default(add_413, [8, -1, 512]);  add_413 = None
        var_mean_85 = torch.ops.aten.var_mean.correction(view_1064, [2], correction = 0, keepdim = True)
        getitem_287 = var_mean_85[0]
        getitem_288 = var_mean_85[1];  var_mean_85 = None
        add_414 = torch.ops.aten.add.Tensor(getitem_287, 1e-05);  getitem_287 = None
        rsqrt_85 = torch.ops.aten.rsqrt.default(add_414);  add_414 = None
        sub_124 = torch.ops.aten.sub.Tensor(view_1064, getitem_288);  getitem_288 = None
        mul_323 = torch.ops.aten.mul.Tensor(sub_124, rsqrt_85);  sub_124 = rsqrt_85 = None
        mul_324 = torch.ops.aten.mul.Tensor(mul_323, arg222_1);  mul_323 = arg222_1 = None
        add_415 = torch.ops.aten.add.Tensor(mul_324, arg223_1);  mul_324 = arg223_1 = None
        view_1065 = torch.ops.aten.view.default(add_415, [1568, 512]);  add_415 = None
        permute_401 = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        addmm_155 = torch.ops.aten.addmm.default(arg225_1, view_1065, permute_401);  arg225_1 = view_1065 = permute_401 = None
        view_1066 = torch.ops.aten.view.default(addmm_155, [8, 196, 2048]);  addmm_155 = None
        mul_325 = torch.ops.aten.mul.Tensor(view_1066, 0.5)
        mul_326 = torch.ops.aten.mul.Tensor(view_1066, 0.7071067811865476);  view_1066 = None
        erf_38 = torch.ops.aten.erf.default(mul_326);  mul_326 = None
        add_416 = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_327 = torch.ops.aten.mul.Tensor(mul_325, add_416);  mul_325 = add_416 = None
        view_1067 = torch.ops.aten.view.default(mul_327, [1568, 2048]);  mul_327 = None
        permute_402 = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        addmm_156 = torch.ops.aten.addmm.default(arg227_1, view_1067, permute_402);  arg227_1 = view_1067 = permute_402 = None
        view_1068 = torch.ops.aten.view.default(addmm_156, [8, 196, 512]);  addmm_156 = None
        add_417 = torch.ops.aten.add.Tensor(view_1064, view_1068);  view_1064 = view_1068 = None
        view_1069 = torch.ops.aten.view.default(add_417, [8, 14, 14, 512]);  add_417 = None
        var_mean_86 = torch.ops.aten.var_mean.correction(view_1069, [3], correction = 0, keepdim = True)
        getitem_289 = var_mean_86[0]
        getitem_290 = var_mean_86[1];  var_mean_86 = None
        add_418 = torch.ops.aten.add.Tensor(getitem_289, 1e-05);  getitem_289 = None
        rsqrt_86 = torch.ops.aten.rsqrt.default(add_418);  add_418 = None
        sub_125 = torch.ops.aten.sub.Tensor(view_1069, getitem_290);  getitem_290 = None
        mul_328 = torch.ops.aten.mul.Tensor(sub_125, rsqrt_86);  sub_125 = rsqrt_86 = None
        mul_329 = torch.ops.aten.mul.Tensor(mul_328, arg228_1);  mul_328 = arg228_1 = None
        add_419 = torch.ops.aten.add.Tensor(mul_329, arg229_1);  mul_329 = arg229_1 = None
        iota_72 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_420 = torch.ops.aten.add.Tensor(iota_72, 3);  iota_72 = None
        fmod_72 = torch.ops.aten.fmod.Scalar(add_420, 14);  add_420 = None
        index_111 = torch.ops.aten.index.Tensor(add_419, [None, fmod_72]);  add_419 = fmod_72 = None
        iota_73 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_421 = torch.ops.aten.add.Tensor(iota_73, 3);  iota_73 = None
        fmod_73 = torch.ops.aten.fmod.Scalar(add_421, 14);  add_421 = None
        index_112 = torch.ops.aten.index.Tensor(index_111, [None, None, fmod_73]);  index_111 = fmod_73 = None
        view_1070 = torch.ops.aten.view.default(index_112, [8, 2, 7, 2, 7, 512]);  index_112 = None
        permute_403 = torch.ops.aten.permute.default(view_1070, [0, 1, 3, 2, 4, 5]);  view_1070 = None
        clone_433 = torch.ops.aten.clone.default(permute_403, memory_format = torch.contiguous_format);  permute_403 = None
        view_1071 = torch.ops.aten.view.default(clone_433, [-1, 7, 7, 512]);  clone_433 = None
        view_1072 = torch.ops.aten.view.default(view_1071, [-1, 49, 512]);  view_1071 = None
        view_1073 = torch.ops.aten.view.default(view_1072, [1568, 512]);  view_1072 = None
        permute_404 = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_157 = torch.ops.aten.addmm.default(arg232_1, view_1073, permute_404);  arg232_1 = view_1073 = permute_404 = None
        view_1074 = torch.ops.aten.view.default(addmm_157, [32, 49, 1536]);  addmm_157 = None
        view_1075 = torch.ops.aten.view.default(view_1074, [32, 49, 3, 16, -1]);  view_1074 = None
        permute_405 = torch.ops.aten.permute.default(view_1075, [2, 0, 3, 1, 4]);  view_1075 = None
        unbind_39 = torch.ops.aten.unbind.int(permute_405);  permute_405 = None
        getitem_291 = unbind_39[0]
        getitem_292 = unbind_39[1]
        getitem_293 = unbind_39[2];  unbind_39 = None
        mul_330 = torch.ops.aten.mul.Tensor(getitem_291, 0.1767766952966369);  getitem_291 = None
        permute_406 = torch.ops.aten.permute.default(getitem_292, [0, 1, 3, 2]);  getitem_292 = None
        expand_156 = torch.ops.aten.expand.default(mul_330, [32, 16, 49, 32]);  mul_330 = None
        clone_434 = torch.ops.aten.clone.default(expand_156, memory_format = torch.contiguous_format);  expand_156 = None
        view_1076 = torch.ops.aten.view.default(clone_434, [512, 49, 32]);  clone_434 = None
        expand_157 = torch.ops.aten.expand.default(permute_406, [32, 16, 32, 49]);  permute_406 = None
        clone_435 = torch.ops.aten.clone.default(expand_157, memory_format = torch.contiguous_format);  expand_157 = None
        view_1077 = torch.ops.aten.view.default(clone_435, [512, 32, 49]);  clone_435 = None
        bmm_78 = torch.ops.aten.bmm.default(view_1076, view_1077);  view_1076 = view_1077 = None
        view_1078 = torch.ops.aten.view.default(bmm_78, [32, 16, 49, 49]);  bmm_78 = None
        view_1079 = torch.ops.aten.view.default(arg234_1, [-1]);  arg234_1 = None
        index_113 = torch.ops.aten.index.Tensor(arg233_1, [view_1079]);  arg233_1 = view_1079 = None
        view_1080 = torch.ops.aten.view.default(index_113, [49, 49, -1]);  index_113 = None
        permute_407 = torch.ops.aten.permute.default(view_1080, [2, 0, 1]);  view_1080 = None
        clone_436 = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(clone_436, 0);  clone_436 = None
        add_422 = torch.ops.aten.add.Tensor(view_1078, unsqueeze_75);  view_1078 = unsqueeze_75 = None
        view_1081 = torch.ops.aten.view.default(add_422, [-1, 4, 16, 49, 49]);  add_422 = None
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(arg230_1, 1);  arg230_1 = None
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(unsqueeze_76, 0);  unsqueeze_76 = None
        add_423 = torch.ops.aten.add.Tensor(view_1081, unsqueeze_77);  view_1081 = unsqueeze_77 = None
        view_1082 = torch.ops.aten.view.default(add_423, [-1, 16, 49, 49]);  add_423 = None
        amax_39 = torch.ops.aten.amax.default(view_1082, [-1], True)
        sub_126 = torch.ops.aten.sub.Tensor(view_1082, amax_39);  view_1082 = amax_39 = None
        exp_39 = torch.ops.aten.exp.default(sub_126);  sub_126 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(exp_39, [-1], True)
        div_39 = torch.ops.aten.div.Tensor(exp_39, sum_40);  exp_39 = sum_40 = None
        expand_158 = torch.ops.aten.expand.default(div_39, [32, 16, 49, 49]);  div_39 = None
        view_1083 = torch.ops.aten.view.default(expand_158, [512, 49, 49]);  expand_158 = None
        expand_159 = torch.ops.aten.expand.default(getitem_293, [32, 16, 49, 32]);  getitem_293 = None
        clone_438 = torch.ops.aten.clone.default(expand_159, memory_format = torch.contiguous_format);  expand_159 = None
        view_1084 = torch.ops.aten.view.default(clone_438, [512, 49, 32]);  clone_438 = None
        bmm_79 = torch.ops.aten.bmm.default(view_1083, view_1084);  view_1083 = view_1084 = None
        view_1085 = torch.ops.aten.view.default(bmm_79, [32, 16, 49, 32]);  bmm_79 = None
        permute_408 = torch.ops.aten.permute.default(view_1085, [0, 2, 1, 3]);  view_1085 = None
        clone_439 = torch.ops.aten.clone.default(permute_408, memory_format = torch.contiguous_format);  permute_408 = None
        view_1086 = torch.ops.aten.view.default(clone_439, [32, 49, 512]);  clone_439 = None
        view_1087 = torch.ops.aten.view.default(view_1086, [1568, 512]);  view_1086 = None
        permute_409 = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_158 = torch.ops.aten.addmm.default(arg236_1, view_1087, permute_409);  arg236_1 = view_1087 = permute_409 = None
        view_1088 = torch.ops.aten.view.default(addmm_158, [32, 49, 512]);  addmm_158 = None
        view_1089 = torch.ops.aten.view.default(view_1088, [-1, 7, 7, 512]);  view_1088 = None
        view_1090 = torch.ops.aten.view.default(view_1089, [-1, 2, 2, 7, 7, 512]);  view_1089 = None
        permute_410 = torch.ops.aten.permute.default(view_1090, [0, 1, 3, 2, 4, 5]);  view_1090 = None
        clone_441 = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
        view_1091 = torch.ops.aten.view.default(clone_441, [-1, 14, 14, 512]);  clone_441 = None
        iota_74 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_424 = torch.ops.aten.add.Tensor(iota_74, 11);  iota_74 = None
        fmod_74 = torch.ops.aten.fmod.Scalar(add_424, 14);  add_424 = None
        index_114 = torch.ops.aten.index.Tensor(view_1091, [None, fmod_74]);  view_1091 = fmod_74 = None
        iota_75 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_425 = torch.ops.aten.add.Tensor(iota_75, 11);  iota_75 = None
        fmod_75 = torch.ops.aten.fmod.Scalar(add_425, 14);  add_425 = None
        index_115 = torch.ops.aten.index.Tensor(index_114, [None, None, fmod_75]);  index_114 = fmod_75 = None
        add_426 = torch.ops.aten.add.Tensor(view_1069, index_115);  view_1069 = index_115 = None
        view_1092 = torch.ops.aten.view.default(add_426, [8, -1, 512]);  add_426 = None
        var_mean_87 = torch.ops.aten.var_mean.correction(view_1092, [2], correction = 0, keepdim = True)
        getitem_294 = var_mean_87[0]
        getitem_295 = var_mean_87[1];  var_mean_87 = None
        add_427 = torch.ops.aten.add.Tensor(getitem_294, 1e-05);  getitem_294 = None
        rsqrt_87 = torch.ops.aten.rsqrt.default(add_427);  add_427 = None
        sub_127 = torch.ops.aten.sub.Tensor(view_1092, getitem_295);  getitem_295 = None
        mul_331 = torch.ops.aten.mul.Tensor(sub_127, rsqrt_87);  sub_127 = rsqrt_87 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_331, arg237_1);  mul_331 = arg237_1 = None
        add_428 = torch.ops.aten.add.Tensor(mul_332, arg238_1);  mul_332 = arg238_1 = None
        view_1093 = torch.ops.aten.view.default(add_428, [1568, 512]);  add_428 = None
        permute_411 = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_159 = torch.ops.aten.addmm.default(arg240_1, view_1093, permute_411);  arg240_1 = view_1093 = permute_411 = None
        view_1094 = torch.ops.aten.view.default(addmm_159, [8, 196, 2048]);  addmm_159 = None
        mul_333 = torch.ops.aten.mul.Tensor(view_1094, 0.5)
        mul_334 = torch.ops.aten.mul.Tensor(view_1094, 0.7071067811865476);  view_1094 = None
        erf_39 = torch.ops.aten.erf.default(mul_334);  mul_334 = None
        add_429 = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_333, add_429);  mul_333 = add_429 = None
        view_1095 = torch.ops.aten.view.default(mul_335, [1568, 2048]);  mul_335 = None
        permute_412 = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_160 = torch.ops.aten.addmm.default(arg242_1, view_1095, permute_412);  arg242_1 = view_1095 = permute_412 = None
        view_1096 = torch.ops.aten.view.default(addmm_160, [8, 196, 512]);  addmm_160 = None
        add_430 = torch.ops.aten.add.Tensor(view_1092, view_1096);  view_1092 = view_1096 = None
        view_1097 = torch.ops.aten.view.default(add_430, [8, 14, 14, 512]);  add_430 = None
        var_mean_88 = torch.ops.aten.var_mean.correction(view_1097, [3], correction = 0, keepdim = True)
        getitem_296 = var_mean_88[0]
        getitem_297 = var_mean_88[1];  var_mean_88 = None
        add_431 = torch.ops.aten.add.Tensor(getitem_296, 1e-05);  getitem_296 = None
        rsqrt_88 = torch.ops.aten.rsqrt.default(add_431);  add_431 = None
        sub_128 = torch.ops.aten.sub.Tensor(view_1097, getitem_297);  getitem_297 = None
        mul_336 = torch.ops.aten.mul.Tensor(sub_128, rsqrt_88);  sub_128 = rsqrt_88 = None
        mul_337 = torch.ops.aten.mul.Tensor(mul_336, arg243_1);  mul_336 = arg243_1 = None
        add_432 = torch.ops.aten.add.Tensor(mul_337, arg244_1);  mul_337 = arg244_1 = None
        view_1098 = torch.ops.aten.view.default(add_432, [8, 2, 7, 2, 7, 512]);  add_432 = None
        permute_413 = torch.ops.aten.permute.default(view_1098, [0, 1, 3, 2, 4, 5]);  view_1098 = None
        clone_444 = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
        view_1099 = torch.ops.aten.view.default(clone_444, [-1, 7, 7, 512]);  clone_444 = None
        view_1100 = torch.ops.aten.view.default(view_1099, [-1, 49, 512]);  view_1099 = None
        view_1101 = torch.ops.aten.view.default(view_1100, [1568, 512]);  view_1100 = None
        permute_414 = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_161 = torch.ops.aten.addmm.default(arg246_1, view_1101, permute_414);  arg246_1 = view_1101 = permute_414 = None
        view_1102 = torch.ops.aten.view.default(addmm_161, [32, 49, 1536]);  addmm_161 = None
        view_1103 = torch.ops.aten.view.default(view_1102, [32, 49, 3, 16, -1]);  view_1102 = None
        permute_415 = torch.ops.aten.permute.default(view_1103, [2, 0, 3, 1, 4]);  view_1103 = None
        unbind_40 = torch.ops.aten.unbind.int(permute_415);  permute_415 = None
        getitem_298 = unbind_40[0]
        getitem_299 = unbind_40[1]
        getitem_300 = unbind_40[2];  unbind_40 = None
        mul_338 = torch.ops.aten.mul.Tensor(getitem_298, 0.1767766952966369);  getitem_298 = None
        permute_416 = torch.ops.aten.permute.default(getitem_299, [0, 1, 3, 2]);  getitem_299 = None
        expand_160 = torch.ops.aten.expand.default(mul_338, [32, 16, 49, 32]);  mul_338 = None
        clone_445 = torch.ops.aten.clone.default(expand_160, memory_format = torch.contiguous_format);  expand_160 = None
        view_1104 = torch.ops.aten.view.default(clone_445, [512, 49, 32]);  clone_445 = None
        expand_161 = torch.ops.aten.expand.default(permute_416, [32, 16, 32, 49]);  permute_416 = None
        clone_446 = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_1105 = torch.ops.aten.view.default(clone_446, [512, 32, 49]);  clone_446 = None
        bmm_80 = torch.ops.aten.bmm.default(view_1104, view_1105);  view_1104 = view_1105 = None
        view_1106 = torch.ops.aten.view.default(bmm_80, [32, 16, 49, 49]);  bmm_80 = None
        view_1107 = torch.ops.aten.view.default(arg248_1, [-1]);  arg248_1 = None
        index_116 = torch.ops.aten.index.Tensor(arg247_1, [view_1107]);  arg247_1 = view_1107 = None
        view_1108 = torch.ops.aten.view.default(index_116, [49, 49, -1]);  index_116 = None
        permute_417 = torch.ops.aten.permute.default(view_1108, [2, 0, 1]);  view_1108 = None
        clone_447 = torch.ops.aten.clone.default(permute_417, memory_format = torch.contiguous_format);  permute_417 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(clone_447, 0);  clone_447 = None
        add_433 = torch.ops.aten.add.Tensor(view_1106, unsqueeze_78);  view_1106 = unsqueeze_78 = None
        amax_40 = torch.ops.aten.amax.default(add_433, [-1], True)
        sub_129 = torch.ops.aten.sub.Tensor(add_433, amax_40);  add_433 = amax_40 = None
        exp_40 = torch.ops.aten.exp.default(sub_129);  sub_129 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(exp_40, [-1], True)
        div_40 = torch.ops.aten.div.Tensor(exp_40, sum_41);  exp_40 = sum_41 = None
        expand_162 = torch.ops.aten.expand.default(div_40, [32, 16, 49, 49]);  div_40 = None
        view_1109 = torch.ops.aten.view.default(expand_162, [512, 49, 49]);  expand_162 = None
        expand_163 = torch.ops.aten.expand.default(getitem_300, [32, 16, 49, 32]);  getitem_300 = None
        clone_449 = torch.ops.aten.clone.default(expand_163, memory_format = torch.contiguous_format);  expand_163 = None
        view_1110 = torch.ops.aten.view.default(clone_449, [512, 49, 32]);  clone_449 = None
        bmm_81 = torch.ops.aten.bmm.default(view_1109, view_1110);  view_1109 = view_1110 = None
        view_1111 = torch.ops.aten.view.default(bmm_81, [32, 16, 49, 32]);  bmm_81 = None
        permute_418 = torch.ops.aten.permute.default(view_1111, [0, 2, 1, 3]);  view_1111 = None
        clone_450 = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        view_1112 = torch.ops.aten.view.default(clone_450, [32, 49, 512]);  clone_450 = None
        view_1113 = torch.ops.aten.view.default(view_1112, [1568, 512]);  view_1112 = None
        permute_419 = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_162 = torch.ops.aten.addmm.default(arg250_1, view_1113, permute_419);  arg250_1 = view_1113 = permute_419 = None
        view_1114 = torch.ops.aten.view.default(addmm_162, [32, 49, 512]);  addmm_162 = None
        view_1115 = torch.ops.aten.view.default(view_1114, [-1, 7, 7, 512]);  view_1114 = None
        view_1116 = torch.ops.aten.view.default(view_1115, [-1, 2, 2, 7, 7, 512]);  view_1115 = None
        permute_420 = torch.ops.aten.permute.default(view_1116, [0, 1, 3, 2, 4, 5]);  view_1116 = None
        clone_452 = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
        view_1117 = torch.ops.aten.view.default(clone_452, [-1, 14, 14, 512]);  clone_452 = None
        add_434 = torch.ops.aten.add.Tensor(view_1097, view_1117);  view_1097 = view_1117 = None
        view_1118 = torch.ops.aten.view.default(add_434, [8, -1, 512]);  add_434 = None
        var_mean_89 = torch.ops.aten.var_mean.correction(view_1118, [2], correction = 0, keepdim = True)
        getitem_301 = var_mean_89[0]
        getitem_302 = var_mean_89[1];  var_mean_89 = None
        add_435 = torch.ops.aten.add.Tensor(getitem_301, 1e-05);  getitem_301 = None
        rsqrt_89 = torch.ops.aten.rsqrt.default(add_435);  add_435 = None
        sub_130 = torch.ops.aten.sub.Tensor(view_1118, getitem_302);  getitem_302 = None
        mul_339 = torch.ops.aten.mul.Tensor(sub_130, rsqrt_89);  sub_130 = rsqrt_89 = None
        mul_340 = torch.ops.aten.mul.Tensor(mul_339, arg251_1);  mul_339 = arg251_1 = None
        add_436 = torch.ops.aten.add.Tensor(mul_340, arg252_1);  mul_340 = arg252_1 = None
        view_1119 = torch.ops.aten.view.default(add_436, [1568, 512]);  add_436 = None
        permute_421 = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_163 = torch.ops.aten.addmm.default(arg254_1, view_1119, permute_421);  arg254_1 = view_1119 = permute_421 = None
        view_1120 = torch.ops.aten.view.default(addmm_163, [8, 196, 2048]);  addmm_163 = None
        mul_341 = torch.ops.aten.mul.Tensor(view_1120, 0.5)
        mul_342 = torch.ops.aten.mul.Tensor(view_1120, 0.7071067811865476);  view_1120 = None
        erf_40 = torch.ops.aten.erf.default(mul_342);  mul_342 = None
        add_437 = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_343 = torch.ops.aten.mul.Tensor(mul_341, add_437);  mul_341 = add_437 = None
        view_1121 = torch.ops.aten.view.default(mul_343, [1568, 2048]);  mul_343 = None
        permute_422 = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_164 = torch.ops.aten.addmm.default(arg256_1, view_1121, permute_422);  arg256_1 = view_1121 = permute_422 = None
        view_1122 = torch.ops.aten.view.default(addmm_164, [8, 196, 512]);  addmm_164 = None
        add_438 = torch.ops.aten.add.Tensor(view_1118, view_1122);  view_1118 = view_1122 = None
        view_1123 = torch.ops.aten.view.default(add_438, [8, 14, 14, 512]);  add_438 = None
        var_mean_90 = torch.ops.aten.var_mean.correction(view_1123, [3], correction = 0, keepdim = True)
        getitem_303 = var_mean_90[0]
        getitem_304 = var_mean_90[1];  var_mean_90 = None
        add_439 = torch.ops.aten.add.Tensor(getitem_303, 1e-05);  getitem_303 = None
        rsqrt_90 = torch.ops.aten.rsqrt.default(add_439);  add_439 = None
        sub_131 = torch.ops.aten.sub.Tensor(view_1123, getitem_304);  getitem_304 = None
        mul_344 = torch.ops.aten.mul.Tensor(sub_131, rsqrt_90);  sub_131 = rsqrt_90 = None
        mul_345 = torch.ops.aten.mul.Tensor(mul_344, arg257_1);  mul_344 = arg257_1 = None
        add_440 = torch.ops.aten.add.Tensor(mul_345, arg258_1);  mul_345 = arg258_1 = None
        iota_76 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_441 = torch.ops.aten.add.Tensor(iota_76, 3);  iota_76 = None
        fmod_76 = torch.ops.aten.fmod.Scalar(add_441, 14);  add_441 = None
        index_117 = torch.ops.aten.index.Tensor(add_440, [None, fmod_76]);  add_440 = fmod_76 = None
        iota_77 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_442 = torch.ops.aten.add.Tensor(iota_77, 3);  iota_77 = None
        fmod_77 = torch.ops.aten.fmod.Scalar(add_442, 14);  add_442 = None
        index_118 = torch.ops.aten.index.Tensor(index_117, [None, None, fmod_77]);  index_117 = fmod_77 = None
        view_1124 = torch.ops.aten.view.default(index_118, [8, 2, 7, 2, 7, 512]);  index_118 = None
        permute_423 = torch.ops.aten.permute.default(view_1124, [0, 1, 3, 2, 4, 5]);  view_1124 = None
        clone_455 = torch.ops.aten.clone.default(permute_423, memory_format = torch.contiguous_format);  permute_423 = None
        view_1125 = torch.ops.aten.view.default(clone_455, [-1, 7, 7, 512]);  clone_455 = None
        view_1126 = torch.ops.aten.view.default(view_1125, [-1, 49, 512]);  view_1125 = None
        view_1127 = torch.ops.aten.view.default(view_1126, [1568, 512]);  view_1126 = None
        permute_424 = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
        addmm_165 = torch.ops.aten.addmm.default(arg261_1, view_1127, permute_424);  arg261_1 = view_1127 = permute_424 = None
        view_1128 = torch.ops.aten.view.default(addmm_165, [32, 49, 1536]);  addmm_165 = None
        view_1129 = torch.ops.aten.view.default(view_1128, [32, 49, 3, 16, -1]);  view_1128 = None
        permute_425 = torch.ops.aten.permute.default(view_1129, [2, 0, 3, 1, 4]);  view_1129 = None
        unbind_41 = torch.ops.aten.unbind.int(permute_425);  permute_425 = None
        getitem_305 = unbind_41[0]
        getitem_306 = unbind_41[1]
        getitem_307 = unbind_41[2];  unbind_41 = None
        mul_346 = torch.ops.aten.mul.Tensor(getitem_305, 0.1767766952966369);  getitem_305 = None
        permute_426 = torch.ops.aten.permute.default(getitem_306, [0, 1, 3, 2]);  getitem_306 = None
        expand_164 = torch.ops.aten.expand.default(mul_346, [32, 16, 49, 32]);  mul_346 = None
        clone_456 = torch.ops.aten.clone.default(expand_164, memory_format = torch.contiguous_format);  expand_164 = None
        view_1130 = torch.ops.aten.view.default(clone_456, [512, 49, 32]);  clone_456 = None
        expand_165 = torch.ops.aten.expand.default(permute_426, [32, 16, 32, 49]);  permute_426 = None
        clone_457 = torch.ops.aten.clone.default(expand_165, memory_format = torch.contiguous_format);  expand_165 = None
        view_1131 = torch.ops.aten.view.default(clone_457, [512, 32, 49]);  clone_457 = None
        bmm_82 = torch.ops.aten.bmm.default(view_1130, view_1131);  view_1130 = view_1131 = None
        view_1132 = torch.ops.aten.view.default(bmm_82, [32, 16, 49, 49]);  bmm_82 = None
        view_1133 = torch.ops.aten.view.default(arg263_1, [-1]);  arg263_1 = None
        index_119 = torch.ops.aten.index.Tensor(arg262_1, [view_1133]);  arg262_1 = view_1133 = None
        view_1134 = torch.ops.aten.view.default(index_119, [49, 49, -1]);  index_119 = None
        permute_427 = torch.ops.aten.permute.default(view_1134, [2, 0, 1]);  view_1134 = None
        clone_458 = torch.ops.aten.clone.default(permute_427, memory_format = torch.contiguous_format);  permute_427 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(clone_458, 0);  clone_458 = None
        add_443 = torch.ops.aten.add.Tensor(view_1132, unsqueeze_79);  view_1132 = unsqueeze_79 = None
        view_1135 = torch.ops.aten.view.default(add_443, [-1, 4, 16, 49, 49]);  add_443 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(arg259_1, 1);  arg259_1 = None
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, 0);  unsqueeze_80 = None
        add_444 = torch.ops.aten.add.Tensor(view_1135, unsqueeze_81);  view_1135 = unsqueeze_81 = None
        view_1136 = torch.ops.aten.view.default(add_444, [-1, 16, 49, 49]);  add_444 = None
        amax_41 = torch.ops.aten.amax.default(view_1136, [-1], True)
        sub_132 = torch.ops.aten.sub.Tensor(view_1136, amax_41);  view_1136 = amax_41 = None
        exp_41 = torch.ops.aten.exp.default(sub_132);  sub_132 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(exp_41, [-1], True)
        div_41 = torch.ops.aten.div.Tensor(exp_41, sum_42);  exp_41 = sum_42 = None
        expand_166 = torch.ops.aten.expand.default(div_41, [32, 16, 49, 49]);  div_41 = None
        view_1137 = torch.ops.aten.view.default(expand_166, [512, 49, 49]);  expand_166 = None
        expand_167 = torch.ops.aten.expand.default(getitem_307, [32, 16, 49, 32]);  getitem_307 = None
        clone_460 = torch.ops.aten.clone.default(expand_167, memory_format = torch.contiguous_format);  expand_167 = None
        view_1138 = torch.ops.aten.view.default(clone_460, [512, 49, 32]);  clone_460 = None
        bmm_83 = torch.ops.aten.bmm.default(view_1137, view_1138);  view_1137 = view_1138 = None
        view_1139 = torch.ops.aten.view.default(bmm_83, [32, 16, 49, 32]);  bmm_83 = None
        permute_428 = torch.ops.aten.permute.default(view_1139, [0, 2, 1, 3]);  view_1139 = None
        clone_461 = torch.ops.aten.clone.default(permute_428, memory_format = torch.contiguous_format);  permute_428 = None
        view_1140 = torch.ops.aten.view.default(clone_461, [32, 49, 512]);  clone_461 = None
        view_1141 = torch.ops.aten.view.default(view_1140, [1568, 512]);  view_1140 = None
        permute_429 = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        addmm_166 = torch.ops.aten.addmm.default(arg265_1, view_1141, permute_429);  arg265_1 = view_1141 = permute_429 = None
        view_1142 = torch.ops.aten.view.default(addmm_166, [32, 49, 512]);  addmm_166 = None
        view_1143 = torch.ops.aten.view.default(view_1142, [-1, 7, 7, 512]);  view_1142 = None
        view_1144 = torch.ops.aten.view.default(view_1143, [-1, 2, 2, 7, 7, 512]);  view_1143 = None
        permute_430 = torch.ops.aten.permute.default(view_1144, [0, 1, 3, 2, 4, 5]);  view_1144 = None
        clone_463 = torch.ops.aten.clone.default(permute_430, memory_format = torch.contiguous_format);  permute_430 = None
        view_1145 = torch.ops.aten.view.default(clone_463, [-1, 14, 14, 512]);  clone_463 = None
        iota_78 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_445 = torch.ops.aten.add.Tensor(iota_78, 11);  iota_78 = None
        fmod_78 = torch.ops.aten.fmod.Scalar(add_445, 14);  add_445 = None
        index_120 = torch.ops.aten.index.Tensor(view_1145, [None, fmod_78]);  view_1145 = fmod_78 = None
        iota_79 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_446 = torch.ops.aten.add.Tensor(iota_79, 11);  iota_79 = None
        fmod_79 = torch.ops.aten.fmod.Scalar(add_446, 14);  add_446 = None
        index_121 = torch.ops.aten.index.Tensor(index_120, [None, None, fmod_79]);  index_120 = fmod_79 = None
        add_447 = torch.ops.aten.add.Tensor(view_1123, index_121);  view_1123 = index_121 = None
        view_1146 = torch.ops.aten.view.default(add_447, [8, -1, 512]);  add_447 = None
        var_mean_91 = torch.ops.aten.var_mean.correction(view_1146, [2], correction = 0, keepdim = True)
        getitem_308 = var_mean_91[0]
        getitem_309 = var_mean_91[1];  var_mean_91 = None
        add_448 = torch.ops.aten.add.Tensor(getitem_308, 1e-05);  getitem_308 = None
        rsqrt_91 = torch.ops.aten.rsqrt.default(add_448);  add_448 = None
        sub_133 = torch.ops.aten.sub.Tensor(view_1146, getitem_309);  getitem_309 = None
        mul_347 = torch.ops.aten.mul.Tensor(sub_133, rsqrt_91);  sub_133 = rsqrt_91 = None
        mul_348 = torch.ops.aten.mul.Tensor(mul_347, arg266_1);  mul_347 = arg266_1 = None
        add_449 = torch.ops.aten.add.Tensor(mul_348, arg267_1);  mul_348 = arg267_1 = None
        view_1147 = torch.ops.aten.view.default(add_449, [1568, 512]);  add_449 = None
        permute_431 = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_167 = torch.ops.aten.addmm.default(arg269_1, view_1147, permute_431);  arg269_1 = view_1147 = permute_431 = None
        view_1148 = torch.ops.aten.view.default(addmm_167, [8, 196, 2048]);  addmm_167 = None
        mul_349 = torch.ops.aten.mul.Tensor(view_1148, 0.5)
        mul_350 = torch.ops.aten.mul.Tensor(view_1148, 0.7071067811865476);  view_1148 = None
        erf_41 = torch.ops.aten.erf.default(mul_350);  mul_350 = None
        add_450 = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_351 = torch.ops.aten.mul.Tensor(mul_349, add_450);  mul_349 = add_450 = None
        view_1149 = torch.ops.aten.view.default(mul_351, [1568, 2048]);  mul_351 = None
        permute_432 = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        addmm_168 = torch.ops.aten.addmm.default(arg271_1, view_1149, permute_432);  arg271_1 = view_1149 = permute_432 = None
        view_1150 = torch.ops.aten.view.default(addmm_168, [8, 196, 512]);  addmm_168 = None
        add_451 = torch.ops.aten.add.Tensor(view_1146, view_1150);  view_1146 = view_1150 = None
        view_1151 = torch.ops.aten.view.default(add_451, [8, 14, 14, 512]);  add_451 = None
        var_mean_92 = torch.ops.aten.var_mean.correction(view_1151, [3], correction = 0, keepdim = True)
        getitem_310 = var_mean_92[0]
        getitem_311 = var_mean_92[1];  var_mean_92 = None
        add_452 = torch.ops.aten.add.Tensor(getitem_310, 1e-05);  getitem_310 = None
        rsqrt_92 = torch.ops.aten.rsqrt.default(add_452);  add_452 = None
        sub_134 = torch.ops.aten.sub.Tensor(view_1151, getitem_311);  getitem_311 = None
        mul_352 = torch.ops.aten.mul.Tensor(sub_134, rsqrt_92);  sub_134 = rsqrt_92 = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_352, arg272_1);  mul_352 = arg272_1 = None
        add_453 = torch.ops.aten.add.Tensor(mul_353, arg273_1);  mul_353 = arg273_1 = None
        view_1152 = torch.ops.aten.view.default(add_453, [8, 2, 7, 2, 7, 512]);  add_453 = None
        permute_433 = torch.ops.aten.permute.default(view_1152, [0, 1, 3, 2, 4, 5]);  view_1152 = None
        clone_466 = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
        view_1153 = torch.ops.aten.view.default(clone_466, [-1, 7, 7, 512]);  clone_466 = None
        view_1154 = torch.ops.aten.view.default(view_1153, [-1, 49, 512]);  view_1153 = None
        view_1155 = torch.ops.aten.view.default(view_1154, [1568, 512]);  view_1154 = None
        permute_434 = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_169 = torch.ops.aten.addmm.default(arg275_1, view_1155, permute_434);  arg275_1 = view_1155 = permute_434 = None
        view_1156 = torch.ops.aten.view.default(addmm_169, [32, 49, 1536]);  addmm_169 = None
        view_1157 = torch.ops.aten.view.default(view_1156, [32, 49, 3, 16, -1]);  view_1156 = None
        permute_435 = torch.ops.aten.permute.default(view_1157, [2, 0, 3, 1, 4]);  view_1157 = None
        unbind_42 = torch.ops.aten.unbind.int(permute_435);  permute_435 = None
        getitem_312 = unbind_42[0]
        getitem_313 = unbind_42[1]
        getitem_314 = unbind_42[2];  unbind_42 = None
        mul_354 = torch.ops.aten.mul.Tensor(getitem_312, 0.1767766952966369);  getitem_312 = None
        permute_436 = torch.ops.aten.permute.default(getitem_313, [0, 1, 3, 2]);  getitem_313 = None
        expand_168 = torch.ops.aten.expand.default(mul_354, [32, 16, 49, 32]);  mul_354 = None
        clone_467 = torch.ops.aten.clone.default(expand_168, memory_format = torch.contiguous_format);  expand_168 = None
        view_1158 = torch.ops.aten.view.default(clone_467, [512, 49, 32]);  clone_467 = None
        expand_169 = torch.ops.aten.expand.default(permute_436, [32, 16, 32, 49]);  permute_436 = None
        clone_468 = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_1159 = torch.ops.aten.view.default(clone_468, [512, 32, 49]);  clone_468 = None
        bmm_84 = torch.ops.aten.bmm.default(view_1158, view_1159);  view_1158 = view_1159 = None
        view_1160 = torch.ops.aten.view.default(bmm_84, [32, 16, 49, 49]);  bmm_84 = None
        view_1161 = torch.ops.aten.view.default(arg277_1, [-1]);  arg277_1 = None
        index_122 = torch.ops.aten.index.Tensor(arg276_1, [view_1161]);  arg276_1 = view_1161 = None
        view_1162 = torch.ops.aten.view.default(index_122, [49, 49, -1]);  index_122 = None
        permute_437 = torch.ops.aten.permute.default(view_1162, [2, 0, 1]);  view_1162 = None
        clone_469 = torch.ops.aten.clone.default(permute_437, memory_format = torch.contiguous_format);  permute_437 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(clone_469, 0);  clone_469 = None
        add_454 = torch.ops.aten.add.Tensor(view_1160, unsqueeze_82);  view_1160 = unsqueeze_82 = None
        amax_42 = torch.ops.aten.amax.default(add_454, [-1], True)
        sub_135 = torch.ops.aten.sub.Tensor(add_454, amax_42);  add_454 = amax_42 = None
        exp_42 = torch.ops.aten.exp.default(sub_135);  sub_135 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(exp_42, [-1], True)
        div_42 = torch.ops.aten.div.Tensor(exp_42, sum_43);  exp_42 = sum_43 = None
        expand_170 = torch.ops.aten.expand.default(div_42, [32, 16, 49, 49]);  div_42 = None
        view_1163 = torch.ops.aten.view.default(expand_170, [512, 49, 49]);  expand_170 = None
        expand_171 = torch.ops.aten.expand.default(getitem_314, [32, 16, 49, 32]);  getitem_314 = None
        clone_471 = torch.ops.aten.clone.default(expand_171, memory_format = torch.contiguous_format);  expand_171 = None
        view_1164 = torch.ops.aten.view.default(clone_471, [512, 49, 32]);  clone_471 = None
        bmm_85 = torch.ops.aten.bmm.default(view_1163, view_1164);  view_1163 = view_1164 = None
        view_1165 = torch.ops.aten.view.default(bmm_85, [32, 16, 49, 32]);  bmm_85 = None
        permute_438 = torch.ops.aten.permute.default(view_1165, [0, 2, 1, 3]);  view_1165 = None
        clone_472 = torch.ops.aten.clone.default(permute_438, memory_format = torch.contiguous_format);  permute_438 = None
        view_1166 = torch.ops.aten.view.default(clone_472, [32, 49, 512]);  clone_472 = None
        view_1167 = torch.ops.aten.view.default(view_1166, [1568, 512]);  view_1166 = None
        permute_439 = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_170 = torch.ops.aten.addmm.default(arg279_1, view_1167, permute_439);  arg279_1 = view_1167 = permute_439 = None
        view_1168 = torch.ops.aten.view.default(addmm_170, [32, 49, 512]);  addmm_170 = None
        view_1169 = torch.ops.aten.view.default(view_1168, [-1, 7, 7, 512]);  view_1168 = None
        view_1170 = torch.ops.aten.view.default(view_1169, [-1, 2, 2, 7, 7, 512]);  view_1169 = None
        permute_440 = torch.ops.aten.permute.default(view_1170, [0, 1, 3, 2, 4, 5]);  view_1170 = None
        clone_474 = torch.ops.aten.clone.default(permute_440, memory_format = torch.contiguous_format);  permute_440 = None
        view_1171 = torch.ops.aten.view.default(clone_474, [-1, 14, 14, 512]);  clone_474 = None
        add_455 = torch.ops.aten.add.Tensor(view_1151, view_1171);  view_1151 = view_1171 = None
        view_1172 = torch.ops.aten.view.default(add_455, [8, -1, 512]);  add_455 = None
        var_mean_93 = torch.ops.aten.var_mean.correction(view_1172, [2], correction = 0, keepdim = True)
        getitem_315 = var_mean_93[0]
        getitem_316 = var_mean_93[1];  var_mean_93 = None
        add_456 = torch.ops.aten.add.Tensor(getitem_315, 1e-05);  getitem_315 = None
        rsqrt_93 = torch.ops.aten.rsqrt.default(add_456);  add_456 = None
        sub_136 = torch.ops.aten.sub.Tensor(view_1172, getitem_316);  getitem_316 = None
        mul_355 = torch.ops.aten.mul.Tensor(sub_136, rsqrt_93);  sub_136 = rsqrt_93 = None
        mul_356 = torch.ops.aten.mul.Tensor(mul_355, arg280_1);  mul_355 = arg280_1 = None
        add_457 = torch.ops.aten.add.Tensor(mul_356, arg281_1);  mul_356 = arg281_1 = None
        view_1173 = torch.ops.aten.view.default(add_457, [1568, 512]);  add_457 = None
        permute_441 = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        addmm_171 = torch.ops.aten.addmm.default(arg283_1, view_1173, permute_441);  arg283_1 = view_1173 = permute_441 = None
        view_1174 = torch.ops.aten.view.default(addmm_171, [8, 196, 2048]);  addmm_171 = None
        mul_357 = torch.ops.aten.mul.Tensor(view_1174, 0.5)
        mul_358 = torch.ops.aten.mul.Tensor(view_1174, 0.7071067811865476);  view_1174 = None
        erf_42 = torch.ops.aten.erf.default(mul_358);  mul_358 = None
        add_458 = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_359 = torch.ops.aten.mul.Tensor(mul_357, add_458);  mul_357 = add_458 = None
        view_1175 = torch.ops.aten.view.default(mul_359, [1568, 2048]);  mul_359 = None
        permute_442 = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        addmm_172 = torch.ops.aten.addmm.default(arg285_1, view_1175, permute_442);  arg285_1 = view_1175 = permute_442 = None
        view_1176 = torch.ops.aten.view.default(addmm_172, [8, 196, 512]);  addmm_172 = None
        add_459 = torch.ops.aten.add.Tensor(view_1172, view_1176);  view_1172 = view_1176 = None
        view_1177 = torch.ops.aten.view.default(add_459, [8, 14, 14, 512]);  add_459 = None
        var_mean_94 = torch.ops.aten.var_mean.correction(view_1177, [3], correction = 0, keepdim = True)
        getitem_317 = var_mean_94[0]
        getitem_318 = var_mean_94[1];  var_mean_94 = None
        add_460 = torch.ops.aten.add.Tensor(getitem_317, 1e-05);  getitem_317 = None
        rsqrt_94 = torch.ops.aten.rsqrt.default(add_460);  add_460 = None
        sub_137 = torch.ops.aten.sub.Tensor(view_1177, getitem_318);  getitem_318 = None
        mul_360 = torch.ops.aten.mul.Tensor(sub_137, rsqrt_94);  sub_137 = rsqrt_94 = None
        mul_361 = torch.ops.aten.mul.Tensor(mul_360, arg286_1);  mul_360 = arg286_1 = None
        add_461 = torch.ops.aten.add.Tensor(mul_361, arg287_1);  mul_361 = arg287_1 = None
        iota_80 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_462 = torch.ops.aten.add.Tensor(iota_80, 3);  iota_80 = None
        fmod_80 = torch.ops.aten.fmod.Scalar(add_462, 14);  add_462 = None
        index_123 = torch.ops.aten.index.Tensor(add_461, [None, fmod_80]);  add_461 = fmod_80 = None
        iota_81 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_463 = torch.ops.aten.add.Tensor(iota_81, 3);  iota_81 = None
        fmod_81 = torch.ops.aten.fmod.Scalar(add_463, 14);  add_463 = None
        index_124 = torch.ops.aten.index.Tensor(index_123, [None, None, fmod_81]);  index_123 = fmod_81 = None
        view_1178 = torch.ops.aten.view.default(index_124, [8, 2, 7, 2, 7, 512]);  index_124 = None
        permute_443 = torch.ops.aten.permute.default(view_1178, [0, 1, 3, 2, 4, 5]);  view_1178 = None
        clone_477 = torch.ops.aten.clone.default(permute_443, memory_format = torch.contiguous_format);  permute_443 = None
        view_1179 = torch.ops.aten.view.default(clone_477, [-1, 7, 7, 512]);  clone_477 = None
        view_1180 = torch.ops.aten.view.default(view_1179, [-1, 49, 512]);  view_1179 = None
        view_1181 = torch.ops.aten.view.default(view_1180, [1568, 512]);  view_1180 = None
        permute_444 = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_173 = torch.ops.aten.addmm.default(arg290_1, view_1181, permute_444);  arg290_1 = view_1181 = permute_444 = None
        view_1182 = torch.ops.aten.view.default(addmm_173, [32, 49, 1536]);  addmm_173 = None
        view_1183 = torch.ops.aten.view.default(view_1182, [32, 49, 3, 16, -1]);  view_1182 = None
        permute_445 = torch.ops.aten.permute.default(view_1183, [2, 0, 3, 1, 4]);  view_1183 = None
        unbind_43 = torch.ops.aten.unbind.int(permute_445);  permute_445 = None
        getitem_319 = unbind_43[0]
        getitem_320 = unbind_43[1]
        getitem_321 = unbind_43[2];  unbind_43 = None
        mul_362 = torch.ops.aten.mul.Tensor(getitem_319, 0.1767766952966369);  getitem_319 = None
        permute_446 = torch.ops.aten.permute.default(getitem_320, [0, 1, 3, 2]);  getitem_320 = None
        expand_172 = torch.ops.aten.expand.default(mul_362, [32, 16, 49, 32]);  mul_362 = None
        clone_478 = torch.ops.aten.clone.default(expand_172, memory_format = torch.contiguous_format);  expand_172 = None
        view_1184 = torch.ops.aten.view.default(clone_478, [512, 49, 32]);  clone_478 = None
        expand_173 = torch.ops.aten.expand.default(permute_446, [32, 16, 32, 49]);  permute_446 = None
        clone_479 = torch.ops.aten.clone.default(expand_173, memory_format = torch.contiguous_format);  expand_173 = None
        view_1185 = torch.ops.aten.view.default(clone_479, [512, 32, 49]);  clone_479 = None
        bmm_86 = torch.ops.aten.bmm.default(view_1184, view_1185);  view_1184 = view_1185 = None
        view_1186 = torch.ops.aten.view.default(bmm_86, [32, 16, 49, 49]);  bmm_86 = None
        view_1187 = torch.ops.aten.view.default(arg292_1, [-1]);  arg292_1 = None
        index_125 = torch.ops.aten.index.Tensor(arg291_1, [view_1187]);  arg291_1 = view_1187 = None
        view_1188 = torch.ops.aten.view.default(index_125, [49, 49, -1]);  index_125 = None
        permute_447 = torch.ops.aten.permute.default(view_1188, [2, 0, 1]);  view_1188 = None
        clone_480 = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(clone_480, 0);  clone_480 = None
        add_464 = torch.ops.aten.add.Tensor(view_1186, unsqueeze_83);  view_1186 = unsqueeze_83 = None
        view_1189 = torch.ops.aten.view.default(add_464, [-1, 4, 16, 49, 49]);  add_464 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(arg288_1, 1);  arg288_1 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, 0);  unsqueeze_84 = None
        add_465 = torch.ops.aten.add.Tensor(view_1189, unsqueeze_85);  view_1189 = unsqueeze_85 = None
        view_1190 = torch.ops.aten.view.default(add_465, [-1, 16, 49, 49]);  add_465 = None
        amax_43 = torch.ops.aten.amax.default(view_1190, [-1], True)
        sub_138 = torch.ops.aten.sub.Tensor(view_1190, amax_43);  view_1190 = amax_43 = None
        exp_43 = torch.ops.aten.exp.default(sub_138);  sub_138 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(exp_43, [-1], True)
        div_43 = torch.ops.aten.div.Tensor(exp_43, sum_44);  exp_43 = sum_44 = None
        expand_174 = torch.ops.aten.expand.default(div_43, [32, 16, 49, 49]);  div_43 = None
        view_1191 = torch.ops.aten.view.default(expand_174, [512, 49, 49]);  expand_174 = None
        expand_175 = torch.ops.aten.expand.default(getitem_321, [32, 16, 49, 32]);  getitem_321 = None
        clone_482 = torch.ops.aten.clone.default(expand_175, memory_format = torch.contiguous_format);  expand_175 = None
        view_1192 = torch.ops.aten.view.default(clone_482, [512, 49, 32]);  clone_482 = None
        bmm_87 = torch.ops.aten.bmm.default(view_1191, view_1192);  view_1191 = view_1192 = None
        view_1193 = torch.ops.aten.view.default(bmm_87, [32, 16, 49, 32]);  bmm_87 = None
        permute_448 = torch.ops.aten.permute.default(view_1193, [0, 2, 1, 3]);  view_1193 = None
        clone_483 = torch.ops.aten.clone.default(permute_448, memory_format = torch.contiguous_format);  permute_448 = None
        view_1194 = torch.ops.aten.view.default(clone_483, [32, 49, 512]);  clone_483 = None
        view_1195 = torch.ops.aten.view.default(view_1194, [1568, 512]);  view_1194 = None
        permute_449 = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_174 = torch.ops.aten.addmm.default(arg294_1, view_1195, permute_449);  arg294_1 = view_1195 = permute_449 = None
        view_1196 = torch.ops.aten.view.default(addmm_174, [32, 49, 512]);  addmm_174 = None
        view_1197 = torch.ops.aten.view.default(view_1196, [-1, 7, 7, 512]);  view_1196 = None
        view_1198 = torch.ops.aten.view.default(view_1197, [-1, 2, 2, 7, 7, 512]);  view_1197 = None
        permute_450 = torch.ops.aten.permute.default(view_1198, [0, 1, 3, 2, 4, 5]);  view_1198 = None
        clone_485 = torch.ops.aten.clone.default(permute_450, memory_format = torch.contiguous_format);  permute_450 = None
        view_1199 = torch.ops.aten.view.default(clone_485, [-1, 14, 14, 512]);  clone_485 = None
        iota_82 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_466 = torch.ops.aten.add.Tensor(iota_82, 11);  iota_82 = None
        fmod_82 = torch.ops.aten.fmod.Scalar(add_466, 14);  add_466 = None
        index_126 = torch.ops.aten.index.Tensor(view_1199, [None, fmod_82]);  view_1199 = fmod_82 = None
        iota_83 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_467 = torch.ops.aten.add.Tensor(iota_83, 11);  iota_83 = None
        fmod_83 = torch.ops.aten.fmod.Scalar(add_467, 14);  add_467 = None
        index_127 = torch.ops.aten.index.Tensor(index_126, [None, None, fmod_83]);  index_126 = fmod_83 = None
        add_468 = torch.ops.aten.add.Tensor(view_1177, index_127);  view_1177 = index_127 = None
        view_1200 = torch.ops.aten.view.default(add_468, [8, -1, 512]);  add_468 = None
        var_mean_95 = torch.ops.aten.var_mean.correction(view_1200, [2], correction = 0, keepdim = True)
        getitem_322 = var_mean_95[0]
        getitem_323 = var_mean_95[1];  var_mean_95 = None
        add_469 = torch.ops.aten.add.Tensor(getitem_322, 1e-05);  getitem_322 = None
        rsqrt_95 = torch.ops.aten.rsqrt.default(add_469);  add_469 = None
        sub_139 = torch.ops.aten.sub.Tensor(view_1200, getitem_323);  getitem_323 = None
        mul_363 = torch.ops.aten.mul.Tensor(sub_139, rsqrt_95);  sub_139 = rsqrt_95 = None
        mul_364 = torch.ops.aten.mul.Tensor(mul_363, arg295_1);  mul_363 = arg295_1 = None
        add_470 = torch.ops.aten.add.Tensor(mul_364, arg296_1);  mul_364 = arg296_1 = None
        view_1201 = torch.ops.aten.view.default(add_470, [1568, 512]);  add_470 = None
        permute_451 = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_175 = torch.ops.aten.addmm.default(arg298_1, view_1201, permute_451);  arg298_1 = view_1201 = permute_451 = None
        view_1202 = torch.ops.aten.view.default(addmm_175, [8, 196, 2048]);  addmm_175 = None
        mul_365 = torch.ops.aten.mul.Tensor(view_1202, 0.5)
        mul_366 = torch.ops.aten.mul.Tensor(view_1202, 0.7071067811865476);  view_1202 = None
        erf_43 = torch.ops.aten.erf.default(mul_366);  mul_366 = None
        add_471 = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_367 = torch.ops.aten.mul.Tensor(mul_365, add_471);  mul_365 = add_471 = None
        view_1203 = torch.ops.aten.view.default(mul_367, [1568, 2048]);  mul_367 = None
        permute_452 = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_176 = torch.ops.aten.addmm.default(arg300_1, view_1203, permute_452);  arg300_1 = view_1203 = permute_452 = None
        view_1204 = torch.ops.aten.view.default(addmm_176, [8, 196, 512]);  addmm_176 = None
        add_472 = torch.ops.aten.add.Tensor(view_1200, view_1204);  view_1200 = view_1204 = None
        view_1205 = torch.ops.aten.view.default(add_472, [8, 14, 14, 512]);  add_472 = None
        var_mean_96 = torch.ops.aten.var_mean.correction(view_1205, [3], correction = 0, keepdim = True)
        getitem_324 = var_mean_96[0]
        getitem_325 = var_mean_96[1];  var_mean_96 = None
        add_473 = torch.ops.aten.add.Tensor(getitem_324, 1e-05);  getitem_324 = None
        rsqrt_96 = torch.ops.aten.rsqrt.default(add_473);  add_473 = None
        sub_140 = torch.ops.aten.sub.Tensor(view_1205, getitem_325);  getitem_325 = None
        mul_368 = torch.ops.aten.mul.Tensor(sub_140, rsqrt_96);  sub_140 = rsqrt_96 = None
        mul_369 = torch.ops.aten.mul.Tensor(mul_368, arg301_1);  mul_368 = arg301_1 = None
        add_474 = torch.ops.aten.add.Tensor(mul_369, arg302_1);  mul_369 = arg302_1 = None
        view_1206 = torch.ops.aten.view.default(add_474, [8, 2, 7, 2, 7, 512]);  add_474 = None
        permute_453 = torch.ops.aten.permute.default(view_1206, [0, 1, 3, 2, 4, 5]);  view_1206 = None
        clone_488 = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
        view_1207 = torch.ops.aten.view.default(clone_488, [-1, 7, 7, 512]);  clone_488 = None
        view_1208 = torch.ops.aten.view.default(view_1207, [-1, 49, 512]);  view_1207 = None
        view_1209 = torch.ops.aten.view.default(view_1208, [1568, 512]);  view_1208 = None
        permute_454 = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_177 = torch.ops.aten.addmm.default(arg304_1, view_1209, permute_454);  arg304_1 = view_1209 = permute_454 = None
        view_1210 = torch.ops.aten.view.default(addmm_177, [32, 49, 1536]);  addmm_177 = None
        view_1211 = torch.ops.aten.view.default(view_1210, [32, 49, 3, 16, -1]);  view_1210 = None
        permute_455 = torch.ops.aten.permute.default(view_1211, [2, 0, 3, 1, 4]);  view_1211 = None
        unbind_44 = torch.ops.aten.unbind.int(permute_455);  permute_455 = None
        getitem_326 = unbind_44[0]
        getitem_327 = unbind_44[1]
        getitem_328 = unbind_44[2];  unbind_44 = None
        mul_370 = torch.ops.aten.mul.Tensor(getitem_326, 0.1767766952966369);  getitem_326 = None
        permute_456 = torch.ops.aten.permute.default(getitem_327, [0, 1, 3, 2]);  getitem_327 = None
        expand_176 = torch.ops.aten.expand.default(mul_370, [32, 16, 49, 32]);  mul_370 = None
        clone_489 = torch.ops.aten.clone.default(expand_176, memory_format = torch.contiguous_format);  expand_176 = None
        view_1212 = torch.ops.aten.view.default(clone_489, [512, 49, 32]);  clone_489 = None
        expand_177 = torch.ops.aten.expand.default(permute_456, [32, 16, 32, 49]);  permute_456 = None
        clone_490 = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_1213 = torch.ops.aten.view.default(clone_490, [512, 32, 49]);  clone_490 = None
        bmm_88 = torch.ops.aten.bmm.default(view_1212, view_1213);  view_1212 = view_1213 = None
        view_1214 = torch.ops.aten.view.default(bmm_88, [32, 16, 49, 49]);  bmm_88 = None
        view_1215 = torch.ops.aten.view.default(arg306_1, [-1]);  arg306_1 = None
        index_128 = torch.ops.aten.index.Tensor(arg305_1, [view_1215]);  arg305_1 = view_1215 = None
        view_1216 = torch.ops.aten.view.default(index_128, [49, 49, -1]);  index_128 = None
        permute_457 = torch.ops.aten.permute.default(view_1216, [2, 0, 1]);  view_1216 = None
        clone_491 = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(clone_491, 0);  clone_491 = None
        add_475 = torch.ops.aten.add.Tensor(view_1214, unsqueeze_86);  view_1214 = unsqueeze_86 = None
        amax_44 = torch.ops.aten.amax.default(add_475, [-1], True)
        sub_141 = torch.ops.aten.sub.Tensor(add_475, amax_44);  add_475 = amax_44 = None
        exp_44 = torch.ops.aten.exp.default(sub_141);  sub_141 = None
        sum_45 = torch.ops.aten.sum.dim_IntList(exp_44, [-1], True)
        div_44 = torch.ops.aten.div.Tensor(exp_44, sum_45);  exp_44 = sum_45 = None
        expand_178 = torch.ops.aten.expand.default(div_44, [32, 16, 49, 49]);  div_44 = None
        view_1217 = torch.ops.aten.view.default(expand_178, [512, 49, 49]);  expand_178 = None
        expand_179 = torch.ops.aten.expand.default(getitem_328, [32, 16, 49, 32]);  getitem_328 = None
        clone_493 = torch.ops.aten.clone.default(expand_179, memory_format = torch.contiguous_format);  expand_179 = None
        view_1218 = torch.ops.aten.view.default(clone_493, [512, 49, 32]);  clone_493 = None
        bmm_89 = torch.ops.aten.bmm.default(view_1217, view_1218);  view_1217 = view_1218 = None
        view_1219 = torch.ops.aten.view.default(bmm_89, [32, 16, 49, 32]);  bmm_89 = None
        permute_458 = torch.ops.aten.permute.default(view_1219, [0, 2, 1, 3]);  view_1219 = None
        clone_494 = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
        view_1220 = torch.ops.aten.view.default(clone_494, [32, 49, 512]);  clone_494 = None
        view_1221 = torch.ops.aten.view.default(view_1220, [1568, 512]);  view_1220 = None
        permute_459 = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_178 = torch.ops.aten.addmm.default(arg308_1, view_1221, permute_459);  arg308_1 = view_1221 = permute_459 = None
        view_1222 = torch.ops.aten.view.default(addmm_178, [32, 49, 512]);  addmm_178 = None
        view_1223 = torch.ops.aten.view.default(view_1222, [-1, 7, 7, 512]);  view_1222 = None
        view_1224 = torch.ops.aten.view.default(view_1223, [-1, 2, 2, 7, 7, 512]);  view_1223 = None
        permute_460 = torch.ops.aten.permute.default(view_1224, [0, 1, 3, 2, 4, 5]);  view_1224 = None
        clone_496 = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
        view_1225 = torch.ops.aten.view.default(clone_496, [-1, 14, 14, 512]);  clone_496 = None
        add_476 = torch.ops.aten.add.Tensor(view_1205, view_1225);  view_1205 = view_1225 = None
        view_1226 = torch.ops.aten.view.default(add_476, [8, -1, 512]);  add_476 = None
        var_mean_97 = torch.ops.aten.var_mean.correction(view_1226, [2], correction = 0, keepdim = True)
        getitem_329 = var_mean_97[0]
        getitem_330 = var_mean_97[1];  var_mean_97 = None
        add_477 = torch.ops.aten.add.Tensor(getitem_329, 1e-05);  getitem_329 = None
        rsqrt_97 = torch.ops.aten.rsqrt.default(add_477);  add_477 = None
        sub_142 = torch.ops.aten.sub.Tensor(view_1226, getitem_330);  getitem_330 = None
        mul_371 = torch.ops.aten.mul.Tensor(sub_142, rsqrt_97);  sub_142 = rsqrt_97 = None
        mul_372 = torch.ops.aten.mul.Tensor(mul_371, arg309_1);  mul_371 = arg309_1 = None
        add_478 = torch.ops.aten.add.Tensor(mul_372, arg310_1);  mul_372 = arg310_1 = None
        view_1227 = torch.ops.aten.view.default(add_478, [1568, 512]);  add_478 = None
        permute_461 = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_179 = torch.ops.aten.addmm.default(arg312_1, view_1227, permute_461);  arg312_1 = view_1227 = permute_461 = None
        view_1228 = torch.ops.aten.view.default(addmm_179, [8, 196, 2048]);  addmm_179 = None
        mul_373 = torch.ops.aten.mul.Tensor(view_1228, 0.5)
        mul_374 = torch.ops.aten.mul.Tensor(view_1228, 0.7071067811865476);  view_1228 = None
        erf_44 = torch.ops.aten.erf.default(mul_374);  mul_374 = None
        add_479 = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_375 = torch.ops.aten.mul.Tensor(mul_373, add_479);  mul_373 = add_479 = None
        view_1229 = torch.ops.aten.view.default(mul_375, [1568, 2048]);  mul_375 = None
        permute_462 = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_180 = torch.ops.aten.addmm.default(arg314_1, view_1229, permute_462);  arg314_1 = view_1229 = permute_462 = None
        view_1230 = torch.ops.aten.view.default(addmm_180, [8, 196, 512]);  addmm_180 = None
        add_480 = torch.ops.aten.add.Tensor(view_1226, view_1230);  view_1226 = view_1230 = None
        view_1231 = torch.ops.aten.view.default(add_480, [8, 14, 14, 512]);  add_480 = None
        var_mean_98 = torch.ops.aten.var_mean.correction(view_1231, [3], correction = 0, keepdim = True)
        getitem_331 = var_mean_98[0]
        getitem_332 = var_mean_98[1];  var_mean_98 = None
        add_481 = torch.ops.aten.add.Tensor(getitem_331, 1e-05);  getitem_331 = None
        rsqrt_98 = torch.ops.aten.rsqrt.default(add_481);  add_481 = None
        sub_143 = torch.ops.aten.sub.Tensor(view_1231, getitem_332);  getitem_332 = None
        mul_376 = torch.ops.aten.mul.Tensor(sub_143, rsqrt_98);  sub_143 = rsqrt_98 = None
        mul_377 = torch.ops.aten.mul.Tensor(mul_376, arg315_1);  mul_376 = arg315_1 = None
        add_482 = torch.ops.aten.add.Tensor(mul_377, arg316_1);  mul_377 = arg316_1 = None
        iota_84 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_483 = torch.ops.aten.add.Tensor(iota_84, 3);  iota_84 = None
        fmod_84 = torch.ops.aten.fmod.Scalar(add_483, 14);  add_483 = None
        index_129 = torch.ops.aten.index.Tensor(add_482, [None, fmod_84]);  add_482 = fmod_84 = None
        iota_85 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_484 = torch.ops.aten.add.Tensor(iota_85, 3);  iota_85 = None
        fmod_85 = torch.ops.aten.fmod.Scalar(add_484, 14);  add_484 = None
        index_130 = torch.ops.aten.index.Tensor(index_129, [None, None, fmod_85]);  index_129 = fmod_85 = None
        view_1232 = torch.ops.aten.view.default(index_130, [8, 2, 7, 2, 7, 512]);  index_130 = None
        permute_463 = torch.ops.aten.permute.default(view_1232, [0, 1, 3, 2, 4, 5]);  view_1232 = None
        clone_499 = torch.ops.aten.clone.default(permute_463, memory_format = torch.contiguous_format);  permute_463 = None
        view_1233 = torch.ops.aten.view.default(clone_499, [-1, 7, 7, 512]);  clone_499 = None
        view_1234 = torch.ops.aten.view.default(view_1233, [-1, 49, 512]);  view_1233 = None
        view_1235 = torch.ops.aten.view.default(view_1234, [1568, 512]);  view_1234 = None
        permute_464 = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
        addmm_181 = torch.ops.aten.addmm.default(arg319_1, view_1235, permute_464);  arg319_1 = view_1235 = permute_464 = None
        view_1236 = torch.ops.aten.view.default(addmm_181, [32, 49, 1536]);  addmm_181 = None
        view_1237 = torch.ops.aten.view.default(view_1236, [32, 49, 3, 16, -1]);  view_1236 = None
        permute_465 = torch.ops.aten.permute.default(view_1237, [2, 0, 3, 1, 4]);  view_1237 = None
        unbind_45 = torch.ops.aten.unbind.int(permute_465);  permute_465 = None
        getitem_333 = unbind_45[0]
        getitem_334 = unbind_45[1]
        getitem_335 = unbind_45[2];  unbind_45 = None
        mul_378 = torch.ops.aten.mul.Tensor(getitem_333, 0.1767766952966369);  getitem_333 = None
        permute_466 = torch.ops.aten.permute.default(getitem_334, [0, 1, 3, 2]);  getitem_334 = None
        expand_180 = torch.ops.aten.expand.default(mul_378, [32, 16, 49, 32]);  mul_378 = None
        clone_500 = torch.ops.aten.clone.default(expand_180, memory_format = torch.contiguous_format);  expand_180 = None
        view_1238 = torch.ops.aten.view.default(clone_500, [512, 49, 32]);  clone_500 = None
        expand_181 = torch.ops.aten.expand.default(permute_466, [32, 16, 32, 49]);  permute_466 = None
        clone_501 = torch.ops.aten.clone.default(expand_181, memory_format = torch.contiguous_format);  expand_181 = None
        view_1239 = torch.ops.aten.view.default(clone_501, [512, 32, 49]);  clone_501 = None
        bmm_90 = torch.ops.aten.bmm.default(view_1238, view_1239);  view_1238 = view_1239 = None
        view_1240 = torch.ops.aten.view.default(bmm_90, [32, 16, 49, 49]);  bmm_90 = None
        view_1241 = torch.ops.aten.view.default(arg321_1, [-1]);  arg321_1 = None
        index_131 = torch.ops.aten.index.Tensor(arg320_1, [view_1241]);  arg320_1 = view_1241 = None
        view_1242 = torch.ops.aten.view.default(index_131, [49, 49, -1]);  index_131 = None
        permute_467 = torch.ops.aten.permute.default(view_1242, [2, 0, 1]);  view_1242 = None
        clone_502 = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(clone_502, 0);  clone_502 = None
        add_485 = torch.ops.aten.add.Tensor(view_1240, unsqueeze_87);  view_1240 = unsqueeze_87 = None
        view_1243 = torch.ops.aten.view.default(add_485, [-1, 4, 16, 49, 49]);  add_485 = None
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(arg317_1, 1);  arg317_1 = None
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(unsqueeze_88, 0);  unsqueeze_88 = None
        add_486 = torch.ops.aten.add.Tensor(view_1243, unsqueeze_89);  view_1243 = unsqueeze_89 = None
        view_1244 = torch.ops.aten.view.default(add_486, [-1, 16, 49, 49]);  add_486 = None
        amax_45 = torch.ops.aten.amax.default(view_1244, [-1], True)
        sub_144 = torch.ops.aten.sub.Tensor(view_1244, amax_45);  view_1244 = amax_45 = None
        exp_45 = torch.ops.aten.exp.default(sub_144);  sub_144 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(exp_45, [-1], True)
        div_45 = torch.ops.aten.div.Tensor(exp_45, sum_46);  exp_45 = sum_46 = None
        expand_182 = torch.ops.aten.expand.default(div_45, [32, 16, 49, 49]);  div_45 = None
        view_1245 = torch.ops.aten.view.default(expand_182, [512, 49, 49]);  expand_182 = None
        expand_183 = torch.ops.aten.expand.default(getitem_335, [32, 16, 49, 32]);  getitem_335 = None
        clone_504 = torch.ops.aten.clone.default(expand_183, memory_format = torch.contiguous_format);  expand_183 = None
        view_1246 = torch.ops.aten.view.default(clone_504, [512, 49, 32]);  clone_504 = None
        bmm_91 = torch.ops.aten.bmm.default(view_1245, view_1246);  view_1245 = view_1246 = None
        view_1247 = torch.ops.aten.view.default(bmm_91, [32, 16, 49, 32]);  bmm_91 = None
        permute_468 = torch.ops.aten.permute.default(view_1247, [0, 2, 1, 3]);  view_1247 = None
        clone_505 = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
        view_1248 = torch.ops.aten.view.default(clone_505, [32, 49, 512]);  clone_505 = None
        view_1249 = torch.ops.aten.view.default(view_1248, [1568, 512]);  view_1248 = None
        permute_469 = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
        addmm_182 = torch.ops.aten.addmm.default(arg323_1, view_1249, permute_469);  arg323_1 = view_1249 = permute_469 = None
        view_1250 = torch.ops.aten.view.default(addmm_182, [32, 49, 512]);  addmm_182 = None
        view_1251 = torch.ops.aten.view.default(view_1250, [-1, 7, 7, 512]);  view_1250 = None
        view_1252 = torch.ops.aten.view.default(view_1251, [-1, 2, 2, 7, 7, 512]);  view_1251 = None
        permute_470 = torch.ops.aten.permute.default(view_1252, [0, 1, 3, 2, 4, 5]);  view_1252 = None
        clone_507 = torch.ops.aten.clone.default(permute_470, memory_format = torch.contiguous_format);  permute_470 = None
        view_1253 = torch.ops.aten.view.default(clone_507, [-1, 14, 14, 512]);  clone_507 = None
        iota_86 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_487 = torch.ops.aten.add.Tensor(iota_86, 11);  iota_86 = None
        fmod_86 = torch.ops.aten.fmod.Scalar(add_487, 14);  add_487 = None
        index_132 = torch.ops.aten.index.Tensor(view_1253, [None, fmod_86]);  view_1253 = fmod_86 = None
        iota_87 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_488 = torch.ops.aten.add.Tensor(iota_87, 11);  iota_87 = None
        fmod_87 = torch.ops.aten.fmod.Scalar(add_488, 14);  add_488 = None
        index_133 = torch.ops.aten.index.Tensor(index_132, [None, None, fmod_87]);  index_132 = fmod_87 = None
        add_489 = torch.ops.aten.add.Tensor(view_1231, index_133);  view_1231 = index_133 = None
        view_1254 = torch.ops.aten.view.default(add_489, [8, -1, 512]);  add_489 = None
        var_mean_99 = torch.ops.aten.var_mean.correction(view_1254, [2], correction = 0, keepdim = True)
        getitem_336 = var_mean_99[0]
        getitem_337 = var_mean_99[1];  var_mean_99 = None
        add_490 = torch.ops.aten.add.Tensor(getitem_336, 1e-05);  getitem_336 = None
        rsqrt_99 = torch.ops.aten.rsqrt.default(add_490);  add_490 = None
        sub_145 = torch.ops.aten.sub.Tensor(view_1254, getitem_337);  getitem_337 = None
        mul_379 = torch.ops.aten.mul.Tensor(sub_145, rsqrt_99);  sub_145 = rsqrt_99 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_379, arg324_1);  mul_379 = arg324_1 = None
        add_491 = torch.ops.aten.add.Tensor(mul_380, arg325_1);  mul_380 = arg325_1 = None
        view_1255 = torch.ops.aten.view.default(add_491, [1568, 512]);  add_491 = None
        permute_471 = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        addmm_183 = torch.ops.aten.addmm.default(arg327_1, view_1255, permute_471);  arg327_1 = view_1255 = permute_471 = None
        view_1256 = torch.ops.aten.view.default(addmm_183, [8, 196, 2048]);  addmm_183 = None
        mul_381 = torch.ops.aten.mul.Tensor(view_1256, 0.5)
        mul_382 = torch.ops.aten.mul.Tensor(view_1256, 0.7071067811865476);  view_1256 = None
        erf_45 = torch.ops.aten.erf.default(mul_382);  mul_382 = None
        add_492 = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_383 = torch.ops.aten.mul.Tensor(mul_381, add_492);  mul_381 = add_492 = None
        view_1257 = torch.ops.aten.view.default(mul_383, [1568, 2048]);  mul_383 = None
        permute_472 = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_184 = torch.ops.aten.addmm.default(arg329_1, view_1257, permute_472);  arg329_1 = view_1257 = permute_472 = None
        view_1258 = torch.ops.aten.view.default(addmm_184, [8, 196, 512]);  addmm_184 = None
        add_493 = torch.ops.aten.add.Tensor(view_1254, view_1258);  view_1254 = view_1258 = None
        view_1259 = torch.ops.aten.view.default(add_493, [8, 14, 14, 512]);  add_493 = None
        view_1260 = torch.ops.aten.view.default(view_1259, [8, 7, 2, 7, 2, 512]);  view_1259 = None
        permute_473 = torch.ops.aten.permute.default(view_1260, [0, 1, 3, 4, 2, 5]);  view_1260 = None
        clone_510 = torch.ops.aten.clone.default(permute_473, memory_format = torch.contiguous_format);  permute_473 = None
        view_1261 = torch.ops.aten.view.default(clone_510, [8, 7, 7, 2048]);  clone_510 = None
        var_mean_100 = torch.ops.aten.var_mean.correction(view_1261, [3], correction = 0, keepdim = True)
        getitem_338 = var_mean_100[0]
        getitem_339 = var_mean_100[1];  var_mean_100 = None
        add_494 = torch.ops.aten.add.Tensor(getitem_338, 1e-05);  getitem_338 = None
        rsqrt_100 = torch.ops.aten.rsqrt.default(add_494);  add_494 = None
        sub_146 = torch.ops.aten.sub.Tensor(view_1261, getitem_339);  view_1261 = getitem_339 = None
        mul_384 = torch.ops.aten.mul.Tensor(sub_146, rsqrt_100);  sub_146 = rsqrt_100 = None
        mul_385 = torch.ops.aten.mul.Tensor(mul_384, arg330_1);  mul_384 = arg330_1 = None
        add_495 = torch.ops.aten.add.Tensor(mul_385, arg331_1);  mul_385 = arg331_1 = None
        permute_474 = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
        view_1262 = torch.ops.aten.view.default(add_495, [392, 2048]);  add_495 = None
        mm_5 = torch.ops.aten.mm.default(view_1262, permute_474);  view_1262 = permute_474 = None
        view_1263 = torch.ops.aten.view.default(mm_5, [8, 7, 7, 1024]);  mm_5 = None
        var_mean_101 = torch.ops.aten.var_mean.correction(view_1263, [3], correction = 0, keepdim = True)
        getitem_340 = var_mean_101[0]
        getitem_341 = var_mean_101[1];  var_mean_101 = None
        add_496 = torch.ops.aten.add.Tensor(getitem_340, 1e-05);  getitem_340 = None
        rsqrt_101 = torch.ops.aten.rsqrt.default(add_496);  add_496 = None
        sub_147 = torch.ops.aten.sub.Tensor(view_1263, getitem_341);  getitem_341 = None
        mul_386 = torch.ops.aten.mul.Tensor(sub_147, rsqrt_101);  sub_147 = rsqrt_101 = None
        mul_387 = torch.ops.aten.mul.Tensor(mul_386, arg333_1);  mul_386 = arg333_1 = None
        add_497 = torch.ops.aten.add.Tensor(mul_387, arg334_1);  mul_387 = arg334_1 = None
        view_1264 = torch.ops.aten.view.default(add_497, [8, 1, 7, 1, 7, 1024]);  add_497 = None
        permute_475 = torch.ops.aten.permute.default(view_1264, [0, 1, 3, 2, 4, 5]);  view_1264 = None
        view_1265 = torch.ops.aten.view.default(permute_475, [-1, 7, 7, 1024]);  permute_475 = None
        view_1266 = torch.ops.aten.view.default(view_1265, [-1, 49, 1024]);  view_1265 = None
        view_1267 = torch.ops.aten.view.default(view_1266, [392, 1024]);  view_1266 = None
        permute_476 = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_185 = torch.ops.aten.addmm.default(arg336_1, view_1267, permute_476);  arg336_1 = view_1267 = permute_476 = None
        view_1268 = torch.ops.aten.view.default(addmm_185, [8, 49, 3072]);  addmm_185 = None
        view_1269 = torch.ops.aten.view.default(view_1268, [8, 49, 3, 32, -1]);  view_1268 = None
        permute_477 = torch.ops.aten.permute.default(view_1269, [2, 0, 3, 1, 4]);  view_1269 = None
        unbind_46 = torch.ops.aten.unbind.int(permute_477);  permute_477 = None
        getitem_342 = unbind_46[0]
        getitem_343 = unbind_46[1]
        getitem_344 = unbind_46[2];  unbind_46 = None
        mul_388 = torch.ops.aten.mul.Tensor(getitem_342, 0.1767766952966369);  getitem_342 = None
        permute_478 = torch.ops.aten.permute.default(getitem_343, [0, 1, 3, 2]);  getitem_343 = None
        expand_184 = torch.ops.aten.expand.default(mul_388, [8, 32, 49, 32]);  mul_388 = None
        clone_511 = torch.ops.aten.clone.default(expand_184, memory_format = torch.contiguous_format);  expand_184 = None
        view_1270 = torch.ops.aten.view.default(clone_511, [256, 49, 32]);  clone_511 = None
        expand_185 = torch.ops.aten.expand.default(permute_478, [8, 32, 32, 49]);  permute_478 = None
        clone_512 = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_1271 = torch.ops.aten.view.default(clone_512, [256, 32, 49]);  clone_512 = None
        bmm_92 = torch.ops.aten.bmm.default(view_1270, view_1271);  view_1270 = view_1271 = None
        view_1272 = torch.ops.aten.view.default(bmm_92, [8, 32, 49, 49]);  bmm_92 = None
        view_1273 = torch.ops.aten.view.default(arg338_1, [-1]);  arg338_1 = None
        index_134 = torch.ops.aten.index.Tensor(arg337_1, [view_1273]);  arg337_1 = view_1273 = None
        view_1274 = torch.ops.aten.view.default(index_134, [49, 49, -1]);  index_134 = None
        permute_479 = torch.ops.aten.permute.default(view_1274, [2, 0, 1]);  view_1274 = None
        clone_513 = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(clone_513, 0);  clone_513 = None
        add_498 = torch.ops.aten.add.Tensor(view_1272, unsqueeze_90);  view_1272 = unsqueeze_90 = None
        amax_46 = torch.ops.aten.amax.default(add_498, [-1], True)
        sub_148 = torch.ops.aten.sub.Tensor(add_498, amax_46);  add_498 = amax_46 = None
        exp_46 = torch.ops.aten.exp.default(sub_148);  sub_148 = None
        sum_47 = torch.ops.aten.sum.dim_IntList(exp_46, [-1], True)
        div_46 = torch.ops.aten.div.Tensor(exp_46, sum_47);  exp_46 = sum_47 = None
        expand_186 = torch.ops.aten.expand.default(div_46, [8, 32, 49, 49]);  div_46 = None
        view_1275 = torch.ops.aten.view.default(expand_186, [256, 49, 49]);  expand_186 = None
        expand_187 = torch.ops.aten.expand.default(getitem_344, [8, 32, 49, 32]);  getitem_344 = None
        clone_515 = torch.ops.aten.clone.default(expand_187, memory_format = torch.contiguous_format);  expand_187 = None
        view_1276 = torch.ops.aten.view.default(clone_515, [256, 49, 32]);  clone_515 = None
        bmm_93 = torch.ops.aten.bmm.default(view_1275, view_1276);  view_1275 = view_1276 = None
        view_1277 = torch.ops.aten.view.default(bmm_93, [8, 32, 49, 32]);  bmm_93 = None
        permute_480 = torch.ops.aten.permute.default(view_1277, [0, 2, 1, 3]);  view_1277 = None
        clone_516 = torch.ops.aten.clone.default(permute_480, memory_format = torch.contiguous_format);  permute_480 = None
        view_1278 = torch.ops.aten.view.default(clone_516, [8, 49, 1024]);  clone_516 = None
        view_1279 = torch.ops.aten.view.default(view_1278, [392, 1024]);  view_1278 = None
        permute_481 = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        addmm_186 = torch.ops.aten.addmm.default(arg340_1, view_1279, permute_481);  arg340_1 = view_1279 = permute_481 = None
        view_1280 = torch.ops.aten.view.default(addmm_186, [8, 49, 1024]);  addmm_186 = None
        view_1281 = torch.ops.aten.view.default(view_1280, [-1, 7, 7, 1024]);  view_1280 = None
        view_1282 = torch.ops.aten.view.default(view_1281, [-1, 1, 1, 7, 7, 1024]);  view_1281 = None
        permute_482 = torch.ops.aten.permute.default(view_1282, [0, 1, 3, 2, 4, 5]);  view_1282 = None
        view_1283 = torch.ops.aten.view.default(permute_482, [-1, 7, 7, 1024]);  permute_482 = None
        add_499 = torch.ops.aten.add.Tensor(view_1263, view_1283);  view_1263 = view_1283 = None
        view_1284 = torch.ops.aten.view.default(add_499, [8, -1, 1024]);  add_499 = None
        var_mean_102 = torch.ops.aten.var_mean.correction(view_1284, [2], correction = 0, keepdim = True)
        getitem_345 = var_mean_102[0]
        getitem_346 = var_mean_102[1];  var_mean_102 = None
        add_500 = torch.ops.aten.add.Tensor(getitem_345, 1e-05);  getitem_345 = None
        rsqrt_102 = torch.ops.aten.rsqrt.default(add_500);  add_500 = None
        sub_149 = torch.ops.aten.sub.Tensor(view_1284, getitem_346);  getitem_346 = None
        mul_389 = torch.ops.aten.mul.Tensor(sub_149, rsqrt_102);  sub_149 = rsqrt_102 = None
        mul_390 = torch.ops.aten.mul.Tensor(mul_389, arg341_1);  mul_389 = arg341_1 = None
        add_501 = torch.ops.aten.add.Tensor(mul_390, arg342_1);  mul_390 = arg342_1 = None
        view_1285 = torch.ops.aten.view.default(add_501, [392, 1024]);  add_501 = None
        permute_483 = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_187 = torch.ops.aten.addmm.default(arg344_1, view_1285, permute_483);  arg344_1 = view_1285 = permute_483 = None
        view_1286 = torch.ops.aten.view.default(addmm_187, [8, 49, 4096]);  addmm_187 = None
        mul_391 = torch.ops.aten.mul.Tensor(view_1286, 0.5)
        mul_392 = torch.ops.aten.mul.Tensor(view_1286, 0.7071067811865476);  view_1286 = None
        erf_46 = torch.ops.aten.erf.default(mul_392);  mul_392 = None
        add_502 = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_393 = torch.ops.aten.mul.Tensor(mul_391, add_502);  mul_391 = add_502 = None
        view_1287 = torch.ops.aten.view.default(mul_393, [392, 4096]);  mul_393 = None
        permute_484 = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_188 = torch.ops.aten.addmm.default(arg346_1, view_1287, permute_484);  arg346_1 = view_1287 = permute_484 = None
        view_1288 = torch.ops.aten.view.default(addmm_188, [8, 49, 1024]);  addmm_188 = None
        add_503 = torch.ops.aten.add.Tensor(view_1284, view_1288);  view_1284 = view_1288 = None
        view_1289 = torch.ops.aten.view.default(add_503, [8, 7, 7, 1024]);  add_503 = None
        var_mean_103 = torch.ops.aten.var_mean.correction(view_1289, [3], correction = 0, keepdim = True)
        getitem_347 = var_mean_103[0]
        getitem_348 = var_mean_103[1];  var_mean_103 = None
        add_504 = torch.ops.aten.add.Tensor(getitem_347, 1e-05);  getitem_347 = None
        rsqrt_103 = torch.ops.aten.rsqrt.default(add_504);  add_504 = None
        sub_150 = torch.ops.aten.sub.Tensor(view_1289, getitem_348);  getitem_348 = None
        mul_394 = torch.ops.aten.mul.Tensor(sub_150, rsqrt_103);  sub_150 = rsqrt_103 = None
        mul_395 = torch.ops.aten.mul.Tensor(mul_394, arg347_1);  mul_394 = arg347_1 = None
        add_505 = torch.ops.aten.add.Tensor(mul_395, arg348_1);  mul_395 = arg348_1 = None
        view_1290 = torch.ops.aten.view.default(add_505, [8, 1, 7, 1, 7, 1024]);  add_505 = None
        permute_485 = torch.ops.aten.permute.default(view_1290, [0, 1, 3, 2, 4, 5]);  view_1290 = None
        view_1291 = torch.ops.aten.view.default(permute_485, [-1, 7, 7, 1024]);  permute_485 = None
        view_1292 = torch.ops.aten.view.default(view_1291, [-1, 49, 1024]);  view_1291 = None
        view_1293 = torch.ops.aten.view.default(view_1292, [392, 1024]);  view_1292 = None
        permute_486 = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        addmm_189 = torch.ops.aten.addmm.default(arg350_1, view_1293, permute_486);  arg350_1 = view_1293 = permute_486 = None
        view_1294 = torch.ops.aten.view.default(addmm_189, [8, 49, 3072]);  addmm_189 = None
        view_1295 = torch.ops.aten.view.default(view_1294, [8, 49, 3, 32, -1]);  view_1294 = None
        permute_487 = torch.ops.aten.permute.default(view_1295, [2, 0, 3, 1, 4]);  view_1295 = None
        unbind_47 = torch.ops.aten.unbind.int(permute_487);  permute_487 = None
        getitem_349 = unbind_47[0]
        getitem_350 = unbind_47[1]
        getitem_351 = unbind_47[2];  unbind_47 = None
        mul_396 = torch.ops.aten.mul.Tensor(getitem_349, 0.1767766952966369);  getitem_349 = None
        permute_488 = torch.ops.aten.permute.default(getitem_350, [0, 1, 3, 2]);  getitem_350 = None
        expand_188 = torch.ops.aten.expand.default(mul_396, [8, 32, 49, 32]);  mul_396 = None
        clone_520 = torch.ops.aten.clone.default(expand_188, memory_format = torch.contiguous_format);  expand_188 = None
        view_1296 = torch.ops.aten.view.default(clone_520, [256, 49, 32]);  clone_520 = None
        expand_189 = torch.ops.aten.expand.default(permute_488, [8, 32, 32, 49]);  permute_488 = None
        clone_521 = torch.ops.aten.clone.default(expand_189, memory_format = torch.contiguous_format);  expand_189 = None
        view_1297 = torch.ops.aten.view.default(clone_521, [256, 32, 49]);  clone_521 = None
        bmm_94 = torch.ops.aten.bmm.default(view_1296, view_1297);  view_1296 = view_1297 = None
        view_1298 = torch.ops.aten.view.default(bmm_94, [8, 32, 49, 49]);  bmm_94 = None
        view_1299 = torch.ops.aten.view.default(arg352_1, [-1]);  arg352_1 = None
        index_135 = torch.ops.aten.index.Tensor(arg351_1, [view_1299]);  arg351_1 = view_1299 = None
        view_1300 = torch.ops.aten.view.default(index_135, [49, 49, -1]);  index_135 = None
        permute_489 = torch.ops.aten.permute.default(view_1300, [2, 0, 1]);  view_1300 = None
        clone_522 = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(clone_522, 0);  clone_522 = None
        add_506 = torch.ops.aten.add.Tensor(view_1298, unsqueeze_91);  view_1298 = unsqueeze_91 = None
        amax_47 = torch.ops.aten.amax.default(add_506, [-1], True)
        sub_151 = torch.ops.aten.sub.Tensor(add_506, amax_47);  add_506 = amax_47 = None
        exp_47 = torch.ops.aten.exp.default(sub_151);  sub_151 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(exp_47, [-1], True)
        div_47 = torch.ops.aten.div.Tensor(exp_47, sum_48);  exp_47 = sum_48 = None
        expand_190 = torch.ops.aten.expand.default(div_47, [8, 32, 49, 49]);  div_47 = None
        view_1301 = torch.ops.aten.view.default(expand_190, [256, 49, 49]);  expand_190 = None
        expand_191 = torch.ops.aten.expand.default(getitem_351, [8, 32, 49, 32]);  getitem_351 = None
        clone_524 = torch.ops.aten.clone.default(expand_191, memory_format = torch.contiguous_format);  expand_191 = None
        view_1302 = torch.ops.aten.view.default(clone_524, [256, 49, 32]);  clone_524 = None
        bmm_95 = torch.ops.aten.bmm.default(view_1301, view_1302);  view_1301 = view_1302 = None
        view_1303 = torch.ops.aten.view.default(bmm_95, [8, 32, 49, 32]);  bmm_95 = None
        permute_490 = torch.ops.aten.permute.default(view_1303, [0, 2, 1, 3]);  view_1303 = None
        clone_525 = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
        view_1304 = torch.ops.aten.view.default(clone_525, [8, 49, 1024]);  clone_525 = None
        view_1305 = torch.ops.aten.view.default(view_1304, [392, 1024]);  view_1304 = None
        permute_491 = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_190 = torch.ops.aten.addmm.default(arg354_1, view_1305, permute_491);  arg354_1 = view_1305 = permute_491 = None
        view_1306 = torch.ops.aten.view.default(addmm_190, [8, 49, 1024]);  addmm_190 = None
        view_1307 = torch.ops.aten.view.default(view_1306, [-1, 7, 7, 1024]);  view_1306 = None
        view_1308 = torch.ops.aten.view.default(view_1307, [-1, 1, 1, 7, 7, 1024]);  view_1307 = None
        permute_492 = torch.ops.aten.permute.default(view_1308, [0, 1, 3, 2, 4, 5]);  view_1308 = None
        view_1309 = torch.ops.aten.view.default(permute_492, [-1, 7, 7, 1024]);  permute_492 = None
        add_507 = torch.ops.aten.add.Tensor(view_1289, view_1309);  view_1289 = view_1309 = None
        view_1310 = torch.ops.aten.view.default(add_507, [8, -1, 1024]);  add_507 = None
        var_mean_104 = torch.ops.aten.var_mean.correction(view_1310, [2], correction = 0, keepdim = True)
        getitem_352 = var_mean_104[0]
        getitem_353 = var_mean_104[1];  var_mean_104 = None
        add_508 = torch.ops.aten.add.Tensor(getitem_352, 1e-05);  getitem_352 = None
        rsqrt_104 = torch.ops.aten.rsqrt.default(add_508);  add_508 = None
        sub_152 = torch.ops.aten.sub.Tensor(view_1310, getitem_353);  getitem_353 = None
        mul_397 = torch.ops.aten.mul.Tensor(sub_152, rsqrt_104);  sub_152 = rsqrt_104 = None
        mul_398 = torch.ops.aten.mul.Tensor(mul_397, arg355_1);  mul_397 = arg355_1 = None
        add_509 = torch.ops.aten.add.Tensor(mul_398, arg356_1);  mul_398 = arg356_1 = None
        view_1311 = torch.ops.aten.view.default(add_509, [392, 1024]);  add_509 = None
        permute_493 = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_191 = torch.ops.aten.addmm.default(arg358_1, view_1311, permute_493);  arg358_1 = view_1311 = permute_493 = None
        view_1312 = torch.ops.aten.view.default(addmm_191, [8, 49, 4096]);  addmm_191 = None
        mul_399 = torch.ops.aten.mul.Tensor(view_1312, 0.5)
        mul_400 = torch.ops.aten.mul.Tensor(view_1312, 0.7071067811865476);  view_1312 = None
        erf_47 = torch.ops.aten.erf.default(mul_400);  mul_400 = None
        add_510 = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_401 = torch.ops.aten.mul.Tensor(mul_399, add_510);  mul_399 = add_510 = None
        view_1313 = torch.ops.aten.view.default(mul_401, [392, 4096]);  mul_401 = None
        permute_494 = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_192 = torch.ops.aten.addmm.default(arg360_1, view_1313, permute_494);  arg360_1 = view_1313 = permute_494 = None
        view_1314 = torch.ops.aten.view.default(addmm_192, [8, 49, 1024]);  addmm_192 = None
        add_511 = torch.ops.aten.add.Tensor(view_1310, view_1314);  view_1310 = view_1314 = None
        view_1315 = torch.ops.aten.view.default(add_511, [8, 7, 7, 1024]);  add_511 = None
        var_mean_105 = torch.ops.aten.var_mean.correction(view_1315, [3], correction = 0, keepdim = True)
        getitem_354 = var_mean_105[0]
        getitem_355 = var_mean_105[1];  var_mean_105 = None
        add_512 = torch.ops.aten.add.Tensor(getitem_354, 1e-05);  getitem_354 = None
        rsqrt_105 = torch.ops.aten.rsqrt.default(add_512);  add_512 = None
        sub_153 = torch.ops.aten.sub.Tensor(view_1315, getitem_355);  view_1315 = getitem_355 = None
        mul_402 = torch.ops.aten.mul.Tensor(sub_153, rsqrt_105);  sub_153 = rsqrt_105 = None
        mul_403 = torch.ops.aten.mul.Tensor(mul_402, arg361_1);  mul_402 = arg361_1 = None
        add_513 = torch.ops.aten.add.Tensor(mul_403, arg362_1);  mul_403 = arg362_1 = None
        mean_1 = torch.ops.aten.mean.dim(add_513, [1, 2]);  add_513 = None
        permute_495 = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_193 = torch.ops.aten.addmm.default(arg364_1, mean_1, permute_495);  arg364_1 = mean_1 = permute_495 = None
        return (addmm_193,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 224, 224), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128, 3, 4, 4), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf4, (128,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf5, (128,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf6, (128,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf7, (384, 128), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf8, (384,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 2704, device=device(type='cuda', index=0))
    reader.tensor(buf9, (169, 4), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf10, (49, 49), dtype=torch.int64, is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128, 128), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf13, (128,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf14, (128,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf15, (512, 128), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf16, (512,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf17, (128, 512), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf18, (128,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf19, (128,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 614656, device=device(type='cuda', index=0))
    reader.tensor(buf21, (64, 49, 49), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf22, (384, 128), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf23, (384,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 2704, device=device(type='cuda', index=0))
    reader.tensor(buf24, (169, 4), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf25, (49, 49), dtype=torch.int64, is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf26, (128, 128), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf27, (128,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf28, (128,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf29, (128,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf30, (512, 128), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf31, (512,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf32, (128, 512), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf33, (128,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf34, (512,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf35, (512,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf36, (256, 512), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf37, (256,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf38, (256,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768, 256), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 5408, device=device(type='cuda', index=0))
    reader.tensor(buf41, (169, 8), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf42, (49, 49), dtype=torch.int64, is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256, 256), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf44, (256,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf45, (256,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf46, (256,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf47, (1024, 256), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf48, (1024,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf49, (256, 1024), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf50, (256,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf51, (256,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf52, (256,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf53, (16, 49, 49), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768, 256), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 5408, device=device(type='cuda', index=0))
    reader.tensor(buf56, (169, 8), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf57, (49, 49), dtype=torch.int64, is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf58, (256, 256), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf59, (256,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf60, (256,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf61, (256,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf62, (1024, 256), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf63, (1024,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf64, (256, 1024), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf65, (256,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf66, (1024,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf67, (1024,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf68, (512, 1024), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf69, (512,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf70, (512,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf71, (1536, 512), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf72, (1536,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf73, (169, 16), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf74, (49, 49), dtype=torch.int64, is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512, 512), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf76, (512,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf77, (512,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf78, (512,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf79, (2048, 512), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf80, (2048,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf81, (512, 2048), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf82, (512,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf83, (512,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf84, (512,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 38416, device=device(type='cuda', index=0))
    reader.tensor(buf85, (4, 49, 49), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf86, (1536, 512), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf87, (1536,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf88, (169, 16), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf89, (49, 49), dtype=torch.int64, is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf90, (512, 512), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf91, (512,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf92, (512,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf93, (512,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf94, (2048, 512), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf95, (2048,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf96, (512, 2048), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf97, (512,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf98, (512,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf99, (512,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1536, 512), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf101, (1536,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf102, (169, 16), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf103, (49, 49), dtype=torch.int64, is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf104, (512, 512), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf105, (512,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf106, (512,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf107, (512,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf108, (2048, 512), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf109, (2048,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf110, (512, 2048), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf111, (512,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf112, (512,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf113, (512,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 38416, device=device(type='cuda', index=0))
    reader.tensor(buf114, (4, 49, 49), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf115, (1536, 512), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1536,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf117, (169, 16), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf118, (49, 49), dtype=torch.int64, is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf119, (512, 512), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf120, (512,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf121, (512,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf122, (512,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf123, (2048, 512), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf124, (2048,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf125, (512, 2048), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf126, (512,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf127, (512,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf128, (512,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf129, (1536, 512), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf130, (1536,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf131, (169, 16), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf132, (49, 49), dtype=torch.int64, is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf133, (512, 512), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf134, (512,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf135, (512,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf136, (512,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf137, (2048, 512), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf138, (2048,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf139, (512, 2048), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf140, (512,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf141, (512,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf142, (512,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 38416, device=device(type='cuda', index=0))
    reader.tensor(buf143, (4, 49, 49), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf144, (1536, 512), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf145, (1536,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf146, (169, 16), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf147, (49, 49), dtype=torch.int64, is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf148, (512, 512), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf149, (512,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf150, (512,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf151, (512,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf152, (2048, 512), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf153, (2048,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf154, (512, 2048), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf155, (512,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf156, (512,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf157, (512,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf158, (1536, 512), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf159, (1536,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf160, (169, 16), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf161, (49, 49), dtype=torch.int64, is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf162, (512, 512), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf163, (512,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf164, (512,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf165, (512,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf166, (2048, 512), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf167, (2048,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf168, (512, 2048), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf169, (512,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf170, (512,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf171, (512,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 38416, device=device(type='cuda', index=0))
    reader.tensor(buf172, (4, 49, 49), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf173, (1536, 512), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf174, (1536,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf175, (169, 16), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf176, (49, 49), dtype=torch.int64, is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf177, (512, 512), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf178, (512,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf179, (512,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf180, (512,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf181, (2048, 512), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf182, (2048,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf183, (512, 2048), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf184, (512,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf185, (512,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf186, (512,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf187, (1536, 512), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf188, (1536,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf189, (169, 16), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf190, (49, 49), dtype=torch.int64, is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf191, (512, 512), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf192, (512,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf193, (512,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf194, (512,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf195, (2048, 512), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf196, (2048,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf197, (512, 2048), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf198, (512,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf199, (512,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf200, (512,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 38416, device=device(type='cuda', index=0))
    reader.tensor(buf201, (4, 49, 49), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf202, (1536, 512), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf203, (1536,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf204, (169, 16), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf205, (49, 49), dtype=torch.int64, is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf206, (512, 512), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf207, (512,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf208, (512,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf209, (512,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf210, (2048, 512), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf211, (2048,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf212, (512, 2048), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf213, (512,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf214, (512,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf215, (512,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf216, (1536, 512), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf217, (1536,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf218, (169, 16), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf219, (49, 49), dtype=torch.int64, is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf220, (512, 512), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf221, (512,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf222, (512,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf223, (512,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf224, (2048, 512), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf225, (2048,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf226, (512, 2048), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf227, (512,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf228, (512,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf229, (512,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 38416, device=device(type='cuda', index=0))
    reader.tensor(buf230, (4, 49, 49), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf231, (1536, 512), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf232, (1536,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf233, (169, 16), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf234, (49, 49), dtype=torch.int64, is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf235, (512, 512), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf236, (512,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf237, (512,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf238, (512,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf239, (2048, 512), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf240, (2048,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf241, (512, 2048), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf242, (512,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf243, (512,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf244, (512,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf245, (1536, 512), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf246, (1536,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf247, (169, 16), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf248, (49, 49), dtype=torch.int64, is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf249, (512, 512), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf250, (512,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf251, (512,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf252, (512,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf253, (2048, 512), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf254, (2048,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf255, (512, 2048), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf256, (512,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf257, (512,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf258, (512,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 38416, device=device(type='cuda', index=0))
    reader.tensor(buf259, (4, 49, 49), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf260, (1536, 512), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf261, (1536,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf262, (169, 16), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf263, (49, 49), dtype=torch.int64, is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf264, (512, 512), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf265, (512,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf266, (512,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf267, (512,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf268, (2048, 512), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf269, (2048,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf270, (512, 2048), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf271, (512,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf272, (512,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf273, (512,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf274, (1536, 512), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf275, (1536,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf276, (169, 16), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf277, (49, 49), dtype=torch.int64, is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf278, (512, 512), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf279, (512,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf280, (512,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf281, (512,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf282, (2048, 512), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf283, (2048,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf284, (512, 2048), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf285, (512,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf286, (512,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf287, (512,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 38416, device=device(type='cuda', index=0))
    reader.tensor(buf288, (4, 49, 49), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf289, (1536, 512), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf290, (1536,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf291, (169, 16), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf292, (49, 49), dtype=torch.int64, is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf293, (512, 512), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf294, (512,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf295, (512,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf296, (512,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf297, (2048, 512), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf298, (2048,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf299, (512, 2048), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf300, (512,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf301, (512,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf302, (512,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf303, (1536, 512), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf304, (1536,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf305, (169, 16), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf306, (49, 49), dtype=torch.int64, is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf307, (512, 512), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf308, (512,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf309, (512,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf310, (512,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf311, (2048, 512), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf312, (2048,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf313, (512, 2048), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf314, (512,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf315, (512,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf316, (512,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 38416, device=device(type='cuda', index=0))
    reader.tensor(buf317, (4, 49, 49), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf318, (1536, 512), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf319, (1536,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 10816, device=device(type='cuda', index=0))
    reader.tensor(buf320, (169, 16), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf321, (49, 49), dtype=torch.int64, is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf322, (512, 512), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf323, (512,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf324, (512,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf325, (512,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf326, (2048, 512), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf327, (2048,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf328, (512, 2048), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf329, (512,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf330, (2048,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf331, (2048,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf332, (1024, 2048), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf333, (1024,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf334, (1024,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 12582912, device=device(type='cuda', index=0))
    reader.tensor(buf335, (3072, 1024), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf336, (3072,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 21632, device=device(type='cuda', index=0))
    reader.tensor(buf337, (169, 32), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf338, (49, 49), dtype=torch.int64, is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf339, (1024, 1024), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf340, (1024,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf341, (1024,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf342, (1024,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf343, (4096, 1024), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf344, (4096,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf345, (1024, 4096), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf346, (1024,), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf347, (1024,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf348, (1024,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 12582912, device=device(type='cuda', index=0))
    reader.tensor(buf349, (3072, 1024), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf350, (3072,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 21632, device=device(type='cuda', index=0))
    reader.tensor(buf351, (169, 32), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 19208, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf352, (49, 49), dtype=torch.int64, is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf353, (1024, 1024), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf354, (1024,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf355, (1024,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf356, (1024,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf357, (4096, 1024), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf358, (4096,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf359, (1024, 4096), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf360, (1024,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf361, (1024,), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf362, (1024,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 4096000, device=device(type='cuda', index=0))
    reader.tensor(buf363, (1000, 1024), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf364, (1000,), is_leaf=True)  # arg364_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)