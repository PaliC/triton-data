
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1):
        convolution_5 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_171 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_3 = torch.ops.aten.sqrt.default(add_171);  add_171 = None
        reciprocal_3 = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_158 = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(mul_158, -1);  mul_158 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        sub_50 = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_73);  convolution_5 = unsqueeze_73 = None
        mul_159 = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_75);  sub_50 = unsqueeze_75 = None
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_160 = torch.ops.aten.mul.Tensor(mul_159, unsqueeze_77);  mul_159 = unsqueeze_77 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_172 = torch.ops.aten.add.Tensor(mul_160, unsqueeze_79);  mul_160 = unsqueeze_79 = None
        relu_3 = torch.ops.aten.relu.default(add_172);  add_172 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_3, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_3 = arg6_1 = None
        add_173 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_4 = torch.ops.aten.sqrt.default(add_173);  add_173 = None
        reciprocal_4 = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_161 = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(mul_161, -1);  mul_161 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        sub_51 = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_81);  convolution_6 = unsqueeze_81 = None
        mul_162 = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_83);  sub_51 = unsqueeze_83 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_163 = torch.ops.aten.mul.Tensor(mul_162, unsqueeze_85);  mul_162 = unsqueeze_85 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_174 = torch.ops.aten.add.Tensor(mul_163, unsqueeze_87);  mul_163 = unsqueeze_87 = None
        relu_4 = torch.ops.aten.relu.default(add_174);  add_174 = None
        convolution_7 = torch.ops.aten.convolution.default(relu_4, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg11_1 = None
        add_175 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_5 = torch.ops.aten.sqrt.default(add_175);  add_175 = None
        reciprocal_5 = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_164 = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(mul_164, -1);  mul_164 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        sub_52 = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_89);  convolution_7 = unsqueeze_89 = None
        mul_165 = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_91);  sub_52 = unsqueeze_91 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_166 = torch.ops.aten.mul.Tensor(mul_165, unsqueeze_93);  mul_165 = unsqueeze_93 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_176 = torch.ops.aten.add.Tensor(mul_166, unsqueeze_95);  mul_166 = unsqueeze_95 = None
        relu_5 = torch.ops.aten.relu.default(add_176);  add_176 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_5, arg16_1, arg17_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  relu_5 = arg16_1 = arg17_1 = None
        permute_161 = torch.ops.aten.permute.default(convolution_8, [0, 2, 3, 1]);  convolution_8 = None
        clone_129 = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format)
        var_mean_41 = torch.ops.aten.var_mean.correction(clone_129, [3], correction = 0, keepdim = True)
        getitem_186 = var_mean_41[0]
        getitem_187 = var_mean_41[1];  var_mean_41 = None
        add_177 = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        sub_53 = torch.ops.aten.sub.Tensor(clone_129, getitem_187);  clone_129 = getitem_187 = None
        mul_167 = torch.ops.aten.mul.Tensor(sub_53, rsqrt_41);  sub_53 = rsqrt_41 = None
        mul_168 = torch.ops.aten.mul.Tensor(mul_167, arg18_1);  mul_167 = arg18_1 = None
        add_178 = torch.ops.aten.add.Tensor(mul_168, arg19_1);  mul_168 = arg19_1 = None
        permute_162 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        view_253 = torch.ops.aten.view.default(add_178, [6272, 192])
        mm_27 = torch.ops.aten.mm.default(view_253, permute_162);  view_253 = permute_162 = None
        view_254 = torch.ops.aten.view.default(mm_27, [8, 28, 28, 192]);  mm_27 = None
        permute_163 = torch.ops.aten.permute.default(view_254, [0, 3, 1, 2]);  view_254 = None
        iota_32 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(iota_32, 0);  iota_32 = None
        iota_33 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(iota_33, -1);  iota_33 = None
        add_179 = torch.ops.aten.add.Tensor(unsqueeze_96, unsqueeze_97);  unsqueeze_96 = unsqueeze_97 = None
        iota_34 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(iota_34, 0);  iota_34 = None
        iota_35 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(iota_35, -1);  iota_35 = None
        add_180 = torch.ops.aten.add.Tensor(unsqueeze_98, unsqueeze_99);  unsqueeze_98 = unsqueeze_99 = None
        constant_pad_nd_8 = torch.ops.aten.constant_pad_nd.default(permute_163, [1, 1, 1, 1], 0.0);  permute_163 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(add_179, -1);  add_179 = None
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        index_4 = torch.ops.aten.index.Tensor(constant_pad_nd_8, [None, None, unsqueeze_101, add_180]);  constant_pad_nd_8 = unsqueeze_101 = add_180 = None
        permute_164 = torch.ops.aten.permute.default(index_4, [0, 1, 2, 4, 3, 5]);  index_4 = None
        clone_130 = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
        view_255 = torch.ops.aten.view.default(clone_130, [8, 1728, 196]);  clone_130 = None
        view_256 = torch.ops.aten.view.default(view_255, [8, 6, 32, 9, 196]);  view_255 = None
        permute_165 = torch.ops.aten.permute.default(view_256, [0, 1, 4, 3, 2]);  view_256 = None
        permute_166 = torch.ops.aten.permute.default(add_178, [0, 3, 1, 2]);  add_178 = None
        avg_pool2d_4 = torch.ops.aten.avg_pool2d.default(permute_166, [2, 2], [2, 2], [0, 0], True);  permute_166 = None
        permute_167 = torch.ops.aten.permute.default(avg_pool2d_4, [0, 2, 3, 1]);  avg_pool2d_4 = None
        view_257 = torch.ops.aten.view.default(permute_167, [1568, 192]);  permute_167 = None
        permute_168 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg22_1, view_257, permute_168);  arg22_1 = view_257 = permute_168 = None
        view_258 = torch.ops.aten.view.default(addmm_61, [8, 14, 14, 486]);  addmm_61 = None
        view_259 = torch.ops.aten.view.default(view_258, [8, 196, 6, 9, 9]);  view_258 = None
        permute_169 = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3, 4]);  view_259 = None
        mul_169 = torch.ops.aten.mul.Tensor(permute_169, 0.1767766952966369);  permute_169 = None
        clone_131 = torch.ops.aten.clone.default(mul_169, memory_format = torch.contiguous_format);  mul_169 = None
        amax_6 = torch.ops.aten.amax.default(clone_131, [-1], True)
        sub_54 = torch.ops.aten.sub.Tensor(clone_131, amax_6);  clone_131 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_54);  sub_54 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        expand_17 = torch.ops.aten.expand.default(div_6, [8, 6, 196, 9, 9]);  div_6 = None
        view_260 = torch.ops.aten.view.default(expand_17, [9408, 9, 9]);  expand_17 = None
        expand_18 = torch.ops.aten.expand.default(permute_165, [8, 6, 196, 9, 32]);  permute_165 = None
        clone_133 = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_261 = torch.ops.aten.view.default(clone_133, [9408, 9, 32]);  clone_133 = None
        bmm_8 = torch.ops.aten.bmm.default(view_260, view_261);  view_260 = view_261 = None
        view_262 = torch.ops.aten.view.default(bmm_8, [8, 6, 196, 9, 32]);  bmm_8 = None
        permute_170 = torch.ops.aten.permute.default(view_262, [0, 1, 4, 3, 2]);  view_262 = None
        clone_134 = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_263 = torch.ops.aten.view.default(clone_134, [8, 1728, 196]);  clone_134 = None
        view_264 = torch.ops.aten.view.default(view_263, [8, 192, 3, 3, 14, 14]);  view_263 = None
        permute_171 = torch.ops.aten.permute.default(view_264, [0, 1, 2, 4, 3, 5]);  view_264 = None
        iota_36 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(iota_36, 0);  iota_36 = None
        iota_37 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(iota_37, -1);  iota_37 = None
        add_181 = torch.ops.aten.add.Tensor(unsqueeze_102, unsqueeze_103);  unsqueeze_102 = unsqueeze_103 = None
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(add_181, -1);  add_181 = None
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        iota_38 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(iota_38, 0);  iota_38 = None
        iota_39 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(iota_39, -1);  iota_39 = None
        add_182 = torch.ops.aten.add.Tensor(unsqueeze_106, unsqueeze_107);  unsqueeze_106 = unsqueeze_107 = None
        full_default = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_4 = torch.ops.aten.index_put.default(full_default, [None, None, unsqueeze_105, add_182], permute_171, True);  full_default = unsqueeze_105 = add_182 = permute_171 = None
        constant_pad_nd_9 = torch.ops.aten.constant_pad_nd.default(index_put_4, [-1, -1, -1, -1], 0.0);  index_put_4 = None
        permute_172 = torch.ops.aten.permute.default(constant_pad_nd_9, [0, 2, 3, 1]);  constant_pad_nd_9 = None
        permute_173 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        clone_135 = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_265 = torch.ops.aten.view.default(clone_135, [6272, 192]);  clone_135 = None
        mm_28 = torch.ops.aten.mm.default(view_265, permute_173);  view_265 = permute_173 = None
        view_266 = torch.ops.aten.view.default(mm_28, [8, 28, 28, 192]);  mm_28 = None
        add_183 = torch.ops.aten.add.Tensor(view_266, arg24_1);  view_266 = arg24_1 = None
        add_184 = torch.ops.aten.add.Tensor(permute_161, add_183);  permute_161 = add_183 = None
        clone_137 = torch.ops.aten.clone.default(add_184, memory_format = torch.contiguous_format)
        var_mean_42 = torch.ops.aten.var_mean.correction(clone_137, [3], correction = 0, keepdim = True)
        getitem_188 = var_mean_42[0]
        getitem_189 = var_mean_42[1];  var_mean_42 = None
        add_185 = torch.ops.aten.add.Tensor(getitem_188, 1e-05);  getitem_188 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        sub_55 = torch.ops.aten.sub.Tensor(clone_137, getitem_189);  clone_137 = getitem_189 = None
        mul_170 = torch.ops.aten.mul.Tensor(sub_55, rsqrt_42);  sub_55 = rsqrt_42 = None
        mul_171 = torch.ops.aten.mul.Tensor(mul_170, arg25_1);  mul_170 = arg25_1 = None
        add_186 = torch.ops.aten.add.Tensor(mul_171, arg26_1);  mul_171 = arg26_1 = None
        view_267 = torch.ops.aten.view.default(add_186, [6272, 192]);  add_186 = None
        permute_174 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg28_1, view_267, permute_174);  arg28_1 = view_267 = permute_174 = None
        view_268 = torch.ops.aten.view.default(addmm_62, [8, 28, 28, 576]);  addmm_62 = None
        mul_172 = torch.ops.aten.mul.Tensor(view_268, 0.5)
        mul_173 = torch.ops.aten.mul.Tensor(view_268, 0.7071067811865476);  view_268 = None
        erf_20 = torch.ops.aten.erf.default(mul_173);  mul_173 = None
        add_187 = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_174 = torch.ops.aten.mul.Tensor(mul_172, add_187);  mul_172 = add_187 = None
        view_269 = torch.ops.aten.view.default(mul_174, [6272, 576]);  mul_174 = None
        permute_175 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg30_1, view_269, permute_175);  arg30_1 = view_269 = permute_175 = None
        view_270 = torch.ops.aten.view.default(addmm_63, [8, 28, 28, 192]);  addmm_63 = None
        add_188 = torch.ops.aten.add.Tensor(add_184, view_270);  add_184 = view_270 = None
        clone_140 = torch.ops.aten.clone.default(add_188, memory_format = torch.contiguous_format)
        var_mean_43 = torch.ops.aten.var_mean.correction(clone_140, [3], correction = 0, keepdim = True)
        getitem_190 = var_mean_43[0]
        getitem_191 = var_mean_43[1];  var_mean_43 = None
        add_189 = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
        sub_56 = torch.ops.aten.sub.Tensor(clone_140, getitem_191);  clone_140 = getitem_191 = None
        mul_175 = torch.ops.aten.mul.Tensor(sub_56, rsqrt_43);  sub_56 = rsqrt_43 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_175, arg31_1);  mul_175 = arg31_1 = None
        add_190 = torch.ops.aten.add.Tensor(mul_176, arg32_1);  mul_176 = arg32_1 = None
        permute_176 = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        view_271 = torch.ops.aten.view.default(add_190, [6272, 192])
        mm_29 = torch.ops.aten.mm.default(view_271, permute_176);  view_271 = permute_176 = None
        view_272 = torch.ops.aten.view.default(mm_29, [8, 28, 28, 192]);  mm_29 = None
        permute_177 = torch.ops.aten.permute.default(view_272, [0, 3, 1, 2]);  view_272 = None
        iota_40 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(iota_40, 0);  iota_40 = None
        iota_41 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(iota_41, -1);  iota_41 = None
        add_191 = torch.ops.aten.add.Tensor(unsqueeze_108, unsqueeze_109);  unsqueeze_108 = unsqueeze_109 = None
        iota_42 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(iota_42, 0);  iota_42 = None
        iota_43 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(iota_43, -1);  iota_43 = None
        add_192 = torch.ops.aten.add.Tensor(unsqueeze_110, unsqueeze_111);  unsqueeze_110 = unsqueeze_111 = None
        constant_pad_nd_10 = torch.ops.aten.constant_pad_nd.default(permute_177, [1, 1, 1, 1], 0.0);  permute_177 = None
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(add_191, -1);  add_191 = None
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        index_5 = torch.ops.aten.index.Tensor(constant_pad_nd_10, [None, None, unsqueeze_113, add_192]);  constant_pad_nd_10 = unsqueeze_113 = add_192 = None
        permute_178 = torch.ops.aten.permute.default(index_5, [0, 1, 2, 4, 3, 5]);  index_5 = None
        clone_141 = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
        view_273 = torch.ops.aten.view.default(clone_141, [8, 1728, 196]);  clone_141 = None
        view_274 = torch.ops.aten.view.default(view_273, [8, 6, 32, 9, 196]);  view_273 = None
        permute_179 = torch.ops.aten.permute.default(view_274, [0, 1, 4, 3, 2]);  view_274 = None
        permute_180 = torch.ops.aten.permute.default(add_190, [0, 3, 1, 2]);  add_190 = None
        avg_pool2d_5 = torch.ops.aten.avg_pool2d.default(permute_180, [2, 2], [2, 2], [0, 0], True);  permute_180 = None
        permute_181 = torch.ops.aten.permute.default(avg_pool2d_5, [0, 2, 3, 1]);  avg_pool2d_5 = None
        view_275 = torch.ops.aten.view.default(permute_181, [1568, 192]);  permute_181 = None
        permute_182 = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg35_1, view_275, permute_182);  arg35_1 = view_275 = permute_182 = None
        view_276 = torch.ops.aten.view.default(addmm_64, [8, 14, 14, 486]);  addmm_64 = None
        view_277 = torch.ops.aten.view.default(view_276, [8, 196, 6, 9, 9]);  view_276 = None
        permute_183 = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3, 4]);  view_277 = None
        mul_177 = torch.ops.aten.mul.Tensor(permute_183, 0.1767766952966369);  permute_183 = None
        clone_142 = torch.ops.aten.clone.default(mul_177, memory_format = torch.contiguous_format);  mul_177 = None
        amax_7 = torch.ops.aten.amax.default(clone_142, [-1], True)
        sub_57 = torch.ops.aten.sub.Tensor(clone_142, amax_7);  clone_142 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_57);  sub_57 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        expand_19 = torch.ops.aten.expand.default(div_7, [8, 6, 196, 9, 9]);  div_7 = None
        view_278 = torch.ops.aten.view.default(expand_19, [9408, 9, 9]);  expand_19 = None
        expand_20 = torch.ops.aten.expand.default(permute_179, [8, 6, 196, 9, 32]);  permute_179 = None
        clone_144 = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_279 = torch.ops.aten.view.default(clone_144, [9408, 9, 32]);  clone_144 = None
        bmm_9 = torch.ops.aten.bmm.default(view_278, view_279);  view_278 = view_279 = None
        view_280 = torch.ops.aten.view.default(bmm_9, [8, 6, 196, 9, 32]);  bmm_9 = None
        permute_184 = torch.ops.aten.permute.default(view_280, [0, 1, 4, 3, 2]);  view_280 = None
        clone_145 = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_281 = torch.ops.aten.view.default(clone_145, [8, 1728, 196]);  clone_145 = None
        view_282 = torch.ops.aten.view.default(view_281, [8, 192, 3, 3, 14, 14]);  view_281 = None
        permute_185 = torch.ops.aten.permute.default(view_282, [0, 1, 2, 4, 3, 5]);  view_282 = None
        iota_44 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_114 = torch.ops.aten.unsqueeze.default(iota_44, 0);  iota_44 = None
        iota_45 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(iota_45, -1);  iota_45 = None
        add_193 = torch.ops.aten.add.Tensor(unsqueeze_114, unsqueeze_115);  unsqueeze_114 = unsqueeze_115 = None
        unsqueeze_116 = torch.ops.aten.unsqueeze.default(add_193, -1);  add_193 = None
        unsqueeze_117 = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        iota_46 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_118 = torch.ops.aten.unsqueeze.default(iota_46, 0);  iota_46 = None
        iota_47 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_119 = torch.ops.aten.unsqueeze.default(iota_47, -1);  iota_47 = None
        add_194 = torch.ops.aten.add.Tensor(unsqueeze_118, unsqueeze_119);  unsqueeze_118 = unsqueeze_119 = None
        full_default_1 = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_5 = torch.ops.aten.index_put.default(full_default_1, [None, None, unsqueeze_117, add_194], permute_185, True);  full_default_1 = unsqueeze_117 = add_194 = permute_185 = None
        constant_pad_nd_11 = torch.ops.aten.constant_pad_nd.default(index_put_5, [-1, -1, -1, -1], 0.0);  index_put_5 = None
        permute_186 = torch.ops.aten.permute.default(constant_pad_nd_11, [0, 2, 3, 1]);  constant_pad_nd_11 = None
        permute_187 = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        clone_146 = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
        view_283 = torch.ops.aten.view.default(clone_146, [6272, 192]);  clone_146 = None
        mm_30 = torch.ops.aten.mm.default(view_283, permute_187);  view_283 = permute_187 = None
        view_284 = torch.ops.aten.view.default(mm_30, [8, 28, 28, 192]);  mm_30 = None
        add_195 = torch.ops.aten.add.Tensor(view_284, arg37_1);  view_284 = arg37_1 = None
        add_196 = torch.ops.aten.add.Tensor(add_188, add_195);  add_188 = add_195 = None
        clone_148 = torch.ops.aten.clone.default(add_196, memory_format = torch.contiguous_format)
        var_mean_44 = torch.ops.aten.var_mean.correction(clone_148, [3], correction = 0, keepdim = True)
        getitem_192 = var_mean_44[0]
        getitem_193 = var_mean_44[1];  var_mean_44 = None
        add_197 = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_58 = torch.ops.aten.sub.Tensor(clone_148, getitem_193);  clone_148 = getitem_193 = None
        mul_178 = torch.ops.aten.mul.Tensor(sub_58, rsqrt_44);  sub_58 = rsqrt_44 = None
        mul_179 = torch.ops.aten.mul.Tensor(mul_178, arg38_1);  mul_178 = arg38_1 = None
        add_198 = torch.ops.aten.add.Tensor(mul_179, arg39_1);  mul_179 = arg39_1 = None
        view_285 = torch.ops.aten.view.default(add_198, [6272, 192]);  add_198 = None
        permute_188 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg41_1, view_285, permute_188);  arg41_1 = view_285 = permute_188 = None
        view_286 = torch.ops.aten.view.default(addmm_65, [8, 28, 28, 576]);  addmm_65 = None
        mul_180 = torch.ops.aten.mul.Tensor(view_286, 0.5)
        mul_181 = torch.ops.aten.mul.Tensor(view_286, 0.7071067811865476);  view_286 = None
        erf_21 = torch.ops.aten.erf.default(mul_181);  mul_181 = None
        add_199 = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_182 = torch.ops.aten.mul.Tensor(mul_180, add_199);  mul_180 = add_199 = None
        view_287 = torch.ops.aten.view.default(mul_182, [6272, 576]);  mul_182 = None
        permute_189 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg43_1, view_287, permute_189);  arg43_1 = view_287 = permute_189 = None
        view_288 = torch.ops.aten.view.default(addmm_66, [8, 28, 28, 192]);  addmm_66 = None
        add_200 = torch.ops.aten.add.Tensor(add_196, view_288);  add_196 = view_288 = None
        clone_151 = torch.ops.aten.clone.default(add_200, memory_format = torch.contiguous_format)
        var_mean_45 = torch.ops.aten.var_mean.correction(clone_151, [3], correction = 0, keepdim = True)
        getitem_194 = var_mean_45[0]
        getitem_195 = var_mean_45[1];  var_mean_45 = None
        add_201 = torch.ops.aten.add.Tensor(getitem_194, 1e-05);  getitem_194 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        sub_59 = torch.ops.aten.sub.Tensor(clone_151, getitem_195);  clone_151 = getitem_195 = None
        mul_183 = torch.ops.aten.mul.Tensor(sub_59, rsqrt_45);  sub_59 = rsqrt_45 = None
        mul_184 = torch.ops.aten.mul.Tensor(mul_183, arg44_1);  mul_183 = arg44_1 = None
        add_202 = torch.ops.aten.add.Tensor(mul_184, arg45_1);  mul_184 = arg45_1 = None
        permute_190 = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        view_289 = torch.ops.aten.view.default(add_202, [6272, 192])
        mm_31 = torch.ops.aten.mm.default(view_289, permute_190);  view_289 = permute_190 = None
        view_290 = torch.ops.aten.view.default(mm_31, [8, 28, 28, 192]);  mm_31 = None
        permute_191 = torch.ops.aten.permute.default(view_290, [0, 3, 1, 2]);  view_290 = None
        iota_48 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_120 = torch.ops.aten.unsqueeze.default(iota_48, 0);  iota_48 = None
        iota_49 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_121 = torch.ops.aten.unsqueeze.default(iota_49, -1);  iota_49 = None
        add_203 = torch.ops.aten.add.Tensor(unsqueeze_120, unsqueeze_121);  unsqueeze_120 = unsqueeze_121 = None
        iota_50 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(iota_50, 0);  iota_50 = None
        iota_51 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(iota_51, -1);  iota_51 = None
        add_204 = torch.ops.aten.add.Tensor(unsqueeze_122, unsqueeze_123);  unsqueeze_122 = unsqueeze_123 = None
        constant_pad_nd_12 = torch.ops.aten.constant_pad_nd.default(permute_191, [1, 1, 1, 1], 0.0);  permute_191 = None
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(add_203, -1);  add_203 = None
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        index_6 = torch.ops.aten.index.Tensor(constant_pad_nd_12, [None, None, unsqueeze_125, add_204]);  constant_pad_nd_12 = unsqueeze_125 = add_204 = None
        permute_192 = torch.ops.aten.permute.default(index_6, [0, 1, 2, 4, 3, 5]);  index_6 = None
        clone_152 = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_291 = torch.ops.aten.view.default(clone_152, [8, 1728, 196]);  clone_152 = None
        view_292 = torch.ops.aten.view.default(view_291, [8, 6, 32, 9, 196]);  view_291 = None
        permute_193 = torch.ops.aten.permute.default(view_292, [0, 1, 4, 3, 2]);  view_292 = None
        permute_194 = torch.ops.aten.permute.default(add_202, [0, 3, 1, 2]);  add_202 = None
        avg_pool2d_6 = torch.ops.aten.avg_pool2d.default(permute_194, [2, 2], [2, 2], [0, 0], True);  permute_194 = None
        permute_195 = torch.ops.aten.permute.default(avg_pool2d_6, [0, 2, 3, 1]);  avg_pool2d_6 = None
        view_293 = torch.ops.aten.view.default(permute_195, [1568, 192]);  permute_195 = None
        permute_196 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg48_1, view_293, permute_196);  arg48_1 = view_293 = permute_196 = None
        view_294 = torch.ops.aten.view.default(addmm_67, [8, 14, 14, 486]);  addmm_67 = None
        view_295 = torch.ops.aten.view.default(view_294, [8, 196, 6, 9, 9]);  view_294 = None
        permute_197 = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3, 4]);  view_295 = None
        mul_185 = torch.ops.aten.mul.Tensor(permute_197, 0.1767766952966369);  permute_197 = None
        clone_153 = torch.ops.aten.clone.default(mul_185, memory_format = torch.contiguous_format);  mul_185 = None
        amax_8 = torch.ops.aten.amax.default(clone_153, [-1], True)
        sub_60 = torch.ops.aten.sub.Tensor(clone_153, amax_8);  clone_153 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        expand_21 = torch.ops.aten.expand.default(div_8, [8, 6, 196, 9, 9]);  div_8 = None
        view_296 = torch.ops.aten.view.default(expand_21, [9408, 9, 9]);  expand_21 = None
        expand_22 = torch.ops.aten.expand.default(permute_193, [8, 6, 196, 9, 32]);  permute_193 = None
        clone_155 = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_297 = torch.ops.aten.view.default(clone_155, [9408, 9, 32]);  clone_155 = None
        bmm_10 = torch.ops.aten.bmm.default(view_296, view_297);  view_296 = view_297 = None
        view_298 = torch.ops.aten.view.default(bmm_10, [8, 6, 196, 9, 32]);  bmm_10 = None
        permute_198 = torch.ops.aten.permute.default(view_298, [0, 1, 4, 3, 2]);  view_298 = None
        clone_156 = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
        view_299 = torch.ops.aten.view.default(clone_156, [8, 1728, 196]);  clone_156 = None
        view_300 = torch.ops.aten.view.default(view_299, [8, 192, 3, 3, 14, 14]);  view_299 = None
        permute_199 = torch.ops.aten.permute.default(view_300, [0, 1, 2, 4, 3, 5]);  view_300 = None
        iota_52 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(iota_52, 0);  iota_52 = None
        iota_53 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(iota_53, -1);  iota_53 = None
        add_205 = torch.ops.aten.add.Tensor(unsqueeze_126, unsqueeze_127);  unsqueeze_126 = unsqueeze_127 = None
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(add_205, -1);  add_205 = None
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        iota_54 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(iota_54, 0);  iota_54 = None
        iota_55 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(iota_55, -1);  iota_55 = None
        add_206 = torch.ops.aten.add.Tensor(unsqueeze_130, unsqueeze_131);  unsqueeze_130 = unsqueeze_131 = None
        full_default_2 = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_6 = torch.ops.aten.index_put.default(full_default_2, [None, None, unsqueeze_129, add_206], permute_199, True);  full_default_2 = unsqueeze_129 = add_206 = permute_199 = None
        constant_pad_nd_13 = torch.ops.aten.constant_pad_nd.default(index_put_6, [-1, -1, -1, -1], 0.0);  index_put_6 = None
        permute_200 = torch.ops.aten.permute.default(constant_pad_nd_13, [0, 2, 3, 1]);  constant_pad_nd_13 = None
        permute_201 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        clone_157 = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
        view_301 = torch.ops.aten.view.default(clone_157, [6272, 192]);  clone_157 = None
        mm_32 = torch.ops.aten.mm.default(view_301, permute_201);  view_301 = permute_201 = None
        view_302 = torch.ops.aten.view.default(mm_32, [8, 28, 28, 192]);  mm_32 = None
        add_207 = torch.ops.aten.add.Tensor(view_302, arg50_1);  view_302 = arg50_1 = None
        add_208 = torch.ops.aten.add.Tensor(add_200, add_207);  add_200 = add_207 = None
        clone_159 = torch.ops.aten.clone.default(add_208, memory_format = torch.contiguous_format)
        var_mean_46 = torch.ops.aten.var_mean.correction(clone_159, [3], correction = 0, keepdim = True)
        getitem_196 = var_mean_46[0]
        getitem_197 = var_mean_46[1];  var_mean_46 = None
        add_209 = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
        sub_61 = torch.ops.aten.sub.Tensor(clone_159, getitem_197);  clone_159 = getitem_197 = None
        mul_186 = torch.ops.aten.mul.Tensor(sub_61, rsqrt_46);  sub_61 = rsqrt_46 = None
        mul_187 = torch.ops.aten.mul.Tensor(mul_186, arg51_1);  mul_186 = arg51_1 = None
        add_210 = torch.ops.aten.add.Tensor(mul_187, arg52_1);  mul_187 = arg52_1 = None
        view_303 = torch.ops.aten.view.default(add_210, [6272, 192]);  add_210 = None
        permute_202 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg54_1, view_303, permute_202);  arg54_1 = view_303 = permute_202 = None
        view_304 = torch.ops.aten.view.default(addmm_68, [8, 28, 28, 576]);  addmm_68 = None
        mul_188 = torch.ops.aten.mul.Tensor(view_304, 0.5)
        mul_189 = torch.ops.aten.mul.Tensor(view_304, 0.7071067811865476);  view_304 = None
        erf_22 = torch.ops.aten.erf.default(mul_189);  mul_189 = None
        add_211 = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_190 = torch.ops.aten.mul.Tensor(mul_188, add_211);  mul_188 = add_211 = None
        view_305 = torch.ops.aten.view.default(mul_190, [6272, 576]);  mul_190 = None
        permute_203 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg56_1, view_305, permute_203);  arg56_1 = view_305 = permute_203 = None
        view_306 = torch.ops.aten.view.default(addmm_69, [8, 28, 28, 192]);  addmm_69 = None
        add_212 = torch.ops.aten.add.Tensor(add_208, view_306);  add_208 = view_306 = None
        clone_162 = torch.ops.aten.clone.default(add_212, memory_format = torch.contiguous_format)
        var_mean_47 = torch.ops.aten.var_mean.correction(clone_162, [3], correction = 0, keepdim = True)
        getitem_198 = var_mean_47[0]
        getitem_199 = var_mean_47[1];  var_mean_47 = None
        add_213 = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        sub_62 = torch.ops.aten.sub.Tensor(clone_162, getitem_199);  clone_162 = getitem_199 = None
        mul_191 = torch.ops.aten.mul.Tensor(sub_62, rsqrt_47);  sub_62 = rsqrt_47 = None
        mul_192 = torch.ops.aten.mul.Tensor(mul_191, arg57_1);  mul_191 = arg57_1 = None
        add_214 = torch.ops.aten.add.Tensor(mul_192, arg58_1);  mul_192 = arg58_1 = None
        permute_204 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        view_307 = torch.ops.aten.view.default(add_214, [6272, 192])
        mm_33 = torch.ops.aten.mm.default(view_307, permute_204);  view_307 = permute_204 = None
        view_308 = torch.ops.aten.view.default(mm_33, [8, 28, 28, 192]);  mm_33 = None
        permute_205 = torch.ops.aten.permute.default(view_308, [0, 3, 1, 2]);  view_308 = None
        iota_56 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(iota_56, 0);  iota_56 = None
        iota_57 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(iota_57, -1);  iota_57 = None
        add_215 = torch.ops.aten.add.Tensor(unsqueeze_132, unsqueeze_133);  unsqueeze_132 = unsqueeze_133 = None
        iota_58 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(iota_58, 0);  iota_58 = None
        iota_59 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(iota_59, -1);  iota_59 = None
        add_216 = torch.ops.aten.add.Tensor(unsqueeze_134, unsqueeze_135);  unsqueeze_134 = unsqueeze_135 = None
        constant_pad_nd_14 = torch.ops.aten.constant_pad_nd.default(permute_205, [1, 1, 1, 1], 0.0);  permute_205 = None
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(add_215, -1);  add_215 = None
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        index_7 = torch.ops.aten.index.Tensor(constant_pad_nd_14, [None, None, unsqueeze_137, add_216]);  constant_pad_nd_14 = unsqueeze_137 = add_216 = None
        permute_206 = torch.ops.aten.permute.default(index_7, [0, 1, 2, 4, 3, 5]);  index_7 = None
        clone_163 = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
        view_309 = torch.ops.aten.view.default(clone_163, [8, 1728, 196]);  clone_163 = None
        view_310 = torch.ops.aten.view.default(view_309, [8, 6, 32, 9, 196]);  view_309 = None
        permute_207 = torch.ops.aten.permute.default(view_310, [0, 1, 4, 3, 2]);  view_310 = None
        permute_208 = torch.ops.aten.permute.default(add_214, [0, 3, 1, 2]);  add_214 = None
        avg_pool2d_7 = torch.ops.aten.avg_pool2d.default(permute_208, [2, 2], [2, 2], [0, 0], True);  permute_208 = None
        permute_209 = torch.ops.aten.permute.default(avg_pool2d_7, [0, 2, 3, 1]);  avg_pool2d_7 = None
        view_311 = torch.ops.aten.view.default(permute_209, [1568, 192]);  permute_209 = None
        permute_210 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg61_1, view_311, permute_210);  arg61_1 = view_311 = permute_210 = None
        view_312 = torch.ops.aten.view.default(addmm_70, [8, 14, 14, 486]);  addmm_70 = None
        view_313 = torch.ops.aten.view.default(view_312, [8, 196, 6, 9, 9]);  view_312 = None
        permute_211 = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3, 4]);  view_313 = None
        mul_193 = torch.ops.aten.mul.Tensor(permute_211, 0.1767766952966369);  permute_211 = None
        clone_164 = torch.ops.aten.clone.default(mul_193, memory_format = torch.contiguous_format);  mul_193 = None
        amax_9 = torch.ops.aten.amax.default(clone_164, [-1], True)
        sub_63 = torch.ops.aten.sub.Tensor(clone_164, amax_9);  clone_164 = amax_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_63);  sub_63 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        expand_23 = torch.ops.aten.expand.default(div_9, [8, 6, 196, 9, 9]);  div_9 = None
        view_314 = torch.ops.aten.view.default(expand_23, [9408, 9, 9]);  expand_23 = None
        expand_24 = torch.ops.aten.expand.default(permute_207, [8, 6, 196, 9, 32]);  permute_207 = None
        clone_166 = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_315 = torch.ops.aten.view.default(clone_166, [9408, 9, 32]);  clone_166 = None
        bmm_11 = torch.ops.aten.bmm.default(view_314, view_315);  view_314 = view_315 = None
        view_316 = torch.ops.aten.view.default(bmm_11, [8, 6, 196, 9, 32]);  bmm_11 = None
        permute_212 = torch.ops.aten.permute.default(view_316, [0, 1, 4, 3, 2]);  view_316 = None
        clone_167 = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_317 = torch.ops.aten.view.default(clone_167, [8, 1728, 196]);  clone_167 = None
        view_318 = torch.ops.aten.view.default(view_317, [8, 192, 3, 3, 14, 14]);  view_317 = None
        permute_213 = torch.ops.aten.permute.default(view_318, [0, 1, 2, 4, 3, 5]);  view_318 = None
        iota_60 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(iota_60, 0);  iota_60 = None
        iota_61 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(iota_61, -1);  iota_61 = None
        add_217 = torch.ops.aten.add.Tensor(unsqueeze_138, unsqueeze_139);  unsqueeze_138 = unsqueeze_139 = None
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(add_217, -1);  add_217 = None
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        iota_62 = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(iota_62, 0);  iota_62 = None
        iota_63 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(iota_63, -1);  iota_63 = None
        add_218 = torch.ops.aten.add.Tensor(unsqueeze_142, unsqueeze_143);  unsqueeze_142 = unsqueeze_143 = None
        full_default_3 = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_7 = torch.ops.aten.index_put.default(full_default_3, [None, None, unsqueeze_141, add_218], permute_213, True);  full_default_3 = unsqueeze_141 = add_218 = permute_213 = None
        constant_pad_nd_15 = torch.ops.aten.constant_pad_nd.default(index_put_7, [-1, -1, -1, -1], 0.0);  index_put_7 = None
        permute_214 = torch.ops.aten.permute.default(constant_pad_nd_15, [0, 2, 3, 1]);  constant_pad_nd_15 = None
        permute_215 = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        clone_168 = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_319 = torch.ops.aten.view.default(clone_168, [6272, 192]);  clone_168 = None
        mm_34 = torch.ops.aten.mm.default(view_319, permute_215);  view_319 = permute_215 = None
        view_320 = torch.ops.aten.view.default(mm_34, [8, 28, 28, 192]);  mm_34 = None
        add_219 = torch.ops.aten.add.Tensor(view_320, arg63_1);  view_320 = arg63_1 = None
        add_220 = torch.ops.aten.add.Tensor(add_212, add_219);  add_212 = add_219 = None
        clone_170 = torch.ops.aten.clone.default(add_220, memory_format = torch.contiguous_format)
        var_mean_48 = torch.ops.aten.var_mean.correction(clone_170, [3], correction = 0, keepdim = True)
        getitem_200 = var_mean_48[0]
        getitem_201 = var_mean_48[1];  var_mean_48 = None
        add_221 = torch.ops.aten.add.Tensor(getitem_200, 1e-05);  getitem_200 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
        sub_64 = torch.ops.aten.sub.Tensor(clone_170, getitem_201);  clone_170 = getitem_201 = None
        mul_194 = torch.ops.aten.mul.Tensor(sub_64, rsqrt_48);  sub_64 = rsqrt_48 = None
        mul_195 = torch.ops.aten.mul.Tensor(mul_194, arg64_1);  mul_194 = arg64_1 = None
        add_222 = torch.ops.aten.add.Tensor(mul_195, arg65_1);  mul_195 = arg65_1 = None
        view_321 = torch.ops.aten.view.default(add_222, [6272, 192]);  add_222 = None
        permute_216 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg67_1, view_321, permute_216);  arg67_1 = view_321 = permute_216 = None
        view_322 = torch.ops.aten.view.default(addmm_71, [8, 28, 28, 576]);  addmm_71 = None
        mul_196 = torch.ops.aten.mul.Tensor(view_322, 0.5)
        mul_197 = torch.ops.aten.mul.Tensor(view_322, 0.7071067811865476);  view_322 = None
        erf_23 = torch.ops.aten.erf.default(mul_197);  mul_197 = None
        add_223 = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_198 = torch.ops.aten.mul.Tensor(mul_196, add_223);  mul_196 = add_223 = None
        view_323 = torch.ops.aten.view.default(mul_198, [6272, 576]);  mul_198 = None
        permute_217 = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg69_1, view_323, permute_217);  arg69_1 = view_323 = permute_217 = None
        view_324 = torch.ops.aten.view.default(addmm_72, [8, 28, 28, 192]);  addmm_72 = None
        add_224 = torch.ops.aten.add.Tensor(add_220, view_324);  add_220 = view_324 = None
        permute_218 = torch.ops.aten.permute.default(add_224, [0, 3, 1, 2]);  add_224 = None
        convolution_9 = torch.ops.aten.convolution.default(permute_218, arg70_1, arg71_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_218 = arg70_1 = arg71_1 = None
        permute_219 = torch.ops.aten.permute.default(convolution_9, [0, 2, 3, 1]);  convolution_9 = None
        add_225 = torch.ops.aten.add.Tensor(permute_219, arg72_1);  permute_219 = arg72_1 = None
        clone_174 = torch.ops.aten.clone.default(add_225, memory_format = torch.contiguous_format)
        var_mean_49 = torch.ops.aten.var_mean.correction(clone_174, [3], correction = 0, keepdim = True)
        getitem_202 = var_mean_49[0]
        getitem_203 = var_mean_49[1];  var_mean_49 = None
        add_226 = torch.ops.aten.add.Tensor(getitem_202, 1e-05);  getitem_202 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        sub_65 = torch.ops.aten.sub.Tensor(clone_174, getitem_203);  clone_174 = getitem_203 = None
        mul_199 = torch.ops.aten.mul.Tensor(sub_65, rsqrt_49);  sub_65 = rsqrt_49 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, arg73_1);  mul_199 = arg73_1 = None
        add_227 = torch.ops.aten.add.Tensor(mul_200, arg74_1);  mul_200 = arg74_1 = None
        permute_220 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        view_325 = torch.ops.aten.view.default(add_227, [1568, 384]);  add_227 = None
        mm_35 = torch.ops.aten.mm.default(view_325, permute_220);  view_325 = permute_220 = None
        view_326 = torch.ops.aten.view.default(mm_35, [8, 14, 14, 1152]);  mm_35 = None
        view_327 = torch.ops.aten.view.default(view_326, [8, 196, 3, 12, 32]);  view_326 = None
        permute_221 = torch.ops.aten.permute.default(view_327, [2, 0, 3, 1, 4]);  view_327 = None
        unbind_16 = torch.ops.aten.unbind.int(permute_221);  permute_221 = None
        getitem_204 = unbind_16[0]
        getitem_205 = unbind_16[1]
        getitem_206 = unbind_16[2];  unbind_16 = None
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_204, getitem_205, getitem_206, None, False);  getitem_204 = getitem_205 = getitem_206 = None
        getitem_207 = _scaled_dot_product_efficient_attention_14[0];  _scaled_dot_product_efficient_attention_14 = None
        permute_222 = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None
        view_328 = torch.ops.aten.view.default(permute_222, [8, 14, 14, 384]);  permute_222 = None
        view_329 = torch.ops.aten.view.default(view_328, [1568, 384]);  view_328 = None
        permute_223 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg77_1, view_329, permute_223);  arg77_1 = view_329 = permute_223 = None
        view_330 = torch.ops.aten.view.default(addmm_73, [8, 14, 14, 384]);  addmm_73 = None
        add_228 = torch.ops.aten.add.Tensor(add_225, view_330);  add_225 = view_330 = None
        clone_176 = torch.ops.aten.clone.default(add_228, memory_format = torch.contiguous_format)
        var_mean_50 = torch.ops.aten.var_mean.correction(clone_176, [3], correction = 0, keepdim = True)
        getitem_211 = var_mean_50[0]
        getitem_212 = var_mean_50[1];  var_mean_50 = None
        add_229 = torch.ops.aten.add.Tensor(getitem_211, 1e-05);  getitem_211 = None
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_66 = torch.ops.aten.sub.Tensor(clone_176, getitem_212);  clone_176 = getitem_212 = None
        mul_201 = torch.ops.aten.mul.Tensor(sub_66, rsqrt_50);  sub_66 = rsqrt_50 = None
        mul_202 = torch.ops.aten.mul.Tensor(mul_201, arg78_1);  mul_201 = arg78_1 = None
        add_230 = torch.ops.aten.add.Tensor(mul_202, arg79_1);  mul_202 = arg79_1 = None
        view_331 = torch.ops.aten.view.default(add_230, [1568, 384]);  add_230 = None
        permute_224 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg81_1, view_331, permute_224);  arg81_1 = view_331 = permute_224 = None
        view_332 = torch.ops.aten.view.default(addmm_74, [8, 14, 14, 1152]);  addmm_74 = None
        mul_203 = torch.ops.aten.mul.Tensor(view_332, 0.5)
        mul_204 = torch.ops.aten.mul.Tensor(view_332, 0.7071067811865476);  view_332 = None
        erf_24 = torch.ops.aten.erf.default(mul_204);  mul_204 = None
        add_231 = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_205 = torch.ops.aten.mul.Tensor(mul_203, add_231);  mul_203 = add_231 = None
        view_333 = torch.ops.aten.view.default(mul_205, [1568, 1152]);  mul_205 = None
        permute_225 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg83_1, view_333, permute_225);  arg83_1 = view_333 = permute_225 = None
        view_334 = torch.ops.aten.view.default(addmm_75, [8, 14, 14, 384]);  addmm_75 = None
        add_232 = torch.ops.aten.add.Tensor(add_228, view_334);  add_228 = view_334 = None
        clone_179 = torch.ops.aten.clone.default(add_232, memory_format = torch.contiguous_format)
        var_mean_51 = torch.ops.aten.var_mean.correction(clone_179, [3], correction = 0, keepdim = True)
        getitem_213 = var_mean_51[0]
        getitem_214 = var_mean_51[1];  var_mean_51 = None
        add_233 = torch.ops.aten.add.Tensor(getitem_213, 1e-05);  getitem_213 = None
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
        sub_67 = torch.ops.aten.sub.Tensor(clone_179, getitem_214);  clone_179 = getitem_214 = None
        mul_206 = torch.ops.aten.mul.Tensor(sub_67, rsqrt_51);  sub_67 = rsqrt_51 = None
        mul_207 = torch.ops.aten.mul.Tensor(mul_206, arg84_1);  mul_206 = arg84_1 = None
        add_234 = torch.ops.aten.add.Tensor(mul_207, arg85_1);  mul_207 = arg85_1 = None
        permute_226 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        view_335 = torch.ops.aten.view.default(add_234, [1568, 384]);  add_234 = None
        mm_36 = torch.ops.aten.mm.default(view_335, permute_226);  view_335 = permute_226 = None
        view_336 = torch.ops.aten.view.default(mm_36, [8, 14, 14, 1152]);  mm_36 = None
        view_337 = torch.ops.aten.view.default(view_336, [8, 196, 3, 12, 32]);  view_336 = None
        permute_227 = torch.ops.aten.permute.default(view_337, [2, 0, 3, 1, 4]);  view_337 = None
        unbind_17 = torch.ops.aten.unbind.int(permute_227);  permute_227 = None
        getitem_215 = unbind_17[0]
        getitem_216 = unbind_17[1]
        getitem_217 = unbind_17[2];  unbind_17 = None
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_215, getitem_216, getitem_217, None, False);  getitem_215 = getitem_216 = getitem_217 = None
        getitem_218 = _scaled_dot_product_efficient_attention_15[0];  _scaled_dot_product_efficient_attention_15 = None
        permute_228 = torch.ops.aten.permute.default(getitem_218, [0, 2, 1, 3]);  getitem_218 = None
        view_338 = torch.ops.aten.view.default(permute_228, [8, 14, 14, 384]);  permute_228 = None
        view_339 = torch.ops.aten.view.default(view_338, [1568, 384]);  view_338 = None
        permute_229 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg88_1, view_339, permute_229);  arg88_1 = view_339 = permute_229 = None
        view_340 = torch.ops.aten.view.default(addmm_76, [8, 14, 14, 384]);  addmm_76 = None
        add_235 = torch.ops.aten.add.Tensor(add_232, view_340);  add_232 = view_340 = None
        clone_181 = torch.ops.aten.clone.default(add_235, memory_format = torch.contiguous_format)
        var_mean_52 = torch.ops.aten.var_mean.correction(clone_181, [3], correction = 0, keepdim = True)
        getitem_222 = var_mean_52[0]
        getitem_223 = var_mean_52[1];  var_mean_52 = None
        add_236 = torch.ops.aten.add.Tensor(getitem_222, 1e-05);  getitem_222 = None
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_68 = torch.ops.aten.sub.Tensor(clone_181, getitem_223);  clone_181 = getitem_223 = None
        mul_208 = torch.ops.aten.mul.Tensor(sub_68, rsqrt_52);  sub_68 = rsqrt_52 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, arg89_1);  mul_208 = arg89_1 = None
        add_237 = torch.ops.aten.add.Tensor(mul_209, arg90_1);  mul_209 = arg90_1 = None
        view_341 = torch.ops.aten.view.default(add_237, [1568, 384]);  add_237 = None
        permute_230 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg92_1, view_341, permute_230);  arg92_1 = view_341 = permute_230 = None
        view_342 = torch.ops.aten.view.default(addmm_77, [8, 14, 14, 1152]);  addmm_77 = None
        mul_210 = torch.ops.aten.mul.Tensor(view_342, 0.5)
        mul_211 = torch.ops.aten.mul.Tensor(view_342, 0.7071067811865476);  view_342 = None
        erf_25 = torch.ops.aten.erf.default(mul_211);  mul_211 = None
        add_238 = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_212 = torch.ops.aten.mul.Tensor(mul_210, add_238);  mul_210 = add_238 = None
        view_343 = torch.ops.aten.view.default(mul_212, [1568, 1152]);  mul_212 = None
        permute_231 = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg94_1, view_343, permute_231);  arg94_1 = view_343 = permute_231 = None
        view_344 = torch.ops.aten.view.default(addmm_78, [8, 14, 14, 384]);  addmm_78 = None
        add_239 = torch.ops.aten.add.Tensor(add_235, view_344);  add_235 = view_344 = None
        clone_184 = torch.ops.aten.clone.default(add_239, memory_format = torch.contiguous_format)
        var_mean_53 = torch.ops.aten.var_mean.correction(clone_184, [3], correction = 0, keepdim = True)
        getitem_224 = var_mean_53[0]
        getitem_225 = var_mean_53[1];  var_mean_53 = None
        add_240 = torch.ops.aten.add.Tensor(getitem_224, 1e-05);  getitem_224 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_69 = torch.ops.aten.sub.Tensor(clone_184, getitem_225);  clone_184 = getitem_225 = None
        mul_213 = torch.ops.aten.mul.Tensor(sub_69, rsqrt_53);  sub_69 = rsqrt_53 = None
        mul_214 = torch.ops.aten.mul.Tensor(mul_213, arg95_1);  mul_213 = arg95_1 = None
        add_241 = torch.ops.aten.add.Tensor(mul_214, arg96_1);  mul_214 = arg96_1 = None
        permute_232 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        view_345 = torch.ops.aten.view.default(add_241, [1568, 384]);  add_241 = None
        mm_37 = torch.ops.aten.mm.default(view_345, permute_232);  view_345 = permute_232 = None
        view_346 = torch.ops.aten.view.default(mm_37, [8, 14, 14, 1152]);  mm_37 = None
        view_347 = torch.ops.aten.view.default(view_346, [8, 196, 3, 12, 32]);  view_346 = None
        permute_233 = torch.ops.aten.permute.default(view_347, [2, 0, 3, 1, 4]);  view_347 = None
        unbind_18 = torch.ops.aten.unbind.int(permute_233);  permute_233 = None
        getitem_226 = unbind_18[0]
        getitem_227 = unbind_18[1]
        getitem_228 = unbind_18[2];  unbind_18 = None
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_226, getitem_227, getitem_228, None, False);  getitem_226 = getitem_227 = getitem_228 = None
        getitem_229 = _scaled_dot_product_efficient_attention_16[0];  _scaled_dot_product_efficient_attention_16 = None
        permute_234 = torch.ops.aten.permute.default(getitem_229, [0, 2, 1, 3]);  getitem_229 = None
        view_348 = torch.ops.aten.view.default(permute_234, [8, 14, 14, 384]);  permute_234 = None
        view_349 = torch.ops.aten.view.default(view_348, [1568, 384]);  view_348 = None
        permute_235 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg99_1, view_349, permute_235);  arg99_1 = view_349 = permute_235 = None
        view_350 = torch.ops.aten.view.default(addmm_79, [8, 14, 14, 384]);  addmm_79 = None
        add_242 = torch.ops.aten.add.Tensor(add_239, view_350);  add_239 = view_350 = None
        clone_186 = torch.ops.aten.clone.default(add_242, memory_format = torch.contiguous_format)
        var_mean_54 = torch.ops.aten.var_mean.correction(clone_186, [3], correction = 0, keepdim = True)
        getitem_233 = var_mean_54[0]
        getitem_234 = var_mean_54[1];  var_mean_54 = None
        add_243 = torch.ops.aten.add.Tensor(getitem_233, 1e-05);  getitem_233 = None
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        sub_70 = torch.ops.aten.sub.Tensor(clone_186, getitem_234);  clone_186 = getitem_234 = None
        mul_215 = torch.ops.aten.mul.Tensor(sub_70, rsqrt_54);  sub_70 = rsqrt_54 = None
        mul_216 = torch.ops.aten.mul.Tensor(mul_215, arg100_1);  mul_215 = arg100_1 = None
        add_244 = torch.ops.aten.add.Tensor(mul_216, arg101_1);  mul_216 = arg101_1 = None
        view_351 = torch.ops.aten.view.default(add_244, [1568, 384]);  add_244 = None
        permute_236 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg103_1, view_351, permute_236);  arg103_1 = view_351 = permute_236 = None
        view_352 = torch.ops.aten.view.default(addmm_80, [8, 14, 14, 1152]);  addmm_80 = None
        mul_217 = torch.ops.aten.mul.Tensor(view_352, 0.5)
        mul_218 = torch.ops.aten.mul.Tensor(view_352, 0.7071067811865476);  view_352 = None
        erf_26 = torch.ops.aten.erf.default(mul_218);  mul_218 = None
        add_245 = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_219 = torch.ops.aten.mul.Tensor(mul_217, add_245);  mul_217 = add_245 = None
        view_353 = torch.ops.aten.view.default(mul_219, [1568, 1152]);  mul_219 = None
        permute_237 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg105_1, view_353, permute_237);  arg105_1 = view_353 = permute_237 = None
        view_354 = torch.ops.aten.view.default(addmm_81, [8, 14, 14, 384]);  addmm_81 = None
        add_246 = torch.ops.aten.add.Tensor(add_242, view_354);  add_242 = view_354 = None
        clone_189 = torch.ops.aten.clone.default(add_246, memory_format = torch.contiguous_format)
        var_mean_55 = torch.ops.aten.var_mean.correction(clone_189, [3], correction = 0, keepdim = True)
        getitem_235 = var_mean_55[0]
        getitem_236 = var_mean_55[1];  var_mean_55 = None
        add_247 = torch.ops.aten.add.Tensor(getitem_235, 1e-05);  getitem_235 = None
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        sub_71 = torch.ops.aten.sub.Tensor(clone_189, getitem_236);  clone_189 = getitem_236 = None
        mul_220 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_55);  sub_71 = rsqrt_55 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, arg106_1);  mul_220 = arg106_1 = None
        add_248 = torch.ops.aten.add.Tensor(mul_221, arg107_1);  mul_221 = arg107_1 = None
        permute_238 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        view_355 = torch.ops.aten.view.default(add_248, [1568, 384]);  add_248 = None
        mm_38 = torch.ops.aten.mm.default(view_355, permute_238);  view_355 = permute_238 = None
        view_356 = torch.ops.aten.view.default(mm_38, [8, 14, 14, 1152]);  mm_38 = None
        view_357 = torch.ops.aten.view.default(view_356, [8, 196, 3, 12, 32]);  view_356 = None
        permute_239 = torch.ops.aten.permute.default(view_357, [2, 0, 3, 1, 4]);  view_357 = None
        unbind_19 = torch.ops.aten.unbind.int(permute_239);  permute_239 = None
        getitem_237 = unbind_19[0]
        getitem_238 = unbind_19[1]
        getitem_239 = unbind_19[2];  unbind_19 = None
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_237, getitem_238, getitem_239, None, False);  getitem_237 = getitem_238 = getitem_239 = None
        getitem_240 = _scaled_dot_product_efficient_attention_17[0];  _scaled_dot_product_efficient_attention_17 = None
        permute_240 = torch.ops.aten.permute.default(getitem_240, [0, 2, 1, 3]);  getitem_240 = None
        view_358 = torch.ops.aten.view.default(permute_240, [8, 14, 14, 384]);  permute_240 = None
        view_359 = torch.ops.aten.view.default(view_358, [1568, 384]);  view_358 = None
        permute_241 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg110_1, view_359, permute_241);  arg110_1 = view_359 = permute_241 = None
        view_360 = torch.ops.aten.view.default(addmm_82, [8, 14, 14, 384]);  addmm_82 = None
        add_249 = torch.ops.aten.add.Tensor(add_246, view_360);  add_246 = view_360 = None
        clone_191 = torch.ops.aten.clone.default(add_249, memory_format = torch.contiguous_format)
        var_mean_56 = torch.ops.aten.var_mean.correction(clone_191, [3], correction = 0, keepdim = True)
        getitem_244 = var_mean_56[0]
        getitem_245 = var_mean_56[1];  var_mean_56 = None
        add_250 = torch.ops.aten.add.Tensor(getitem_244, 1e-05);  getitem_244 = None
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        sub_72 = torch.ops.aten.sub.Tensor(clone_191, getitem_245);  clone_191 = getitem_245 = None
        mul_222 = torch.ops.aten.mul.Tensor(sub_72, rsqrt_56);  sub_72 = rsqrt_56 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_222, arg111_1);  mul_222 = arg111_1 = None
        add_251 = torch.ops.aten.add.Tensor(mul_223, arg112_1);  mul_223 = arg112_1 = None
        view_361 = torch.ops.aten.view.default(add_251, [1568, 384]);  add_251 = None
        permute_242 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg114_1, view_361, permute_242);  arg114_1 = view_361 = permute_242 = None
        view_362 = torch.ops.aten.view.default(addmm_83, [8, 14, 14, 1152]);  addmm_83 = None
        mul_224 = torch.ops.aten.mul.Tensor(view_362, 0.5)
        mul_225 = torch.ops.aten.mul.Tensor(view_362, 0.7071067811865476);  view_362 = None
        erf_27 = torch.ops.aten.erf.default(mul_225);  mul_225 = None
        add_252 = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_226 = torch.ops.aten.mul.Tensor(mul_224, add_252);  mul_224 = add_252 = None
        view_363 = torch.ops.aten.view.default(mul_226, [1568, 1152]);  mul_226 = None
        permute_243 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg116_1, view_363, permute_243);  arg116_1 = view_363 = permute_243 = None
        view_364 = torch.ops.aten.view.default(addmm_84, [8, 14, 14, 384]);  addmm_84 = None
        add_253 = torch.ops.aten.add.Tensor(add_249, view_364);  add_249 = view_364 = None
        clone_194 = torch.ops.aten.clone.default(add_253, memory_format = torch.contiguous_format)
        var_mean_57 = torch.ops.aten.var_mean.correction(clone_194, [3], correction = 0, keepdim = True)
        getitem_246 = var_mean_57[0]
        getitem_247 = var_mean_57[1];  var_mean_57 = None
        add_254 = torch.ops.aten.add.Tensor(getitem_246, 1e-05);  getitem_246 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        sub_73 = torch.ops.aten.sub.Tensor(clone_194, getitem_247);  clone_194 = getitem_247 = None
        mul_227 = torch.ops.aten.mul.Tensor(sub_73, rsqrt_57);  sub_73 = rsqrt_57 = None
        mul_228 = torch.ops.aten.mul.Tensor(mul_227, arg117_1);  mul_227 = arg117_1 = None
        add_255 = torch.ops.aten.add.Tensor(mul_228, arg118_1);  mul_228 = arg118_1 = None
        permute_244 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        view_365 = torch.ops.aten.view.default(add_255, [1568, 384]);  add_255 = None
        mm_39 = torch.ops.aten.mm.default(view_365, permute_244);  view_365 = permute_244 = None
        view_366 = torch.ops.aten.view.default(mm_39, [8, 14, 14, 1152]);  mm_39 = None
        view_367 = torch.ops.aten.view.default(view_366, [8, 196, 3, 12, 32]);  view_366 = None
        permute_245 = torch.ops.aten.permute.default(view_367, [2, 0, 3, 1, 4]);  view_367 = None
        unbind_20 = torch.ops.aten.unbind.int(permute_245);  permute_245 = None
        getitem_248 = unbind_20[0]
        getitem_249 = unbind_20[1]
        getitem_250 = unbind_20[2];  unbind_20 = None
        _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_248, getitem_249, getitem_250, None, False);  getitem_248 = getitem_249 = getitem_250 = None
        getitem_251 = _scaled_dot_product_efficient_attention_18[0];  _scaled_dot_product_efficient_attention_18 = None
        permute_246 = torch.ops.aten.permute.default(getitem_251, [0, 2, 1, 3]);  getitem_251 = None
        view_368 = torch.ops.aten.view.default(permute_246, [8, 14, 14, 384]);  permute_246 = None
        view_369 = torch.ops.aten.view.default(view_368, [1568, 384]);  view_368 = None
        permute_247 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg121_1, view_369, permute_247);  arg121_1 = view_369 = permute_247 = None
        view_370 = torch.ops.aten.view.default(addmm_85, [8, 14, 14, 384]);  addmm_85 = None
        add_256 = torch.ops.aten.add.Tensor(add_253, view_370);  add_253 = view_370 = None
        clone_196 = torch.ops.aten.clone.default(add_256, memory_format = torch.contiguous_format)
        var_mean_58 = torch.ops.aten.var_mean.correction(clone_196, [3], correction = 0, keepdim = True)
        getitem_255 = var_mean_58[0]
        getitem_256 = var_mean_58[1];  var_mean_58 = None
        add_257 = torch.ops.aten.add.Tensor(getitem_255, 1e-05);  getitem_255 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
        sub_74 = torch.ops.aten.sub.Tensor(clone_196, getitem_256);  clone_196 = getitem_256 = None
        mul_229 = torch.ops.aten.mul.Tensor(sub_74, rsqrt_58);  sub_74 = rsqrt_58 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_229, arg122_1);  mul_229 = arg122_1 = None
        add_258 = torch.ops.aten.add.Tensor(mul_230, arg123_1);  mul_230 = arg123_1 = None
        view_371 = torch.ops.aten.view.default(add_258, [1568, 384]);  add_258 = None
        permute_248 = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg125_1, view_371, permute_248);  arg125_1 = view_371 = permute_248 = None
        view_372 = torch.ops.aten.view.default(addmm_86, [8, 14, 14, 1152]);  addmm_86 = None
        mul_231 = torch.ops.aten.mul.Tensor(view_372, 0.5)
        mul_232 = torch.ops.aten.mul.Tensor(view_372, 0.7071067811865476);  view_372 = None
        erf_28 = torch.ops.aten.erf.default(mul_232);  mul_232 = None
        add_259 = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_231, add_259);  mul_231 = add_259 = None
        view_373 = torch.ops.aten.view.default(mul_233, [1568, 1152]);  mul_233 = None
        permute_249 = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg127_1, view_373, permute_249);  arg127_1 = view_373 = permute_249 = None
        view_374 = torch.ops.aten.view.default(addmm_87, [8, 14, 14, 384]);  addmm_87 = None
        add_260 = torch.ops.aten.add.Tensor(add_256, view_374);  add_256 = view_374 = None
        clone_199 = torch.ops.aten.clone.default(add_260, memory_format = torch.contiguous_format)
        var_mean_59 = torch.ops.aten.var_mean.correction(clone_199, [3], correction = 0, keepdim = True)
        getitem_257 = var_mean_59[0]
        getitem_258 = var_mean_59[1];  var_mean_59 = None
        add_261 = torch.ops.aten.add.Tensor(getitem_257, 1e-05);  getitem_257 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        sub_75 = torch.ops.aten.sub.Tensor(clone_199, getitem_258);  clone_199 = getitem_258 = None
        mul_234 = torch.ops.aten.mul.Tensor(sub_75, rsqrt_59);  sub_75 = rsqrt_59 = None
        mul_235 = torch.ops.aten.mul.Tensor(mul_234, arg128_1);  mul_234 = arg128_1 = None
        add_262 = torch.ops.aten.add.Tensor(mul_235, arg129_1);  mul_235 = arg129_1 = None
        permute_250 = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        view_375 = torch.ops.aten.view.default(add_262, [1568, 384]);  add_262 = None
        mm_40 = torch.ops.aten.mm.default(view_375, permute_250);  view_375 = permute_250 = None
        view_376 = torch.ops.aten.view.default(mm_40, [8, 14, 14, 1152]);  mm_40 = None
        view_377 = torch.ops.aten.view.default(view_376, [8, 196, 3, 12, 32]);  view_376 = None
        permute_251 = torch.ops.aten.permute.default(view_377, [2, 0, 3, 1, 4]);  view_377 = None
        unbind_21 = torch.ops.aten.unbind.int(permute_251);  permute_251 = None
        getitem_259 = unbind_21[0]
        getitem_260 = unbind_21[1]
        getitem_261 = unbind_21[2];  unbind_21 = None
        _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_259, getitem_260, getitem_261, None, False);  getitem_259 = getitem_260 = getitem_261 = None
        getitem_262 = _scaled_dot_product_efficient_attention_19[0];  _scaled_dot_product_efficient_attention_19 = None
        permute_252 = torch.ops.aten.permute.default(getitem_262, [0, 2, 1, 3]);  getitem_262 = None
        view_378 = torch.ops.aten.view.default(permute_252, [8, 14, 14, 384]);  permute_252 = None
        view_379 = torch.ops.aten.view.default(view_378, [1568, 384]);  view_378 = None
        permute_253 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg132_1, view_379, permute_253);  arg132_1 = view_379 = permute_253 = None
        view_380 = torch.ops.aten.view.default(addmm_88, [8, 14, 14, 384]);  addmm_88 = None
        add_263 = torch.ops.aten.add.Tensor(add_260, view_380);  add_260 = view_380 = None
        clone_201 = torch.ops.aten.clone.default(add_263, memory_format = torch.contiguous_format)
        var_mean_60 = torch.ops.aten.var_mean.correction(clone_201, [3], correction = 0, keepdim = True)
        getitem_266 = var_mean_60[0]
        getitem_267 = var_mean_60[1];  var_mean_60 = None
        add_264 = torch.ops.aten.add.Tensor(getitem_266, 1e-05);  getitem_266 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
        sub_76 = torch.ops.aten.sub.Tensor(clone_201, getitem_267);  clone_201 = getitem_267 = None
        mul_236 = torch.ops.aten.mul.Tensor(sub_76, rsqrt_60);  sub_76 = rsqrt_60 = None
        mul_237 = torch.ops.aten.mul.Tensor(mul_236, arg133_1);  mul_236 = arg133_1 = None
        add_265 = torch.ops.aten.add.Tensor(mul_237, arg134_1);  mul_237 = arg134_1 = None
        view_381 = torch.ops.aten.view.default(add_265, [1568, 384]);  add_265 = None
        permute_254 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg136_1, view_381, permute_254);  arg136_1 = view_381 = permute_254 = None
        view_382 = torch.ops.aten.view.default(addmm_89, [8, 14, 14, 1152]);  addmm_89 = None
        mul_238 = torch.ops.aten.mul.Tensor(view_382, 0.5)
        mul_239 = torch.ops.aten.mul.Tensor(view_382, 0.7071067811865476);  view_382 = None
        erf_29 = torch.ops.aten.erf.default(mul_239);  mul_239 = None
        add_266 = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_240 = torch.ops.aten.mul.Tensor(mul_238, add_266);  mul_238 = add_266 = None
        view_383 = torch.ops.aten.view.default(mul_240, [1568, 1152]);  mul_240 = None
        permute_255 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg138_1, view_383, permute_255);  arg138_1 = view_383 = permute_255 = None
        view_384 = torch.ops.aten.view.default(addmm_90, [8, 14, 14, 384]);  addmm_90 = None
        add_267 = torch.ops.aten.add.Tensor(add_263, view_384);  add_263 = view_384 = None
        clone_204 = torch.ops.aten.clone.default(add_267, memory_format = torch.contiguous_format)
        var_mean_61 = torch.ops.aten.var_mean.correction(clone_204, [3], correction = 0, keepdim = True)
        getitem_268 = var_mean_61[0]
        getitem_269 = var_mean_61[1];  var_mean_61 = None
        add_268 = torch.ops.aten.add.Tensor(getitem_268, 1e-05);  getitem_268 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        sub_77 = torch.ops.aten.sub.Tensor(clone_204, getitem_269);  clone_204 = getitem_269 = None
        mul_241 = torch.ops.aten.mul.Tensor(sub_77, rsqrt_61);  sub_77 = rsqrt_61 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_241, arg139_1);  mul_241 = arg139_1 = None
        add_269 = torch.ops.aten.add.Tensor(mul_242, arg140_1);  mul_242 = arg140_1 = None
        permute_256 = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        view_385 = torch.ops.aten.view.default(add_269, [1568, 384]);  add_269 = None
        mm_41 = torch.ops.aten.mm.default(view_385, permute_256);  view_385 = permute_256 = None
        view_386 = torch.ops.aten.view.default(mm_41, [8, 14, 14, 1152]);  mm_41 = None
        view_387 = torch.ops.aten.view.default(view_386, [8, 196, 3, 12, 32]);  view_386 = None
        permute_257 = torch.ops.aten.permute.default(view_387, [2, 0, 3, 1, 4]);  view_387 = None
        unbind_22 = torch.ops.aten.unbind.int(permute_257);  permute_257 = None
        getitem_270 = unbind_22[0]
        getitem_271 = unbind_22[1]
        getitem_272 = unbind_22[2];  unbind_22 = None
        _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_270, getitem_271, getitem_272, None, False);  getitem_270 = getitem_271 = getitem_272 = None
        getitem_273 = _scaled_dot_product_efficient_attention_20[0];  _scaled_dot_product_efficient_attention_20 = None
        permute_258 = torch.ops.aten.permute.default(getitem_273, [0, 2, 1, 3]);  getitem_273 = None
        view_388 = torch.ops.aten.view.default(permute_258, [8, 14, 14, 384]);  permute_258 = None
        view_389 = torch.ops.aten.view.default(view_388, [1568, 384]);  view_388 = None
        permute_259 = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg143_1, view_389, permute_259);  arg143_1 = view_389 = permute_259 = None
        view_390 = torch.ops.aten.view.default(addmm_91, [8, 14, 14, 384]);  addmm_91 = None
        add_270 = torch.ops.aten.add.Tensor(add_267, view_390);  add_267 = view_390 = None
        clone_206 = torch.ops.aten.clone.default(add_270, memory_format = torch.contiguous_format)
        var_mean_62 = torch.ops.aten.var_mean.correction(clone_206, [3], correction = 0, keepdim = True)
        getitem_277 = var_mean_62[0]
        getitem_278 = var_mean_62[1];  var_mean_62 = None
        add_271 = torch.ops.aten.add.Tensor(getitem_277, 1e-05);  getitem_277 = None
        rsqrt_62 = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_78 = torch.ops.aten.sub.Tensor(clone_206, getitem_278);  clone_206 = getitem_278 = None
        mul_243 = torch.ops.aten.mul.Tensor(sub_78, rsqrt_62);  sub_78 = rsqrt_62 = None
        mul_244 = torch.ops.aten.mul.Tensor(mul_243, arg144_1);  mul_243 = arg144_1 = None
        add_272 = torch.ops.aten.add.Tensor(mul_244, arg145_1);  mul_244 = arg145_1 = None
        view_391 = torch.ops.aten.view.default(add_272, [1568, 384]);  add_272 = None
        permute_260 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg147_1, view_391, permute_260);  arg147_1 = view_391 = permute_260 = None
        view_392 = torch.ops.aten.view.default(addmm_92, [8, 14, 14, 1152]);  addmm_92 = None
        mul_245 = torch.ops.aten.mul.Tensor(view_392, 0.5)
        mul_246 = torch.ops.aten.mul.Tensor(view_392, 0.7071067811865476);  view_392 = None
        erf_30 = torch.ops.aten.erf.default(mul_246);  mul_246 = None
        add_273 = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_247 = torch.ops.aten.mul.Tensor(mul_245, add_273);  mul_245 = add_273 = None
        view_393 = torch.ops.aten.view.default(mul_247, [1568, 1152]);  mul_247 = None
        permute_261 = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg149_1, view_393, permute_261);  arg149_1 = view_393 = permute_261 = None
        view_394 = torch.ops.aten.view.default(addmm_93, [8, 14, 14, 384]);  addmm_93 = None
        add_274 = torch.ops.aten.add.Tensor(add_270, view_394);  add_270 = view_394 = None
        clone_209 = torch.ops.aten.clone.default(add_274, memory_format = torch.contiguous_format)
        var_mean_63 = torch.ops.aten.var_mean.correction(clone_209, [3], correction = 0, keepdim = True)
        getitem_279 = var_mean_63[0]
        getitem_280 = var_mean_63[1];  var_mean_63 = None
        add_275 = torch.ops.aten.add.Tensor(getitem_279, 1e-05);  getitem_279 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
        sub_79 = torch.ops.aten.sub.Tensor(clone_209, getitem_280);  clone_209 = getitem_280 = None
        mul_248 = torch.ops.aten.mul.Tensor(sub_79, rsqrt_63);  sub_79 = rsqrt_63 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, arg150_1);  mul_248 = arg150_1 = None
        add_276 = torch.ops.aten.add.Tensor(mul_249, arg151_1);  mul_249 = arg151_1 = None
        permute_262 = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        view_395 = torch.ops.aten.view.default(add_276, [1568, 384]);  add_276 = None
        mm_42 = torch.ops.aten.mm.default(view_395, permute_262);  view_395 = permute_262 = None
        view_396 = torch.ops.aten.view.default(mm_42, [8, 14, 14, 1152]);  mm_42 = None
        view_397 = torch.ops.aten.view.default(view_396, [8, 196, 3, 12, 32]);  view_396 = None
        permute_263 = torch.ops.aten.permute.default(view_397, [2, 0, 3, 1, 4]);  view_397 = None
        unbind_23 = torch.ops.aten.unbind.int(permute_263);  permute_263 = None
        getitem_281 = unbind_23[0]
        getitem_282 = unbind_23[1]
        getitem_283 = unbind_23[2];  unbind_23 = None
        _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_281, getitem_282, getitem_283, None, False);  getitem_281 = getitem_282 = getitem_283 = None
        getitem_284 = _scaled_dot_product_efficient_attention_21[0];  _scaled_dot_product_efficient_attention_21 = None
        permute_264 = torch.ops.aten.permute.default(getitem_284, [0, 2, 1, 3]);  getitem_284 = None
        view_398 = torch.ops.aten.view.default(permute_264, [8, 14, 14, 384]);  permute_264 = None
        view_399 = torch.ops.aten.view.default(view_398, [1568, 384]);  view_398 = None
        permute_265 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg154_1, view_399, permute_265);  arg154_1 = view_399 = permute_265 = None
        view_400 = torch.ops.aten.view.default(addmm_94, [8, 14, 14, 384]);  addmm_94 = None
        add_277 = torch.ops.aten.add.Tensor(add_274, view_400);  add_274 = view_400 = None
        clone_211 = torch.ops.aten.clone.default(add_277, memory_format = torch.contiguous_format)
        var_mean_64 = torch.ops.aten.var_mean.correction(clone_211, [3], correction = 0, keepdim = True)
        getitem_288 = var_mean_64[0]
        getitem_289 = var_mean_64[1];  var_mean_64 = None
        add_278 = torch.ops.aten.add.Tensor(getitem_288, 1e-05);  getitem_288 = None
        rsqrt_64 = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
        sub_80 = torch.ops.aten.sub.Tensor(clone_211, getitem_289);  clone_211 = getitem_289 = None
        mul_250 = torch.ops.aten.mul.Tensor(sub_80, rsqrt_64);  sub_80 = rsqrt_64 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_250, arg155_1);  mul_250 = arg155_1 = None
        add_279 = torch.ops.aten.add.Tensor(mul_251, arg156_1);  mul_251 = arg156_1 = None
        view_401 = torch.ops.aten.view.default(add_279, [1568, 384]);  add_279 = None
        permute_266 = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg158_1, view_401, permute_266);  arg158_1 = view_401 = permute_266 = None
        view_402 = torch.ops.aten.view.default(addmm_95, [8, 14, 14, 1152]);  addmm_95 = None
        mul_252 = torch.ops.aten.mul.Tensor(view_402, 0.5)
        mul_253 = torch.ops.aten.mul.Tensor(view_402, 0.7071067811865476);  view_402 = None
        erf_31 = torch.ops.aten.erf.default(mul_253);  mul_253 = None
        add_280 = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_254 = torch.ops.aten.mul.Tensor(mul_252, add_280);  mul_252 = add_280 = None
        view_403 = torch.ops.aten.view.default(mul_254, [1568, 1152]);  mul_254 = None
        permute_267 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg160_1, view_403, permute_267);  arg160_1 = view_403 = permute_267 = None
        view_404 = torch.ops.aten.view.default(addmm_96, [8, 14, 14, 384]);  addmm_96 = None
        add_281 = torch.ops.aten.add.Tensor(add_277, view_404);  add_277 = view_404 = None
        clone_214 = torch.ops.aten.clone.default(add_281, memory_format = torch.contiguous_format)
        var_mean_65 = torch.ops.aten.var_mean.correction(clone_214, [3], correction = 0, keepdim = True)
        getitem_290 = var_mean_65[0]
        getitem_291 = var_mean_65[1];  var_mean_65 = None
        add_282 = torch.ops.aten.add.Tensor(getitem_290, 1e-05);  getitem_290 = None
        rsqrt_65 = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        sub_81 = torch.ops.aten.sub.Tensor(clone_214, getitem_291);  clone_214 = getitem_291 = None
        mul_255 = torch.ops.aten.mul.Tensor(sub_81, rsqrt_65);  sub_81 = rsqrt_65 = None
        mul_256 = torch.ops.aten.mul.Tensor(mul_255, arg161_1);  mul_255 = arg161_1 = None
        add_283 = torch.ops.aten.add.Tensor(mul_256, arg162_1);  mul_256 = arg162_1 = None
        permute_268 = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        view_405 = torch.ops.aten.view.default(add_283, [1568, 384]);  add_283 = None
        mm_43 = torch.ops.aten.mm.default(view_405, permute_268);  view_405 = permute_268 = None
        view_406 = torch.ops.aten.view.default(mm_43, [8, 14, 14, 1152]);  mm_43 = None
        view_407 = torch.ops.aten.view.default(view_406, [8, 196, 3, 12, 32]);  view_406 = None
        permute_269 = torch.ops.aten.permute.default(view_407, [2, 0, 3, 1, 4]);  view_407 = None
        unbind_24 = torch.ops.aten.unbind.int(permute_269);  permute_269 = None
        getitem_292 = unbind_24[0]
        getitem_293 = unbind_24[1]
        getitem_294 = unbind_24[2];  unbind_24 = None
        _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_292, getitem_293, getitem_294, None, False);  getitem_292 = getitem_293 = getitem_294 = None
        getitem_295 = _scaled_dot_product_efficient_attention_22[0];  _scaled_dot_product_efficient_attention_22 = None
        permute_270 = torch.ops.aten.permute.default(getitem_295, [0, 2, 1, 3]);  getitem_295 = None
        view_408 = torch.ops.aten.view.default(permute_270, [8, 14, 14, 384]);  permute_270 = None
        view_409 = torch.ops.aten.view.default(view_408, [1568, 384]);  view_408 = None
        permute_271 = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg165_1, view_409, permute_271);  arg165_1 = view_409 = permute_271 = None
        view_410 = torch.ops.aten.view.default(addmm_97, [8, 14, 14, 384]);  addmm_97 = None
        add_284 = torch.ops.aten.add.Tensor(add_281, view_410);  add_281 = view_410 = None
        clone_216 = torch.ops.aten.clone.default(add_284, memory_format = torch.contiguous_format)
        var_mean_66 = torch.ops.aten.var_mean.correction(clone_216, [3], correction = 0, keepdim = True)
        getitem_299 = var_mean_66[0]
        getitem_300 = var_mean_66[1];  var_mean_66 = None
        add_285 = torch.ops.aten.add.Tensor(getitem_299, 1e-05);  getitem_299 = None
        rsqrt_66 = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
        sub_82 = torch.ops.aten.sub.Tensor(clone_216, getitem_300);  clone_216 = getitem_300 = None
        mul_257 = torch.ops.aten.mul.Tensor(sub_82, rsqrt_66);  sub_82 = rsqrt_66 = None
        mul_258 = torch.ops.aten.mul.Tensor(mul_257, arg166_1);  mul_257 = arg166_1 = None
        add_286 = torch.ops.aten.add.Tensor(mul_258, arg167_1);  mul_258 = arg167_1 = None
        view_411 = torch.ops.aten.view.default(add_286, [1568, 384]);  add_286 = None
        permute_272 = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg169_1, view_411, permute_272);  arg169_1 = view_411 = permute_272 = None
        view_412 = torch.ops.aten.view.default(addmm_98, [8, 14, 14, 1152]);  addmm_98 = None
        mul_259 = torch.ops.aten.mul.Tensor(view_412, 0.5)
        mul_260 = torch.ops.aten.mul.Tensor(view_412, 0.7071067811865476);  view_412 = None
        erf_32 = torch.ops.aten.erf.default(mul_260);  mul_260 = None
        add_287 = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_261 = torch.ops.aten.mul.Tensor(mul_259, add_287);  mul_259 = add_287 = None
        view_413 = torch.ops.aten.view.default(mul_261, [1568, 1152]);  mul_261 = None
        permute_273 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg171_1, view_413, permute_273);  arg171_1 = view_413 = permute_273 = None
        view_414 = torch.ops.aten.view.default(addmm_99, [8, 14, 14, 384]);  addmm_99 = None
        add_288 = torch.ops.aten.add.Tensor(add_284, view_414);  add_284 = view_414 = None
        clone_219 = torch.ops.aten.clone.default(add_288, memory_format = torch.contiguous_format)
        var_mean_67 = torch.ops.aten.var_mean.correction(clone_219, [3], correction = 0, keepdim = True)
        getitem_301 = var_mean_67[0]
        getitem_302 = var_mean_67[1];  var_mean_67 = None
        add_289 = torch.ops.aten.add.Tensor(getitem_301, 1e-05);  getitem_301 = None
        rsqrt_67 = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
        sub_83 = torch.ops.aten.sub.Tensor(clone_219, getitem_302);  clone_219 = getitem_302 = None
        mul_262 = torch.ops.aten.mul.Tensor(sub_83, rsqrt_67);  sub_83 = rsqrt_67 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, arg172_1);  mul_262 = arg172_1 = None
        add_290 = torch.ops.aten.add.Tensor(mul_263, arg173_1);  mul_263 = arg173_1 = None
        permute_274 = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        view_415 = torch.ops.aten.view.default(add_290, [1568, 384]);  add_290 = None
        mm_44 = torch.ops.aten.mm.default(view_415, permute_274);  view_415 = permute_274 = None
        view_416 = torch.ops.aten.view.default(mm_44, [8, 14, 14, 1152]);  mm_44 = None
        view_417 = torch.ops.aten.view.default(view_416, [8, 196, 3, 12, 32]);  view_416 = None
        permute_275 = torch.ops.aten.permute.default(view_417, [2, 0, 3, 1, 4]);  view_417 = None
        unbind_25 = torch.ops.aten.unbind.int(permute_275);  permute_275 = None
        getitem_303 = unbind_25[0]
        getitem_304 = unbind_25[1]
        getitem_305 = unbind_25[2];  unbind_25 = None
        _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_303, getitem_304, getitem_305, None, False);  getitem_303 = getitem_304 = getitem_305 = None
        getitem_306 = _scaled_dot_product_efficient_attention_23[0];  _scaled_dot_product_efficient_attention_23 = None
        permute_276 = torch.ops.aten.permute.default(getitem_306, [0, 2, 1, 3]);  getitem_306 = None
        view_418 = torch.ops.aten.view.default(permute_276, [8, 14, 14, 384]);  permute_276 = None
        view_419 = torch.ops.aten.view.default(view_418, [1568, 384]);  view_418 = None
        permute_277 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg176_1, view_419, permute_277);  arg176_1 = view_419 = permute_277 = None
        view_420 = torch.ops.aten.view.default(addmm_100, [8, 14, 14, 384]);  addmm_100 = None
        add_291 = torch.ops.aten.add.Tensor(add_288, view_420);  add_288 = view_420 = None
        clone_221 = torch.ops.aten.clone.default(add_291, memory_format = torch.contiguous_format)
        var_mean_68 = torch.ops.aten.var_mean.correction(clone_221, [3], correction = 0, keepdim = True)
        getitem_310 = var_mean_68[0]
        getitem_311 = var_mean_68[1];  var_mean_68 = None
        add_292 = torch.ops.aten.add.Tensor(getitem_310, 1e-05);  getitem_310 = None
        rsqrt_68 = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        sub_84 = torch.ops.aten.sub.Tensor(clone_221, getitem_311);  clone_221 = getitem_311 = None
        mul_264 = torch.ops.aten.mul.Tensor(sub_84, rsqrt_68);  sub_84 = rsqrt_68 = None
        mul_265 = torch.ops.aten.mul.Tensor(mul_264, arg177_1);  mul_264 = arg177_1 = None
        add_293 = torch.ops.aten.add.Tensor(mul_265, arg178_1);  mul_265 = arg178_1 = None
        view_421 = torch.ops.aten.view.default(add_293, [1568, 384]);  add_293 = None
        permute_278 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg180_1, view_421, permute_278);  arg180_1 = view_421 = permute_278 = None
        view_422 = torch.ops.aten.view.default(addmm_101, [8, 14, 14, 1152]);  addmm_101 = None
        mul_266 = torch.ops.aten.mul.Tensor(view_422, 0.5)
        mul_267 = torch.ops.aten.mul.Tensor(view_422, 0.7071067811865476);  view_422 = None
        erf_33 = torch.ops.aten.erf.default(mul_267);  mul_267 = None
        add_294 = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_268 = torch.ops.aten.mul.Tensor(mul_266, add_294);  mul_266 = add_294 = None
        view_423 = torch.ops.aten.view.default(mul_268, [1568, 1152]);  mul_268 = None
        permute_279 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg182_1, view_423, permute_279);  arg182_1 = view_423 = permute_279 = None
        view_424 = torch.ops.aten.view.default(addmm_102, [8, 14, 14, 384]);  addmm_102 = None
        add_295 = torch.ops.aten.add.Tensor(add_291, view_424);  add_291 = view_424 = None
        clone_224 = torch.ops.aten.clone.default(add_295, memory_format = torch.contiguous_format)
        var_mean_69 = torch.ops.aten.var_mean.correction(clone_224, [3], correction = 0, keepdim = True)
        getitem_312 = var_mean_69[0]
        getitem_313 = var_mean_69[1];  var_mean_69 = None
        add_296 = torch.ops.aten.add.Tensor(getitem_312, 1e-05);  getitem_312 = None
        rsqrt_69 = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        sub_85 = torch.ops.aten.sub.Tensor(clone_224, getitem_313);  clone_224 = getitem_313 = None
        mul_269 = torch.ops.aten.mul.Tensor(sub_85, rsqrt_69);  sub_85 = rsqrt_69 = None
        mul_270 = torch.ops.aten.mul.Tensor(mul_269, arg183_1);  mul_269 = arg183_1 = None
        add_297 = torch.ops.aten.add.Tensor(mul_270, arg184_1);  mul_270 = arg184_1 = None
        permute_280 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        view_425 = torch.ops.aten.view.default(add_297, [1568, 384]);  add_297 = None
        mm_45 = torch.ops.aten.mm.default(view_425, permute_280);  view_425 = permute_280 = None
        view_426 = torch.ops.aten.view.default(mm_45, [8, 14, 14, 1152]);  mm_45 = None
        view_427 = torch.ops.aten.view.default(view_426, [8, 196, 3, 12, 32]);  view_426 = None
        permute_281 = torch.ops.aten.permute.default(view_427, [2, 0, 3, 1, 4]);  view_427 = None
        unbind_26 = torch.ops.aten.unbind.int(permute_281);  permute_281 = None
        getitem_314 = unbind_26[0]
        getitem_315 = unbind_26[1]
        getitem_316 = unbind_26[2];  unbind_26 = None
        _scaled_dot_product_efficient_attention_24 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_314, getitem_315, getitem_316, None, False);  getitem_314 = getitem_315 = getitem_316 = None
        getitem_317 = _scaled_dot_product_efficient_attention_24[0];  _scaled_dot_product_efficient_attention_24 = None
        permute_282 = torch.ops.aten.permute.default(getitem_317, [0, 2, 1, 3]);  getitem_317 = None
        view_428 = torch.ops.aten.view.default(permute_282, [8, 14, 14, 384]);  permute_282 = None
        view_429 = torch.ops.aten.view.default(view_428, [1568, 384]);  view_428 = None
        permute_283 = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg187_1, view_429, permute_283);  arg187_1 = view_429 = permute_283 = None
        view_430 = torch.ops.aten.view.default(addmm_103, [8, 14, 14, 384]);  addmm_103 = None
        add_298 = torch.ops.aten.add.Tensor(add_295, view_430);  add_295 = view_430 = None
        clone_226 = torch.ops.aten.clone.default(add_298, memory_format = torch.contiguous_format)
        var_mean_70 = torch.ops.aten.var_mean.correction(clone_226, [3], correction = 0, keepdim = True)
        getitem_321 = var_mean_70[0]
        getitem_322 = var_mean_70[1];  var_mean_70 = None
        add_299 = torch.ops.aten.add.Tensor(getitem_321, 1e-05);  getitem_321 = None
        rsqrt_70 = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
        sub_86 = torch.ops.aten.sub.Tensor(clone_226, getitem_322);  clone_226 = getitem_322 = None
        mul_271 = torch.ops.aten.mul.Tensor(sub_86, rsqrt_70);  sub_86 = rsqrt_70 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_271, arg188_1);  mul_271 = arg188_1 = None
        add_300 = torch.ops.aten.add.Tensor(mul_272, arg189_1);  mul_272 = arg189_1 = None
        view_431 = torch.ops.aten.view.default(add_300, [1568, 384]);  add_300 = None
        permute_284 = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg191_1, view_431, permute_284);  arg191_1 = view_431 = permute_284 = None
        view_432 = torch.ops.aten.view.default(addmm_104, [8, 14, 14, 1152]);  addmm_104 = None
        mul_273 = torch.ops.aten.mul.Tensor(view_432, 0.5)
        mul_274 = torch.ops.aten.mul.Tensor(view_432, 0.7071067811865476);  view_432 = None
        erf_34 = torch.ops.aten.erf.default(mul_274);  mul_274 = None
        add_301 = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_273, add_301);  mul_273 = add_301 = None
        view_433 = torch.ops.aten.view.default(mul_275, [1568, 1152]);  mul_275 = None
        permute_285 = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg193_1, view_433, permute_285);  arg193_1 = view_433 = permute_285 = None
        view_434 = torch.ops.aten.view.default(addmm_105, [8, 14, 14, 384]);  addmm_105 = None
        add_302 = torch.ops.aten.add.Tensor(add_298, view_434);  add_298 = view_434 = None
        clone_229 = torch.ops.aten.clone.default(add_302, memory_format = torch.contiguous_format)
        var_mean_71 = torch.ops.aten.var_mean.correction(clone_229, [3], correction = 0, keepdim = True)
        getitem_323 = var_mean_71[0]
        getitem_324 = var_mean_71[1];  var_mean_71 = None
        add_303 = torch.ops.aten.add.Tensor(getitem_323, 1e-05);  getitem_323 = None
        rsqrt_71 = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        sub_87 = torch.ops.aten.sub.Tensor(clone_229, getitem_324);  clone_229 = getitem_324 = None
        mul_276 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_71);  sub_87 = rsqrt_71 = None
        mul_277 = torch.ops.aten.mul.Tensor(mul_276, arg194_1);  mul_276 = arg194_1 = None
        add_304 = torch.ops.aten.add.Tensor(mul_277, arg195_1);  mul_277 = arg195_1 = None
        permute_286 = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        view_435 = torch.ops.aten.view.default(add_304, [1568, 384]);  add_304 = None
        mm_46 = torch.ops.aten.mm.default(view_435, permute_286);  view_435 = permute_286 = None
        view_436 = torch.ops.aten.view.default(mm_46, [8, 14, 14, 1152]);  mm_46 = None
        view_437 = torch.ops.aten.view.default(view_436, [8, 196, 3, 12, 32]);  view_436 = None
        permute_287 = torch.ops.aten.permute.default(view_437, [2, 0, 3, 1, 4]);  view_437 = None
        unbind_27 = torch.ops.aten.unbind.int(permute_287);  permute_287 = None
        getitem_325 = unbind_27[0]
        getitem_326 = unbind_27[1]
        getitem_327 = unbind_27[2];  unbind_27 = None
        _scaled_dot_product_efficient_attention_25 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_325, getitem_326, getitem_327, None, False);  getitem_325 = getitem_326 = getitem_327 = None
        getitem_328 = _scaled_dot_product_efficient_attention_25[0];  _scaled_dot_product_efficient_attention_25 = None
        permute_288 = torch.ops.aten.permute.default(getitem_328, [0, 2, 1, 3]);  getitem_328 = None
        view_438 = torch.ops.aten.view.default(permute_288, [8, 14, 14, 384]);  permute_288 = None
        view_439 = torch.ops.aten.view.default(view_438, [1568, 384]);  view_438 = None
        permute_289 = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg198_1, view_439, permute_289);  arg198_1 = view_439 = permute_289 = None
        view_440 = torch.ops.aten.view.default(addmm_106, [8, 14, 14, 384]);  addmm_106 = None
        add_305 = torch.ops.aten.add.Tensor(add_302, view_440);  add_302 = view_440 = None
        clone_231 = torch.ops.aten.clone.default(add_305, memory_format = torch.contiguous_format)
        var_mean_72 = torch.ops.aten.var_mean.correction(clone_231, [3], correction = 0, keepdim = True)
        getitem_332 = var_mean_72[0]
        getitem_333 = var_mean_72[1];  var_mean_72 = None
        add_306 = torch.ops.aten.add.Tensor(getitem_332, 1e-05);  getitem_332 = None
        rsqrt_72 = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        sub_88 = torch.ops.aten.sub.Tensor(clone_231, getitem_333);  clone_231 = getitem_333 = None
        mul_278 = torch.ops.aten.mul.Tensor(sub_88, rsqrt_72);  sub_88 = rsqrt_72 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_278, arg199_1);  mul_278 = arg199_1 = None
        add_307 = torch.ops.aten.add.Tensor(mul_279, arg200_1);  mul_279 = arg200_1 = None
        view_441 = torch.ops.aten.view.default(add_307, [1568, 384]);  add_307 = None
        permute_290 = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg202_1, view_441, permute_290);  arg202_1 = view_441 = permute_290 = None
        view_442 = torch.ops.aten.view.default(addmm_107, [8, 14, 14, 1152]);  addmm_107 = None
        mul_280 = torch.ops.aten.mul.Tensor(view_442, 0.5)
        mul_281 = torch.ops.aten.mul.Tensor(view_442, 0.7071067811865476);  view_442 = None
        erf_35 = torch.ops.aten.erf.default(mul_281);  mul_281 = None
        add_308 = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_282 = torch.ops.aten.mul.Tensor(mul_280, add_308);  mul_280 = add_308 = None
        view_443 = torch.ops.aten.view.default(mul_282, [1568, 1152]);  mul_282 = None
        permute_291 = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg204_1, view_443, permute_291);  arg204_1 = view_443 = permute_291 = None
        view_444 = torch.ops.aten.view.default(addmm_108, [8, 14, 14, 384]);  addmm_108 = None
        add_309 = torch.ops.aten.add.Tensor(add_305, view_444);  add_305 = view_444 = None
        clone_234 = torch.ops.aten.clone.default(add_309, memory_format = torch.contiguous_format)
        var_mean_73 = torch.ops.aten.var_mean.correction(clone_234, [3], correction = 0, keepdim = True)
        getitem_334 = var_mean_73[0]
        getitem_335 = var_mean_73[1];  var_mean_73 = None
        add_310 = torch.ops.aten.add.Tensor(getitem_334, 1e-05);  getitem_334 = None
        rsqrt_73 = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        sub_89 = torch.ops.aten.sub.Tensor(clone_234, getitem_335);  clone_234 = getitem_335 = None
        mul_283 = torch.ops.aten.mul.Tensor(sub_89, rsqrt_73);  sub_89 = rsqrt_73 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_283, arg205_1);  mul_283 = arg205_1 = None
        add_311 = torch.ops.aten.add.Tensor(mul_284, arg206_1);  mul_284 = arg206_1 = None
        permute_292 = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        view_445 = torch.ops.aten.view.default(add_311, [1568, 384]);  add_311 = None
        mm_47 = torch.ops.aten.mm.default(view_445, permute_292);  view_445 = permute_292 = None
        view_446 = torch.ops.aten.view.default(mm_47, [8, 14, 14, 1152]);  mm_47 = None
        view_447 = torch.ops.aten.view.default(view_446, [8, 196, 3, 12, 32]);  view_446 = None
        permute_293 = torch.ops.aten.permute.default(view_447, [2, 0, 3, 1, 4]);  view_447 = None
        unbind_28 = torch.ops.aten.unbind.int(permute_293);  permute_293 = None
        getitem_336 = unbind_28[0]
        getitem_337 = unbind_28[1]
        getitem_338 = unbind_28[2];  unbind_28 = None
        _scaled_dot_product_efficient_attention_26 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_336, getitem_337, getitem_338, None, False);  getitem_336 = getitem_337 = getitem_338 = None
        getitem_339 = _scaled_dot_product_efficient_attention_26[0];  _scaled_dot_product_efficient_attention_26 = None
        permute_294 = torch.ops.aten.permute.default(getitem_339, [0, 2, 1, 3]);  getitem_339 = None
        view_448 = torch.ops.aten.view.default(permute_294, [8, 14, 14, 384]);  permute_294 = None
        view_449 = torch.ops.aten.view.default(view_448, [1568, 384]);  view_448 = None
        permute_295 = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg209_1, view_449, permute_295);  arg209_1 = view_449 = permute_295 = None
        view_450 = torch.ops.aten.view.default(addmm_109, [8, 14, 14, 384]);  addmm_109 = None
        add_312 = torch.ops.aten.add.Tensor(add_309, view_450);  add_309 = view_450 = None
        clone_236 = torch.ops.aten.clone.default(add_312, memory_format = torch.contiguous_format)
        var_mean_74 = torch.ops.aten.var_mean.correction(clone_236, [3], correction = 0, keepdim = True)
        getitem_343 = var_mean_74[0]
        getitem_344 = var_mean_74[1];  var_mean_74 = None
        add_313 = torch.ops.aten.add.Tensor(getitem_343, 1e-05);  getitem_343 = None
        rsqrt_74 = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        sub_90 = torch.ops.aten.sub.Tensor(clone_236, getitem_344);  clone_236 = getitem_344 = None
        mul_285 = torch.ops.aten.mul.Tensor(sub_90, rsqrt_74);  sub_90 = rsqrt_74 = None
        mul_286 = torch.ops.aten.mul.Tensor(mul_285, arg210_1);  mul_285 = arg210_1 = None
        add_314 = torch.ops.aten.add.Tensor(mul_286, arg211_1);  mul_286 = arg211_1 = None
        view_451 = torch.ops.aten.view.default(add_314, [1568, 384]);  add_314 = None
        permute_296 = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg213_1, view_451, permute_296);  arg213_1 = view_451 = permute_296 = None
        view_452 = torch.ops.aten.view.default(addmm_110, [8, 14, 14, 1152]);  addmm_110 = None
        mul_287 = torch.ops.aten.mul.Tensor(view_452, 0.5)
        mul_288 = torch.ops.aten.mul.Tensor(view_452, 0.7071067811865476);  view_452 = None
        erf_36 = torch.ops.aten.erf.default(mul_288);  mul_288 = None
        add_315 = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_289 = torch.ops.aten.mul.Tensor(mul_287, add_315);  mul_287 = add_315 = None
        view_453 = torch.ops.aten.view.default(mul_289, [1568, 1152]);  mul_289 = None
        permute_297 = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg215_1, view_453, permute_297);  arg215_1 = view_453 = permute_297 = None
        view_454 = torch.ops.aten.view.default(addmm_111, [8, 14, 14, 384]);  addmm_111 = None
        add_316 = torch.ops.aten.add.Tensor(add_312, view_454);  add_312 = view_454 = None
        clone_239 = torch.ops.aten.clone.default(add_316, memory_format = torch.contiguous_format)
        var_mean_75 = torch.ops.aten.var_mean.correction(clone_239, [3], correction = 0, keepdim = True)
        getitem_345 = var_mean_75[0]
        getitem_346 = var_mean_75[1];  var_mean_75 = None
        add_317 = torch.ops.aten.add.Tensor(getitem_345, 1e-05);  getitem_345 = None
        rsqrt_75 = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        sub_91 = torch.ops.aten.sub.Tensor(clone_239, getitem_346);  clone_239 = getitem_346 = None
        mul_290 = torch.ops.aten.mul.Tensor(sub_91, rsqrt_75);  sub_91 = rsqrt_75 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, arg216_1);  mul_290 = arg216_1 = None
        add_318 = torch.ops.aten.add.Tensor(mul_291, arg217_1);  mul_291 = arg217_1 = None
        permute_298 = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        view_455 = torch.ops.aten.view.default(add_318, [1568, 384]);  add_318 = None
        mm_48 = torch.ops.aten.mm.default(view_455, permute_298);  view_455 = permute_298 = None
        view_456 = torch.ops.aten.view.default(mm_48, [8, 14, 14, 1152]);  mm_48 = None
        view_457 = torch.ops.aten.view.default(view_456, [8, 196, 3, 12, 32]);  view_456 = None
        permute_299 = torch.ops.aten.permute.default(view_457, [2, 0, 3, 1, 4]);  view_457 = None
        unbind_29 = torch.ops.aten.unbind.int(permute_299);  permute_299 = None
        getitem_347 = unbind_29[0]
        getitem_348 = unbind_29[1]
        getitem_349 = unbind_29[2];  unbind_29 = None
        _scaled_dot_product_efficient_attention_27 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_347, getitem_348, getitem_349, None, False);  getitem_347 = getitem_348 = getitem_349 = None
        getitem_350 = _scaled_dot_product_efficient_attention_27[0];  _scaled_dot_product_efficient_attention_27 = None
        permute_300 = torch.ops.aten.permute.default(getitem_350, [0, 2, 1, 3]);  getitem_350 = None
        view_458 = torch.ops.aten.view.default(permute_300, [8, 14, 14, 384]);  permute_300 = None
        view_459 = torch.ops.aten.view.default(view_458, [1568, 384]);  view_458 = None
        permute_301 = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg220_1, view_459, permute_301);  arg220_1 = view_459 = permute_301 = None
        view_460 = torch.ops.aten.view.default(addmm_112, [8, 14, 14, 384]);  addmm_112 = None
        add_319 = torch.ops.aten.add.Tensor(add_316, view_460);  add_316 = view_460 = None
        clone_241 = torch.ops.aten.clone.default(add_319, memory_format = torch.contiguous_format)
        var_mean_76 = torch.ops.aten.var_mean.correction(clone_241, [3], correction = 0, keepdim = True)
        getitem_354 = var_mean_76[0]
        getitem_355 = var_mean_76[1];  var_mean_76 = None
        add_320 = torch.ops.aten.add.Tensor(getitem_354, 1e-05);  getitem_354 = None
        rsqrt_76 = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        sub_92 = torch.ops.aten.sub.Tensor(clone_241, getitem_355);  clone_241 = getitem_355 = None
        mul_292 = torch.ops.aten.mul.Tensor(sub_92, rsqrt_76);  sub_92 = rsqrt_76 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_292, arg221_1);  mul_292 = arg221_1 = None
        add_321 = torch.ops.aten.add.Tensor(mul_293, arg222_1);  mul_293 = arg222_1 = None
        view_461 = torch.ops.aten.view.default(add_321, [1568, 384]);  add_321 = None
        permute_302 = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg224_1, view_461, permute_302);  arg224_1 = view_461 = permute_302 = None
        view_462 = torch.ops.aten.view.default(addmm_113, [8, 14, 14, 1152]);  addmm_113 = None
        mul_294 = torch.ops.aten.mul.Tensor(view_462, 0.5)
        mul_295 = torch.ops.aten.mul.Tensor(view_462, 0.7071067811865476);  view_462 = None
        erf_37 = torch.ops.aten.erf.default(mul_295);  mul_295 = None
        add_322 = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_296 = torch.ops.aten.mul.Tensor(mul_294, add_322);  mul_294 = add_322 = None
        view_463 = torch.ops.aten.view.default(mul_296, [1568, 1152]);  mul_296 = None
        permute_303 = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg226_1, view_463, permute_303);  arg226_1 = view_463 = permute_303 = None
        view_464 = torch.ops.aten.view.default(addmm_114, [8, 14, 14, 384]);  addmm_114 = None
        add_323 = torch.ops.aten.add.Tensor(add_319, view_464);  add_319 = view_464 = None
        view_465 = torch.ops.aten.view.default(add_323, [8, 196, 384]);  add_323 = None
        expand_25 = torch.ops.aten.expand.default(arg227_1, [8, -1, -1]);  arg227_1 = None
        cat_3 = torch.ops.aten.cat.default([expand_25, view_465], 1);  expand_25 = view_465 = None
        slice_35 = torch.ops.aten.slice.Tensor(cat_3, 1, 0, 1)
        var_mean_77 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
        getitem_356 = var_mean_77[0]
        getitem_357 = var_mean_77[1];  var_mean_77 = None
        add_324 = torch.ops.aten.add.Tensor(getitem_356, 1e-05);  getitem_356 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        sub_93 = torch.ops.aten.sub.Tensor(cat_3, getitem_357);  getitem_357 = None
        mul_297 = torch.ops.aten.mul.Tensor(sub_93, rsqrt_77);  sub_93 = rsqrt_77 = None
        mul_298 = torch.ops.aten.mul.Tensor(mul_297, arg228_1);  mul_297 = arg228_1 = None
        add_325 = torch.ops.aten.add.Tensor(mul_298, arg229_1);  mul_298 = arg229_1 = None
        permute_304 = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
        view_466 = torch.ops.aten.view.default(add_325, [1576, 384])
        mm_49 = torch.ops.aten.mm.default(view_466, permute_304);  view_466 = permute_304 = None
        view_467 = torch.ops.aten.view.default(mm_49, [8, 197, 768]);  mm_49 = None
        view_468 = torch.ops.aten.view.default(view_467, [8, 197, 2, 12, 32]);  view_467 = None
        permute_305 = torch.ops.aten.permute.default(view_468, [2, 0, 3, 1, 4]);  view_468 = None
        unbind_30 = torch.ops.aten.unbind.int(permute_305);  permute_305 = None
        getitem_358 = unbind_30[0]
        getitem_359 = unbind_30[1];  unbind_30 = None
        slice_37 = torch.ops.aten.slice.Tensor(add_325, 1, 0, 1);  add_325 = None
        permute_306 = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        view_469 = torch.ops.aten.view.default(slice_37, [8, 384]);  slice_37 = None
        mm_50 = torch.ops.aten.mm.default(view_469, permute_306);  view_469 = permute_306 = None
        view_470 = torch.ops.aten.view.default(mm_50, [8, 1, 384]);  mm_50 = None
        view_471 = torch.ops.aten.view.default(view_470, [8, 12, 1, 32]);  view_470 = None
        mul_299 = torch.ops.aten.mul.Tensor(view_471, 0.1767766952966369);  view_471 = None
        permute_307 = torch.ops.aten.permute.default(getitem_358, [0, 1, 3, 2]);  getitem_358 = None
        expand_26 = torch.ops.aten.expand.default(mul_299, [8, 12, 1, 32]);  mul_299 = None
        view_472 = torch.ops.aten.view.default(expand_26, [96, 1, 32]);  expand_26 = None
        expand_27 = torch.ops.aten.expand.default(permute_307, [8, 12, 32, 197]);  permute_307 = None
        clone_244 = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_473 = torch.ops.aten.view.default(clone_244, [96, 32, 197]);  clone_244 = None
        bmm_12 = torch.ops.aten.bmm.default(view_472, view_473);  view_472 = view_473 = None
        view_474 = torch.ops.aten.view.default(bmm_12, [8, 12, 1, 197]);  bmm_12 = None
        amax_10 = torch.ops.aten.amax.default(view_474, [-1], True)
        sub_94 = torch.ops.aten.sub.Tensor(view_474, amax_10);  view_474 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_94);  sub_94 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        expand_28 = torch.ops.aten.expand.default(div_10, [8, 12, 1, 197]);  div_10 = None
        view_475 = torch.ops.aten.view.default(expand_28, [96, 1, 197]);  expand_28 = None
        expand_29 = torch.ops.aten.expand.default(getitem_359, [8, 12, 197, 32]);  getitem_359 = None
        clone_246 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_476 = torch.ops.aten.view.default(clone_246, [96, 197, 32]);  clone_246 = None
        bmm_13 = torch.ops.aten.bmm.default(view_475, view_476);  view_475 = view_476 = None
        view_477 = torch.ops.aten.view.default(bmm_13, [8, 12, 1, 32]);  bmm_13 = None
        permute_308 = torch.ops.aten.permute.default(view_477, [0, 2, 1, 3]);  view_477 = None
        view_478 = torch.ops.aten.view.default(permute_308, [8, 1, 384]);  permute_308 = None
        view_479 = torch.ops.aten.view.default(view_478, [8, 384]);  view_478 = None
        permute_309 = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg233_1, view_479, permute_309);  arg233_1 = view_479 = permute_309 = None
        view_480 = torch.ops.aten.view.default(addmm_115, [8, 1, 384]);  addmm_115 = None
        add_326 = torch.ops.aten.add.Tensor(slice_35, view_480);  slice_35 = view_480 = None
        var_mean_78 = torch.ops.aten.var_mean.correction(add_326, [2], correction = 0, keepdim = True)
        getitem_360 = var_mean_78[0]
        getitem_361 = var_mean_78[1];  var_mean_78 = None
        add_327 = torch.ops.aten.add.Tensor(getitem_360, 1e-05);  getitem_360 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
        sub_95 = torch.ops.aten.sub.Tensor(add_326, getitem_361);  getitem_361 = None
        mul_300 = torch.ops.aten.mul.Tensor(sub_95, rsqrt_78);  sub_95 = rsqrt_78 = None
        mul_301 = torch.ops.aten.mul.Tensor(mul_300, arg234_1);  mul_300 = arg234_1 = None
        add_328 = torch.ops.aten.add.Tensor(mul_301, arg235_1);  mul_301 = arg235_1 = None
        view_481 = torch.ops.aten.view.default(add_328, [8, 384]);  add_328 = None
        permute_310 = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg237_1, view_481, permute_310);  arg237_1 = view_481 = permute_310 = None
        view_482 = torch.ops.aten.view.default(addmm_116, [8, 1, 1152]);  addmm_116 = None
        mul_302 = torch.ops.aten.mul.Tensor(view_482, 0.5)
        mul_303 = torch.ops.aten.mul.Tensor(view_482, 0.7071067811865476);  view_482 = None
        erf_38 = torch.ops.aten.erf.default(mul_303);  mul_303 = None
        add_329 = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_304 = torch.ops.aten.mul.Tensor(mul_302, add_329);  mul_302 = add_329 = None
        view_483 = torch.ops.aten.view.default(mul_304, [8, 1152]);  mul_304 = None
        permute_311 = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg239_1, view_483, permute_311);  arg239_1 = view_483 = permute_311 = None
        view_484 = torch.ops.aten.view.default(addmm_117, [8, 1, 384]);  addmm_117 = None
        add_330 = torch.ops.aten.add.Tensor(add_326, view_484);  add_326 = view_484 = None
        slice_40 = torch.ops.aten.slice.Tensor(cat_3, 1, 1, 9223372036854775807);  cat_3 = None
        cat_4 = torch.ops.aten.cat.default([add_330, slice_40], 1);  add_330 = slice_40 = None
        slice_42 = torch.ops.aten.slice.Tensor(cat_4, 1, 0, 1)
        var_mean_79 = torch.ops.aten.var_mean.correction(cat_4, [2], correction = 0, keepdim = True)
        getitem_362 = var_mean_79[0]
        getitem_363 = var_mean_79[1];  var_mean_79 = None
        add_331 = torch.ops.aten.add.Tensor(getitem_362, 1e-05);  getitem_362 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        sub_96 = torch.ops.aten.sub.Tensor(cat_4, getitem_363);  getitem_363 = None
        mul_305 = torch.ops.aten.mul.Tensor(sub_96, rsqrt_79);  sub_96 = rsqrt_79 = None
        mul_306 = torch.ops.aten.mul.Tensor(mul_305, arg240_1);  mul_305 = arg240_1 = None
        add_332 = torch.ops.aten.add.Tensor(mul_306, arg241_1);  mul_306 = arg241_1 = None
        permute_312 = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        view_485 = torch.ops.aten.view.default(add_332, [1576, 384])
        mm_51 = torch.ops.aten.mm.default(view_485, permute_312);  view_485 = permute_312 = None
        view_486 = torch.ops.aten.view.default(mm_51, [8, 197, 768]);  mm_51 = None
        view_487 = torch.ops.aten.view.default(view_486, [8, 197, 2, 12, 32]);  view_486 = None
        permute_313 = torch.ops.aten.permute.default(view_487, [2, 0, 3, 1, 4]);  view_487 = None
        unbind_31 = torch.ops.aten.unbind.int(permute_313);  permute_313 = None
        getitem_364 = unbind_31[0]
        getitem_365 = unbind_31[1];  unbind_31 = None
        slice_44 = torch.ops.aten.slice.Tensor(add_332, 1, 0, 1);  add_332 = None
        permute_314 = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        view_488 = torch.ops.aten.view.default(slice_44, [8, 384]);  slice_44 = None
        mm_52 = torch.ops.aten.mm.default(view_488, permute_314);  view_488 = permute_314 = None
        view_489 = torch.ops.aten.view.default(mm_52, [8, 1, 384]);  mm_52 = None
        view_490 = torch.ops.aten.view.default(view_489, [8, 12, 1, 32]);  view_489 = None
        mul_307 = torch.ops.aten.mul.Tensor(view_490, 0.1767766952966369);  view_490 = None
        permute_315 = torch.ops.aten.permute.default(getitem_364, [0, 1, 3, 2]);  getitem_364 = None
        expand_30 = torch.ops.aten.expand.default(mul_307, [8, 12, 1, 32]);  mul_307 = None
        view_491 = torch.ops.aten.view.default(expand_30, [96, 1, 32]);  expand_30 = None
        expand_31 = torch.ops.aten.expand.default(permute_315, [8, 12, 32, 197]);  permute_315 = None
        clone_250 = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_492 = torch.ops.aten.view.default(clone_250, [96, 32, 197]);  clone_250 = None
        bmm_14 = torch.ops.aten.bmm.default(view_491, view_492);  view_491 = view_492 = None
        view_493 = torch.ops.aten.view.default(bmm_14, [8, 12, 1, 197]);  bmm_14 = None
        amax_11 = torch.ops.aten.amax.default(view_493, [-1], True)
        sub_97 = torch.ops.aten.sub.Tensor(view_493, amax_11);  view_493 = amax_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_97);  sub_97 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        expand_32 = torch.ops.aten.expand.default(div_11, [8, 12, 1, 197]);  div_11 = None
        view_494 = torch.ops.aten.view.default(expand_32, [96, 1, 197]);  expand_32 = None
        expand_33 = torch.ops.aten.expand.default(getitem_365, [8, 12, 197, 32]);  getitem_365 = None
        clone_252 = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_495 = torch.ops.aten.view.default(clone_252, [96, 197, 32]);  clone_252 = None
        bmm_15 = torch.ops.aten.bmm.default(view_494, view_495);  view_494 = view_495 = None
        view_496 = torch.ops.aten.view.default(bmm_15, [8, 12, 1, 32]);  bmm_15 = None
        permute_316 = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
        view_497 = torch.ops.aten.view.default(permute_316, [8, 1, 384]);  permute_316 = None
        view_498 = torch.ops.aten.view.default(view_497, [8, 384]);  view_497 = None
        permute_317 = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg245_1, view_498, permute_317);  arg245_1 = view_498 = permute_317 = None
        view_499 = torch.ops.aten.view.default(addmm_118, [8, 1, 384]);  addmm_118 = None
        add_333 = torch.ops.aten.add.Tensor(slice_42, view_499);  slice_42 = view_499 = None
        var_mean_80 = torch.ops.aten.var_mean.correction(add_333, [2], correction = 0, keepdim = True)
        getitem_366 = var_mean_80[0]
        getitem_367 = var_mean_80[1];  var_mean_80 = None
        add_334 = torch.ops.aten.add.Tensor(getitem_366, 1e-05);  getitem_366 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        sub_98 = torch.ops.aten.sub.Tensor(add_333, getitem_367);  getitem_367 = None
        mul_308 = torch.ops.aten.mul.Tensor(sub_98, rsqrt_80);  sub_98 = rsqrt_80 = None
        mul_309 = torch.ops.aten.mul.Tensor(mul_308, arg246_1);  mul_308 = arg246_1 = None
        add_335 = torch.ops.aten.add.Tensor(mul_309, arg247_1);  mul_309 = arg247_1 = None
        view_500 = torch.ops.aten.view.default(add_335, [8, 384]);  add_335 = None
        permute_318 = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg249_1, view_500, permute_318);  arg249_1 = view_500 = permute_318 = None
        view_501 = torch.ops.aten.view.default(addmm_119, [8, 1, 1152]);  addmm_119 = None
        mul_310 = torch.ops.aten.mul.Tensor(view_501, 0.5)
        mul_311 = torch.ops.aten.mul.Tensor(view_501, 0.7071067811865476);  view_501 = None
        erf_39 = torch.ops.aten.erf.default(mul_311);  mul_311 = None
        add_336 = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_312 = torch.ops.aten.mul.Tensor(mul_310, add_336);  mul_310 = add_336 = None
        view_502 = torch.ops.aten.view.default(mul_312, [8, 1152]);  mul_312 = None
        permute_319 = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg251_1, view_502, permute_319);  arg251_1 = view_502 = permute_319 = None
        view_503 = torch.ops.aten.view.default(addmm_120, [8, 1, 384]);  addmm_120 = None
        add_337 = torch.ops.aten.add.Tensor(add_333, view_503);  add_333 = view_503 = None
        slice_47 = torch.ops.aten.slice.Tensor(cat_4, 1, 1, 9223372036854775807);  cat_4 = None
        cat_5 = torch.ops.aten.cat.default([add_337, slice_47], 1);  add_337 = slice_47 = None
        var_mean_81 = torch.ops.aten.var_mean.correction(cat_5, [2], correction = 0, keepdim = True)
        getitem_368 = var_mean_81[0]
        getitem_369 = var_mean_81[1];  var_mean_81 = None
        add_338 = torch.ops.aten.add.Tensor(getitem_368, 1e-05);  getitem_368 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
        sub_99 = torch.ops.aten.sub.Tensor(cat_5, getitem_369);  cat_5 = getitem_369 = None
        mul_313 = torch.ops.aten.mul.Tensor(sub_99, rsqrt_81);  sub_99 = rsqrt_81 = None
        mul_314 = torch.ops.aten.mul.Tensor(mul_313, arg252_1);  mul_313 = arg252_1 = None
        add_339 = torch.ops.aten.add.Tensor(mul_314, arg253_1);  mul_314 = arg253_1 = None
        select_1 = torch.ops.aten.select.int(add_339, 1, 0)
        permute_320 = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg255_1, select_1, permute_320);  arg255_1 = select_1 = permute_320 = None
        slice_50 = torch.ops.aten.slice.Tensor(add_339, 1, 1, 9223372036854775807);  add_339 = None
        permute_321 = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
        clone_257 = torch.ops.aten.clone.default(slice_50, memory_format = torch.contiguous_format);  slice_50 = None
        view_504 = torch.ops.aten.view.default(clone_257, [1568, 384]);  clone_257 = None
        mm_53 = torch.ops.aten.mm.default(view_504, permute_321);  view_504 = permute_321 = None
        view_505 = torch.ops.aten.view.default(mm_53, [8, 196, 1000]);  mm_53 = None
        add_340 = torch.ops.aten.add.Tensor(view_505, arg257_1);  view_505 = arg257_1 = None
        max_2 = torch.ops.aten.max.dim(add_340, 1);  add_340 = None
        getitem_370 = max_2[0];  max_2 = None
        mul_315 = torch.ops.aten.mul.Tensor(getitem_370, 0.5);  getitem_370 = None
        add_341 = torch.ops.aten.add.Tensor(addmm_121, mul_315);  addmm_121 = mul_315 = None
        return (add_341,)
        
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
    buf6 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf6, (64, 64, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf9, (64,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf10, (64,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf11, (64, 64, 3, 3), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf12, (64,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf13, (64,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf14, (64,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf15, (64,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf16, (192, 64, 4, 4), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf17, (192,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf18, (192,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf19, (192,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf20, (192, 192), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 373248, device=device(type='cuda', index=0))
    reader.tensor(buf21, (486, 192), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1944, device=device(type='cuda', index=0))
    reader.tensor(buf22, (486,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf23, (192, 192), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf24, (192,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf25, (192,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf26, (192,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf27, (576, 192), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf28, (576,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf29, (192, 576), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf30, (192,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf31, (192,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf32, (192,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf33, (192, 192), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 373248, device=device(type='cuda', index=0))
    reader.tensor(buf34, (486, 192), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1944, device=device(type='cuda', index=0))
    reader.tensor(buf35, (486,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf36, (192, 192), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf37, (192,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf38, (192,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf39, (192,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf40, (576, 192), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf41, (576,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf42, (192, 576), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf43, (192,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf44, (192,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf45, (192,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf46, (192, 192), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 373248, device=device(type='cuda', index=0))
    reader.tensor(buf47, (486, 192), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 1944, device=device(type='cuda', index=0))
    reader.tensor(buf48, (486,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf49, (192, 192), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf50, (192,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf51, (192,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf52, (192,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf53, (576, 192), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf54, (576,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf55, (192, 576), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf56, (192,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf57, (192,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf58, (192,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf59, (192, 192), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 373248, device=device(type='cuda', index=0))
    reader.tensor(buf60, (486, 192), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 1944, device=device(type='cuda', index=0))
    reader.tensor(buf61, (486,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf62, (192, 192), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf63, (192,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf64, (192,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf65, (192,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf66, (576, 192), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf67, (576,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf68, (192, 576), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf69, (192,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf70, (384, 192, 2, 2), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf71, (384,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf72, (1, 14, 14, 384), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf73, (384,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf74, (384,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf75, (1152, 384), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf76, (384, 384), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf77, (384,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf78, (384,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf79, (384,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf80, (1152, 384), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf81, (1152,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf82, (384, 1152), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf83, (384,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf84, (384,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf85, (384,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf86, (1152, 384), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf87, (384, 384), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf88, (384,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf89, (384,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf90, (384,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf91, (1152, 384), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf92, (1152,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf93, (384, 1152), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf94, (384,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf95, (384,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf96, (384,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf97, (1152, 384), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf98, (384, 384), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf99, (384,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf100, (384,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf101, (384,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf102, (1152, 384), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf103, (1152,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf104, (384, 1152), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf105, (384,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf106, (384,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf107, (384,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf108, (1152, 384), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf109, (384, 384), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf110, (384,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf111, (384,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf112, (384,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf113, (1152, 384), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf114, (1152,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf115, (384, 1152), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf116, (384,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf117, (384,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf118, (384,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf119, (1152, 384), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf120, (384, 384), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf121, (384,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf122, (384,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf123, (384,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf124, (1152, 384), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf125, (1152,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf126, (384, 1152), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf127, (384,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf128, (384,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf129, (384,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf130, (1152, 384), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf131, (384, 384), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf132, (384,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf133, (384,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf134, (384,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf135, (1152, 384), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf136, (1152,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf137, (384, 1152), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf138, (384,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf139, (384,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf140, (384,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf141, (1152, 384), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf142, (384, 384), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf143, (384,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf144, (384,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf145, (384,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf146, (1152, 384), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf147, (1152,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf148, (384, 1152), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf149, (384,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf150, (384,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf151, (384,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf152, (1152, 384), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf153, (384, 384), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf154, (384,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf155, (384,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf156, (384,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf157, (1152, 384), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf158, (1152,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf159, (384, 1152), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf160, (384,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf161, (384,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf162, (384,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf163, (1152, 384), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf164, (384, 384), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf165, (384,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf166, (384,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf167, (384,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf168, (1152, 384), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf169, (1152,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf170, (384, 1152), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf171, (384,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf172, (384,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf173, (384,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf174, (1152, 384), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf175, (384, 384), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf176, (384,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf177, (384,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf178, (384,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf179, (1152, 384), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf180, (1152,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf181, (384, 1152), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf182, (384,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf183, (384,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf184, (384,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf185, (1152, 384), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf186, (384, 384), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf187, (384,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf188, (384,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf189, (384,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf190, (1152, 384), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf191, (1152,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf192, (384, 1152), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf193, (384,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf194, (384,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf195, (384,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf196, (1152, 384), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf197, (384, 384), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf198, (384,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf199, (384,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf200, (384,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf201, (1152, 384), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf202, (1152,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf203, (384, 1152), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf204, (384,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf205, (384,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf206, (384,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf207, (1152, 384), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf208, (384, 384), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf209, (384,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf210, (384,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf211, (384,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf212, (1152, 384), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf213, (1152,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf214, (384, 1152), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf215, (384,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf216, (384,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf217, (384,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf218, (1152, 384), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf219, (384, 384), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf220, (384,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf221, (384,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf222, (384,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf223, (1152, 384), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf224, (1152,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf225, (384, 1152), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf226, (384,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf227, (1, 1, 384), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf228, (384,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf229, (384,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf230, (768, 384), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf231, (384, 384), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf232, (384, 384), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf233, (384,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf234, (384,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf235, (384,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf236, (1152, 384), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf237, (1152,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf238, (384, 1152), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf239, (384,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf240, (384,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf241, (384,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf242, (768, 384), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf243, (384, 384), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf244, (384, 384), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf245, (384,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf246, (384,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf247, (384,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf248, (1152, 384), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf249, (1152,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf250, (384, 1152), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf251, (384,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf252, (384,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf253, (384,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 1536000, device=device(type='cuda', index=0))
    reader.tensor(buf254, (1000, 384), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf255, (1000,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 1536000, device=device(type='cuda', index=0))
    reader.tensor(buf256, (1000, 384), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf257, (1000,), is_leaf=True)  # arg257_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)