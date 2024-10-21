
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1):
        convolution_36 = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        view_168 = torch.ops.aten.view.default(convolution_36, [8, 64, 3136]);  convolution_36 = None
        permute_97 = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        clone_65 = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(clone_65, [2], correction = 0, keepdim = True)
        getitem_90 = var_mean_21[0]
        getitem_91 = var_mean_21[1];  var_mean_21 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_29 = torch.ops.aten.sub.Tensor(clone_65, getitem_91);  clone_65 = getitem_91 = None
        mul_82 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_21);  sub_29 = rsqrt_21 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, arg3_1);  mul_82 = arg3_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_83, arg4_1);  mul_83 = arg4_1 = None
        expand_36 = torch.ops.aten.expand.default(arg5_1, [8, -1, -1]);  arg5_1 = None
        cat_20 = torch.ops.aten.cat.default([expand_36, add_83], 1);  expand_36 = add_83 = None
        slice_111 = torch.ops.aten.slice.Tensor(cat_20, 1, 0, 1)
        slice_113 = torch.ops.aten.slice.Tensor(cat_20, 1, 1, 9223372036854775807);  cat_20 = None
        permute_98 = torch.ops.aten.permute.default(slice_113, [0, 2, 1]);  slice_113 = None
        view_169 = torch.ops.aten.view.default(permute_98, [8, 64, 56, 56]);  permute_98 = None
        convolution_37 = torch.ops.aten.convolution.default(view_169, arg6_1, arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 64)
        add_84 = torch.ops.aten.add.Tensor(convolution_37, view_169);  convolution_37 = view_169 = None
        view_170 = torch.ops.aten.view.default(add_84, [8, 64, 3136]);  add_84 = None
        permute_99 = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
        cat_21 = torch.ops.aten.cat.default([slice_111, permute_99], 1);  slice_111 = permute_99 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(cat_21, [2], correction = 0, keepdim = True)
        getitem_92 = var_mean_22[0]
        getitem_93 = var_mean_22[1];  var_mean_22 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_30 = torch.ops.aten.sub.Tensor(cat_21, getitem_93);  getitem_93 = None
        mul_84 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_22);  sub_30 = rsqrt_22 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, arg8_1);  mul_84 = arg8_1 = None
        add_86 = torch.ops.aten.add.Tensor(mul_85, arg9_1);  mul_85 = arg9_1 = None
        view_171 = torch.ops.aten.view.default(add_86, [25096, 64]);  add_86 = None
        permute_100 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg11_1, view_171, permute_100);  arg11_1 = view_171 = permute_100 = None
        view_172 = torch.ops.aten.view.default(addmm_33, [8, 3137, 192]);  addmm_33 = None
        view_173 = torch.ops.aten.view.default(view_172, [8, 3137, 3, 8, 8]);  view_172 = None
        permute_101 = torch.ops.aten.permute.default(view_173, [2, 0, 3, 1, 4]);  view_173 = None
        unbind_8 = torch.ops.aten.unbind.int(permute_101);  permute_101 = None
        getitem_94 = unbind_8[0]
        getitem_95 = unbind_8[1]
        getitem_96 = unbind_8[2];  unbind_8 = None
        clone_66 = torch.ops.aten.clone.default(getitem_95, memory_format = torch.contiguous_format);  getitem_95 = None
        amax_8 = torch.ops.aten.amax.default(clone_66, [2], True)
        sub_31 = torch.ops.aten.sub.Tensor(clone_66, amax_8);  clone_66 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [2], True)
        div_8 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        permute_102 = torch.ops.aten.permute.default(div_8, [0, 1, 3, 2]);  div_8 = None
        expand_37 = torch.ops.aten.expand.default(permute_102, [8, 8, 8, 3137]);  permute_102 = None
        view_174 = torch.ops.aten.view.default(expand_37, [64, 8, 3137]);  expand_37 = None
        expand_38 = torch.ops.aten.expand.default(getitem_96, [8, 8, 3137, 8])
        clone_67 = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_175 = torch.ops.aten.view.default(clone_67, [64, 3137, 8]);  clone_67 = None
        bmm_16 = torch.ops.aten.bmm.default(view_174, view_175);  view_174 = view_175 = None
        view_176 = torch.ops.aten.view.default(bmm_16, [8, 8, 8, 8]);  bmm_16 = None
        expand_39 = torch.ops.aten.expand.default(getitem_94, [8, 8, 3137, 8])
        clone_68 = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_177 = torch.ops.aten.view.default(clone_68, [64, 3137, 8]);  clone_68 = None
        expand_40 = torch.ops.aten.expand.default(view_176, [8, 8, 8, 8]);  view_176 = None
        view_178 = torch.ops.aten.view.default(expand_40, [64, 8, 8]);  expand_40 = None
        bmm_17 = torch.ops.aten.bmm.default(view_177, view_178);  view_177 = view_178 = None
        view_179 = torch.ops.aten.view.default(bmm_17, [8, 8, 3137, 8]);  bmm_17 = None
        slice_116 = torch.ops.aten.slice.Tensor(getitem_94, 2, 1, 9223372036854775807);  getitem_94 = None
        slice_120 = torch.ops.aten.slice.Tensor(getitem_96, 2, 1, 9223372036854775807);  getitem_96 = None
        permute_103 = torch.ops.aten.permute.default(slice_120, [0, 1, 3, 2]);  slice_120 = None
        view_180 = torch.ops.aten.view.default(permute_103, [8, 64, 56, 56]);  permute_103 = None
        split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(view_180, [16, 24, 24], 1);  view_180 = None
        getitem_97 = split_with_sizes_8[0]
        getitem_98 = split_with_sizes_8[1]
        getitem_99 = split_with_sizes_8[2];  split_with_sizes_8 = None
        convolution_38 = torch.ops.aten.convolution.default(getitem_97, arg12_1, arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  getitem_97 = None
        convolution_39 = torch.ops.aten.convolution.default(getitem_98, arg14_1, arg15_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 24);  getitem_98 = None
        convolution_40 = torch.ops.aten.convolution.default(getitem_99, arg16_1, arg17_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 24);  getitem_99 = None
        cat_22 = torch.ops.aten.cat.default([convolution_38, convolution_39, convolution_40], 1);  convolution_38 = convolution_39 = convolution_40 = None
        view_181 = torch.ops.aten.view.default(cat_22, [8, 8, 8, 3136]);  cat_22 = None
        permute_104 = torch.ops.aten.permute.default(view_181, [0, 1, 3, 2]);  view_181 = None
        mul_86 = torch.ops.aten.mul.Tensor(slice_116, permute_104);  slice_116 = permute_104 = None
        constant_pad_nd_8 = torch.ops.aten.constant_pad_nd.default(mul_86, [0, 0, 1, 0, 0, 0], 0.0);  mul_86 = None
        mul_87 = torch.ops.aten.mul.Tensor(view_179, 0.3535533905932738);  view_179 = None
        add_87 = torch.ops.aten.add.Tensor(mul_87, constant_pad_nd_8);  mul_87 = constant_pad_nd_8 = None
        permute_105 = torch.ops.aten.permute.default(add_87, [0, 2, 1, 3]);  add_87 = None
        clone_69 = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_182 = torch.ops.aten.view.default(clone_69, [8, 3137, 64]);  clone_69 = None
        view_183 = torch.ops.aten.view.default(view_182, [25096, 64]);  view_182 = None
        permute_106 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg19_1, view_183, permute_106);  arg19_1 = view_183 = permute_106 = None
        view_184 = torch.ops.aten.view.default(addmm_34, [8, 3137, 64]);  addmm_34 = None
        add_88 = torch.ops.aten.add.Tensor(cat_21, view_184);  cat_21 = view_184 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_100 = var_mean_23[0]
        getitem_101 = var_mean_23[1];  var_mean_23 = None
        add_89 = torch.ops.aten.add.Tensor(getitem_100, 1e-06);  getitem_100 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_32 = torch.ops.aten.sub.Tensor(add_88, getitem_101);  getitem_101 = None
        mul_88 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_23);  sub_32 = rsqrt_23 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, arg20_1);  mul_88 = arg20_1 = None
        add_90 = torch.ops.aten.add.Tensor(mul_89, arg21_1);  mul_89 = arg21_1 = None
        view_185 = torch.ops.aten.view.default(add_90, [25096, 64]);  add_90 = None
        permute_107 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg23_1, view_185, permute_107);  arg23_1 = view_185 = permute_107 = None
        view_186 = torch.ops.aten.view.default(addmm_35, [8, 3137, 512]);  addmm_35 = None
        mul_90 = torch.ops.aten.mul.Tensor(view_186, 0.5)
        mul_91 = torch.ops.aten.mul.Tensor(view_186, 0.7071067811865476);  view_186 = None
        erf_8 = torch.ops.aten.erf.default(mul_91);  mul_91 = None
        add_91 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_92 = torch.ops.aten.mul.Tensor(mul_90, add_91);  mul_90 = add_91 = None
        view_187 = torch.ops.aten.view.default(mul_92, [25096, 512]);  mul_92 = None
        permute_108 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg25_1, view_187, permute_108);  arg25_1 = view_187 = permute_108 = None
        view_188 = torch.ops.aten.view.default(addmm_36, [8, 3137, 64]);  addmm_36 = None
        add_92 = torch.ops.aten.add.Tensor(add_88, view_188);  add_88 = view_188 = None
        slice_123 = torch.ops.aten.slice.Tensor(add_92, 1, 0, 1)
        slice_125 = torch.ops.aten.slice.Tensor(add_92, 1, 1, 9223372036854775807);  add_92 = None
        permute_109 = torch.ops.aten.permute.default(slice_125, [0, 2, 1]);  slice_125 = None
        view_189 = torch.ops.aten.view.default(permute_109, [8, 64, 56, 56]);  permute_109 = None
        convolution_41 = torch.ops.aten.convolution.default(view_189, arg6_1, arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  arg6_1 = arg7_1 = None
        add_93 = torch.ops.aten.add.Tensor(convolution_41, view_189);  convolution_41 = view_189 = None
        view_190 = torch.ops.aten.view.default(add_93, [8, 64, 3136]);  add_93 = None
        permute_110 = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
        cat_23 = torch.ops.aten.cat.default([slice_123, permute_110], 1);  slice_123 = permute_110 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(cat_23, [2], correction = 0, keepdim = True)
        getitem_102 = var_mean_24[0]
        getitem_103 = var_mean_24[1];  var_mean_24 = None
        add_94 = torch.ops.aten.add.Tensor(getitem_102, 1e-06);  getitem_102 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_33 = torch.ops.aten.sub.Tensor(cat_23, getitem_103);  getitem_103 = None
        mul_93 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_24);  sub_33 = rsqrt_24 = None
        mul_94 = torch.ops.aten.mul.Tensor(mul_93, arg26_1);  mul_93 = arg26_1 = None
        add_95 = torch.ops.aten.add.Tensor(mul_94, arg27_1);  mul_94 = arg27_1 = None
        view_191 = torch.ops.aten.view.default(add_95, [25096, 64]);  add_95 = None
        permute_111 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg29_1, view_191, permute_111);  arg29_1 = view_191 = permute_111 = None
        view_192 = torch.ops.aten.view.default(addmm_37, [8, 3137, 192]);  addmm_37 = None
        view_193 = torch.ops.aten.view.default(view_192, [8, 3137, 3, 8, 8]);  view_192 = None
        permute_112 = torch.ops.aten.permute.default(view_193, [2, 0, 3, 1, 4]);  view_193 = None
        unbind_9 = torch.ops.aten.unbind.int(permute_112);  permute_112 = None
        getitem_104 = unbind_9[0]
        getitem_105 = unbind_9[1]
        getitem_106 = unbind_9[2];  unbind_9 = None
        clone_73 = torch.ops.aten.clone.default(getitem_105, memory_format = torch.contiguous_format);  getitem_105 = None
        amax_9 = torch.ops.aten.amax.default(clone_73, [2], True)
        sub_34 = torch.ops.aten.sub.Tensor(clone_73, amax_9);  clone_73 = amax_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [2], True)
        div_9 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        permute_113 = torch.ops.aten.permute.default(div_9, [0, 1, 3, 2]);  div_9 = None
        expand_41 = torch.ops.aten.expand.default(permute_113, [8, 8, 8, 3137]);  permute_113 = None
        view_194 = torch.ops.aten.view.default(expand_41, [64, 8, 3137]);  expand_41 = None
        expand_42 = torch.ops.aten.expand.default(getitem_106, [8, 8, 3137, 8])
        clone_74 = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_195 = torch.ops.aten.view.default(clone_74, [64, 3137, 8]);  clone_74 = None
        bmm_18 = torch.ops.aten.bmm.default(view_194, view_195);  view_194 = view_195 = None
        view_196 = torch.ops.aten.view.default(bmm_18, [8, 8, 8, 8]);  bmm_18 = None
        expand_43 = torch.ops.aten.expand.default(getitem_104, [8, 8, 3137, 8])
        clone_75 = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_197 = torch.ops.aten.view.default(clone_75, [64, 3137, 8]);  clone_75 = None
        expand_44 = torch.ops.aten.expand.default(view_196, [8, 8, 8, 8]);  view_196 = None
        view_198 = torch.ops.aten.view.default(expand_44, [64, 8, 8]);  expand_44 = None
        bmm_19 = torch.ops.aten.bmm.default(view_197, view_198);  view_197 = view_198 = None
        view_199 = torch.ops.aten.view.default(bmm_19, [8, 8, 3137, 8]);  bmm_19 = None
        slice_128 = torch.ops.aten.slice.Tensor(getitem_104, 2, 1, 9223372036854775807);  getitem_104 = None
        slice_132 = torch.ops.aten.slice.Tensor(getitem_106, 2, 1, 9223372036854775807);  getitem_106 = None
        permute_114 = torch.ops.aten.permute.default(slice_132, [0, 1, 3, 2]);  slice_132 = None
        view_200 = torch.ops.aten.view.default(permute_114, [8, 64, 56, 56]);  permute_114 = None
        split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(view_200, [16, 24, 24], 1);  view_200 = None
        getitem_107 = split_with_sizes_9[0]
        getitem_108 = split_with_sizes_9[1]
        getitem_109 = split_with_sizes_9[2];  split_with_sizes_9 = None
        convolution_42 = torch.ops.aten.convolution.default(getitem_107, arg12_1, arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  getitem_107 = arg12_1 = arg13_1 = None
        convolution_43 = torch.ops.aten.convolution.default(getitem_108, arg14_1, arg15_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 24);  getitem_108 = arg14_1 = arg15_1 = None
        convolution_44 = torch.ops.aten.convolution.default(getitem_109, arg16_1, arg17_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 24);  getitem_109 = arg16_1 = arg17_1 = None
        cat_24 = torch.ops.aten.cat.default([convolution_42, convolution_43, convolution_44], 1);  convolution_42 = convolution_43 = convolution_44 = None
        view_201 = torch.ops.aten.view.default(cat_24, [8, 8, 8, 3136]);  cat_24 = None
        permute_115 = torch.ops.aten.permute.default(view_201, [0, 1, 3, 2]);  view_201 = None
        mul_95 = torch.ops.aten.mul.Tensor(slice_128, permute_115);  slice_128 = permute_115 = None
        constant_pad_nd_9 = torch.ops.aten.constant_pad_nd.default(mul_95, [0, 0, 1, 0, 0, 0], 0.0);  mul_95 = None
        mul_96 = torch.ops.aten.mul.Tensor(view_199, 0.3535533905932738);  view_199 = None
        add_96 = torch.ops.aten.add.Tensor(mul_96, constant_pad_nd_9);  mul_96 = constant_pad_nd_9 = None
        permute_116 = torch.ops.aten.permute.default(add_96, [0, 2, 1, 3]);  add_96 = None
        clone_76 = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
        view_202 = torch.ops.aten.view.default(clone_76, [8, 3137, 64]);  clone_76 = None
        view_203 = torch.ops.aten.view.default(view_202, [25096, 64]);  view_202 = None
        permute_117 = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg31_1, view_203, permute_117);  arg31_1 = view_203 = permute_117 = None
        view_204 = torch.ops.aten.view.default(addmm_38, [8, 3137, 64]);  addmm_38 = None
        add_97 = torch.ops.aten.add.Tensor(cat_23, view_204);  cat_23 = view_204 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_110 = var_mean_25[0]
        getitem_111 = var_mean_25[1];  var_mean_25 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_35 = torch.ops.aten.sub.Tensor(add_97, getitem_111);  getitem_111 = None
        mul_97 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_25);  sub_35 = rsqrt_25 = None
        mul_98 = torch.ops.aten.mul.Tensor(mul_97, arg32_1);  mul_97 = arg32_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_98, arg33_1);  mul_98 = arg33_1 = None
        view_205 = torch.ops.aten.view.default(add_99, [25096, 64]);  add_99 = None
        permute_118 = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg35_1, view_205, permute_118);  arg35_1 = view_205 = permute_118 = None
        view_206 = torch.ops.aten.view.default(addmm_39, [8, 3137, 512]);  addmm_39 = None
        mul_99 = torch.ops.aten.mul.Tensor(view_206, 0.5)
        mul_100 = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476);  view_206 = None
        erf_9 = torch.ops.aten.erf.default(mul_100);  mul_100 = None
        add_100 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_99, add_100);  mul_99 = add_100 = None
        view_207 = torch.ops.aten.view.default(mul_101, [25096, 512]);  mul_101 = None
        permute_119 = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg37_1, view_207, permute_119);  arg37_1 = view_207 = permute_119 = None
        view_208 = torch.ops.aten.view.default(addmm_40, [8, 3137, 64]);  addmm_40 = None
        add_101 = torch.ops.aten.add.Tensor(add_97, view_208);  add_97 = view_208 = None
        slice_135 = torch.ops.aten.slice.Tensor(add_101, 1, 1, 9223372036854775807);  add_101 = None
        view_209 = torch.ops.aten.view.default(slice_135, [8, 56, 56, 64]);  slice_135 = None
        permute_120 = torch.ops.aten.permute.default(view_209, [0, 3, 1, 2]);  view_209 = None
        clone_80 = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
        convolution_45 = torch.ops.aten.convolution.default(clone_80, arg38_1, arg39_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_80 = arg38_1 = arg39_1 = None
        view_210 = torch.ops.aten.view.default(convolution_45, [8, 128, 784]);  convolution_45 = None
        permute_121 = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        clone_81 = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(clone_81, [2], correction = 0, keepdim = True)
        getitem_112 = var_mean_26[0]
        getitem_113 = var_mean_26[1];  var_mean_26 = None
        add_102 = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_36 = torch.ops.aten.sub.Tensor(clone_81, getitem_113);  clone_81 = getitem_113 = None
        mul_102 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_26);  sub_36 = rsqrt_26 = None
        mul_103 = torch.ops.aten.mul.Tensor(mul_102, arg40_1);  mul_102 = arg40_1 = None
        add_103 = torch.ops.aten.add.Tensor(mul_103, arg41_1);  mul_103 = arg41_1 = None
        expand_45 = torch.ops.aten.expand.default(arg42_1, [8, -1, -1]);  arg42_1 = None
        cat_25 = torch.ops.aten.cat.default([expand_45, add_103], 1);  expand_45 = add_103 = None
        slice_138 = torch.ops.aten.slice.Tensor(cat_25, 1, 0, 1)
        slice_140 = torch.ops.aten.slice.Tensor(cat_25, 1, 1, 9223372036854775807);  cat_25 = None
        permute_122 = torch.ops.aten.permute.default(slice_140, [0, 2, 1]);  slice_140 = None
        view_211 = torch.ops.aten.view.default(permute_122, [8, 128, 28, 28]);  permute_122 = None
        convolution_46 = torch.ops.aten.convolution.default(view_211, arg43_1, arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128)
        add_104 = torch.ops.aten.add.Tensor(convolution_46, view_211);  convolution_46 = view_211 = None
        view_212 = torch.ops.aten.view.default(add_104, [8, 128, 784]);  add_104 = None
        permute_123 = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
        cat_26 = torch.ops.aten.cat.default([slice_138, permute_123], 1);  slice_138 = permute_123 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(cat_26, [2], correction = 0, keepdim = True)
        getitem_114 = var_mean_27[0]
        getitem_115 = var_mean_27[1];  var_mean_27 = None
        add_105 = torch.ops.aten.add.Tensor(getitem_114, 1e-06);  getitem_114 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        sub_37 = torch.ops.aten.sub.Tensor(cat_26, getitem_115);  getitem_115 = None
        mul_104 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_27);  sub_37 = rsqrt_27 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_104, arg45_1);  mul_104 = arg45_1 = None
        add_106 = torch.ops.aten.add.Tensor(mul_105, arg46_1);  mul_105 = arg46_1 = None
        view_213 = torch.ops.aten.view.default(add_106, [6280, 128]);  add_106 = None
        permute_124 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg48_1, view_213, permute_124);  arg48_1 = view_213 = permute_124 = None
        view_214 = torch.ops.aten.view.default(addmm_41, [8, 785, 384]);  addmm_41 = None
        view_215 = torch.ops.aten.view.default(view_214, [8, 785, 3, 8, 16]);  view_214 = None
        permute_125 = torch.ops.aten.permute.default(view_215, [2, 0, 3, 1, 4]);  view_215 = None
        unbind_10 = torch.ops.aten.unbind.int(permute_125);  permute_125 = None
        getitem_116 = unbind_10[0]
        getitem_117 = unbind_10[1]
        getitem_118 = unbind_10[2];  unbind_10 = None
        clone_82 = torch.ops.aten.clone.default(getitem_117, memory_format = torch.contiguous_format);  getitem_117 = None
        amax_10 = torch.ops.aten.amax.default(clone_82, [2], True)
        sub_38 = torch.ops.aten.sub.Tensor(clone_82, amax_10);  clone_82 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_38);  sub_38 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [2], True)
        div_10 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        permute_126 = torch.ops.aten.permute.default(div_10, [0, 1, 3, 2]);  div_10 = None
        expand_46 = torch.ops.aten.expand.default(permute_126, [8, 8, 16, 785]);  permute_126 = None
        view_216 = torch.ops.aten.view.default(expand_46, [64, 16, 785]);  expand_46 = None
        expand_47 = torch.ops.aten.expand.default(getitem_118, [8, 8, 785, 16])
        clone_83 = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_217 = torch.ops.aten.view.default(clone_83, [64, 785, 16]);  clone_83 = None
        bmm_20 = torch.ops.aten.bmm.default(view_216, view_217);  view_216 = view_217 = None
        view_218 = torch.ops.aten.view.default(bmm_20, [8, 8, 16, 16]);  bmm_20 = None
        expand_48 = torch.ops.aten.expand.default(getitem_116, [8, 8, 785, 16])
        clone_84 = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_219 = torch.ops.aten.view.default(clone_84, [64, 785, 16]);  clone_84 = None
        expand_49 = torch.ops.aten.expand.default(view_218, [8, 8, 16, 16]);  view_218 = None
        view_220 = torch.ops.aten.view.default(expand_49, [64, 16, 16]);  expand_49 = None
        bmm_21 = torch.ops.aten.bmm.default(view_219, view_220);  view_219 = view_220 = None
        view_221 = torch.ops.aten.view.default(bmm_21, [8, 8, 785, 16]);  bmm_21 = None
        slice_143 = torch.ops.aten.slice.Tensor(getitem_116, 2, 1, 9223372036854775807);  getitem_116 = None
        slice_147 = torch.ops.aten.slice.Tensor(getitem_118, 2, 1, 9223372036854775807);  getitem_118 = None
        permute_127 = torch.ops.aten.permute.default(slice_147, [0, 1, 3, 2]);  slice_147 = None
        view_222 = torch.ops.aten.view.default(permute_127, [8, 128, 28, 28]);  permute_127 = None
        split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(view_222, [32, 48, 48], 1);  view_222 = None
        getitem_119 = split_with_sizes_10[0]
        getitem_120 = split_with_sizes_10[1]
        getitem_121 = split_with_sizes_10[2];  split_with_sizes_10 = None
        convolution_47 = torch.ops.aten.convolution.default(getitem_119, arg49_1, arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  getitem_119 = None
        convolution_48 = torch.ops.aten.convolution.default(getitem_120, arg51_1, arg52_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  getitem_120 = None
        convolution_49 = torch.ops.aten.convolution.default(getitem_121, arg53_1, arg54_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 48);  getitem_121 = None
        cat_27 = torch.ops.aten.cat.default([convolution_47, convolution_48, convolution_49], 1);  convolution_47 = convolution_48 = convolution_49 = None
        view_223 = torch.ops.aten.view.default(cat_27, [8, 8, 16, 784]);  cat_27 = None
        permute_128 = torch.ops.aten.permute.default(view_223, [0, 1, 3, 2]);  view_223 = None
        mul_106 = torch.ops.aten.mul.Tensor(slice_143, permute_128);  slice_143 = permute_128 = None
        constant_pad_nd_10 = torch.ops.aten.constant_pad_nd.default(mul_106, [0, 0, 1, 0, 0, 0], 0.0);  mul_106 = None
        mul_107 = torch.ops.aten.mul.Tensor(view_221, 0.25);  view_221 = None
        add_107 = torch.ops.aten.add.Tensor(mul_107, constant_pad_nd_10);  mul_107 = constant_pad_nd_10 = None
        permute_129 = torch.ops.aten.permute.default(add_107, [0, 2, 1, 3]);  add_107 = None
        clone_85 = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_224 = torch.ops.aten.view.default(clone_85, [8, 785, 128]);  clone_85 = None
        view_225 = torch.ops.aten.view.default(view_224, [6280, 128]);  view_224 = None
        permute_130 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg56_1, view_225, permute_130);  arg56_1 = view_225 = permute_130 = None
        view_226 = torch.ops.aten.view.default(addmm_42, [8, 785, 128]);  addmm_42 = None
        add_108 = torch.ops.aten.add.Tensor(cat_26, view_226);  cat_26 = view_226 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_108, [2], correction = 0, keepdim = True)
        getitem_122 = var_mean_28[0]
        getitem_123 = var_mean_28[1];  var_mean_28 = None
        add_109 = torch.ops.aten.add.Tensor(getitem_122, 1e-06);  getitem_122 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_108, getitem_123);  getitem_123 = None
        mul_108 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_28);  sub_39 = rsqrt_28 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, arg57_1);  mul_108 = arg57_1 = None
        add_110 = torch.ops.aten.add.Tensor(mul_109, arg58_1);  mul_109 = arg58_1 = None
        view_227 = torch.ops.aten.view.default(add_110, [6280, 128]);  add_110 = None
        permute_131 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg60_1, view_227, permute_131);  arg60_1 = view_227 = permute_131 = None
        view_228 = torch.ops.aten.view.default(addmm_43, [8, 785, 1024]);  addmm_43 = None
        mul_110 = torch.ops.aten.mul.Tensor(view_228, 0.5)
        mul_111 = torch.ops.aten.mul.Tensor(view_228, 0.7071067811865476);  view_228 = None
        erf_10 = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_111 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_112 = torch.ops.aten.mul.Tensor(mul_110, add_111);  mul_110 = add_111 = None
        view_229 = torch.ops.aten.view.default(mul_112, [6280, 1024]);  mul_112 = None
        permute_132 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg62_1, view_229, permute_132);  arg62_1 = view_229 = permute_132 = None
        view_230 = torch.ops.aten.view.default(addmm_44, [8, 785, 128]);  addmm_44 = None
        add_112 = torch.ops.aten.add.Tensor(add_108, view_230);  add_108 = view_230 = None
        slice_150 = torch.ops.aten.slice.Tensor(add_112, 1, 0, 1)
        slice_152 = torch.ops.aten.slice.Tensor(add_112, 1, 1, 9223372036854775807);  add_112 = None
        permute_133 = torch.ops.aten.permute.default(slice_152, [0, 2, 1]);  slice_152 = None
        view_231 = torch.ops.aten.view.default(permute_133, [8, 128, 28, 28]);  permute_133 = None
        convolution_50 = torch.ops.aten.convolution.default(view_231, arg43_1, arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  arg43_1 = arg44_1 = None
        add_113 = torch.ops.aten.add.Tensor(convolution_50, view_231);  convolution_50 = view_231 = None
        view_232 = torch.ops.aten.view.default(add_113, [8, 128, 784]);  add_113 = None
        permute_134 = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        cat_28 = torch.ops.aten.cat.default([slice_150, permute_134], 1);  slice_150 = permute_134 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(cat_28, [2], correction = 0, keepdim = True)
        getitem_124 = var_mean_29[0]
        getitem_125 = var_mean_29[1];  var_mean_29 = None
        add_114 = torch.ops.aten.add.Tensor(getitem_124, 1e-06);  getitem_124 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_40 = torch.ops.aten.sub.Tensor(cat_28, getitem_125);  getitem_125 = None
        mul_113 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_29);  sub_40 = rsqrt_29 = None
        mul_114 = torch.ops.aten.mul.Tensor(mul_113, arg63_1);  mul_113 = arg63_1 = None
        add_115 = torch.ops.aten.add.Tensor(mul_114, arg64_1);  mul_114 = arg64_1 = None
        view_233 = torch.ops.aten.view.default(add_115, [6280, 128]);  add_115 = None
        permute_135 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg66_1, view_233, permute_135);  arg66_1 = view_233 = permute_135 = None
        view_234 = torch.ops.aten.view.default(addmm_45, [8, 785, 384]);  addmm_45 = None
        view_235 = torch.ops.aten.view.default(view_234, [8, 785, 3, 8, 16]);  view_234 = None
        permute_136 = torch.ops.aten.permute.default(view_235, [2, 0, 3, 1, 4]);  view_235 = None
        unbind_11 = torch.ops.aten.unbind.int(permute_136);  permute_136 = None
        getitem_126 = unbind_11[0]
        getitem_127 = unbind_11[1]
        getitem_128 = unbind_11[2];  unbind_11 = None
        clone_89 = torch.ops.aten.clone.default(getitem_127, memory_format = torch.contiguous_format);  getitem_127 = None
        amax_11 = torch.ops.aten.amax.default(clone_89, [2], True)
        sub_41 = torch.ops.aten.sub.Tensor(clone_89, amax_11);  clone_89 = amax_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_41);  sub_41 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [2], True)
        div_11 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        permute_137 = torch.ops.aten.permute.default(div_11, [0, 1, 3, 2]);  div_11 = None
        expand_50 = torch.ops.aten.expand.default(permute_137, [8, 8, 16, 785]);  permute_137 = None
        view_236 = torch.ops.aten.view.default(expand_50, [64, 16, 785]);  expand_50 = None
        expand_51 = torch.ops.aten.expand.default(getitem_128, [8, 8, 785, 16])
        clone_90 = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_237 = torch.ops.aten.view.default(clone_90, [64, 785, 16]);  clone_90 = None
        bmm_22 = torch.ops.aten.bmm.default(view_236, view_237);  view_236 = view_237 = None
        view_238 = torch.ops.aten.view.default(bmm_22, [8, 8, 16, 16]);  bmm_22 = None
        expand_52 = torch.ops.aten.expand.default(getitem_126, [8, 8, 785, 16])
        clone_91 = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_239 = torch.ops.aten.view.default(clone_91, [64, 785, 16]);  clone_91 = None
        expand_53 = torch.ops.aten.expand.default(view_238, [8, 8, 16, 16]);  view_238 = None
        view_240 = torch.ops.aten.view.default(expand_53, [64, 16, 16]);  expand_53 = None
        bmm_23 = torch.ops.aten.bmm.default(view_239, view_240);  view_239 = view_240 = None
        view_241 = torch.ops.aten.view.default(bmm_23, [8, 8, 785, 16]);  bmm_23 = None
        slice_155 = torch.ops.aten.slice.Tensor(getitem_126, 2, 1, 9223372036854775807);  getitem_126 = None
        slice_159 = torch.ops.aten.slice.Tensor(getitem_128, 2, 1, 9223372036854775807);  getitem_128 = None
        permute_138 = torch.ops.aten.permute.default(slice_159, [0, 1, 3, 2]);  slice_159 = None
        view_242 = torch.ops.aten.view.default(permute_138, [8, 128, 28, 28]);  permute_138 = None
        split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(view_242, [32, 48, 48], 1);  view_242 = None
        getitem_129 = split_with_sizes_11[0]
        getitem_130 = split_with_sizes_11[1]
        getitem_131 = split_with_sizes_11[2];  split_with_sizes_11 = None
        convolution_51 = torch.ops.aten.convolution.default(getitem_129, arg49_1, arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  getitem_129 = arg49_1 = arg50_1 = None
        convolution_52 = torch.ops.aten.convolution.default(getitem_130, arg51_1, arg52_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  getitem_130 = arg51_1 = arg52_1 = None
        convolution_53 = torch.ops.aten.convolution.default(getitem_131, arg53_1, arg54_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 48);  getitem_131 = arg53_1 = arg54_1 = None
        cat_29 = torch.ops.aten.cat.default([convolution_51, convolution_52, convolution_53], 1);  convolution_51 = convolution_52 = convolution_53 = None
        view_243 = torch.ops.aten.view.default(cat_29, [8, 8, 16, 784]);  cat_29 = None
        permute_139 = torch.ops.aten.permute.default(view_243, [0, 1, 3, 2]);  view_243 = None
        mul_115 = torch.ops.aten.mul.Tensor(slice_155, permute_139);  slice_155 = permute_139 = None
        constant_pad_nd_11 = torch.ops.aten.constant_pad_nd.default(mul_115, [0, 0, 1, 0, 0, 0], 0.0);  mul_115 = None
        mul_116 = torch.ops.aten.mul.Tensor(view_241, 0.25);  view_241 = None
        add_116 = torch.ops.aten.add.Tensor(mul_116, constant_pad_nd_11);  mul_116 = constant_pad_nd_11 = None
        permute_140 = torch.ops.aten.permute.default(add_116, [0, 2, 1, 3]);  add_116 = None
        clone_92 = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_244 = torch.ops.aten.view.default(clone_92, [8, 785, 128]);  clone_92 = None
        view_245 = torch.ops.aten.view.default(view_244, [6280, 128]);  view_244 = None
        permute_141 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg68_1, view_245, permute_141);  arg68_1 = view_245 = permute_141 = None
        view_246 = torch.ops.aten.view.default(addmm_46, [8, 785, 128]);  addmm_46 = None
        add_117 = torch.ops.aten.add.Tensor(cat_28, view_246);  cat_28 = view_246 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_132 = var_mean_30[0]
        getitem_133 = var_mean_30[1];  var_mean_30 = None
        add_118 = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_42 = torch.ops.aten.sub.Tensor(add_117, getitem_133);  getitem_133 = None
        mul_117 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_30);  sub_42 = rsqrt_30 = None
        mul_118 = torch.ops.aten.mul.Tensor(mul_117, arg69_1);  mul_117 = arg69_1 = None
        add_119 = torch.ops.aten.add.Tensor(mul_118, arg70_1);  mul_118 = arg70_1 = None
        view_247 = torch.ops.aten.view.default(add_119, [6280, 128]);  add_119 = None
        permute_142 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg72_1, view_247, permute_142);  arg72_1 = view_247 = permute_142 = None
        view_248 = torch.ops.aten.view.default(addmm_47, [8, 785, 1024]);  addmm_47 = None
        mul_119 = torch.ops.aten.mul.Tensor(view_248, 0.5)
        mul_120 = torch.ops.aten.mul.Tensor(view_248, 0.7071067811865476);  view_248 = None
        erf_11 = torch.ops.aten.erf.default(mul_120);  mul_120 = None
        add_120 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_119, add_120);  mul_119 = add_120 = None
        view_249 = torch.ops.aten.view.default(mul_121, [6280, 1024]);  mul_121 = None
        permute_143 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg74_1, view_249, permute_143);  arg74_1 = view_249 = permute_143 = None
        view_250 = torch.ops.aten.view.default(addmm_48, [8, 785, 128]);  addmm_48 = None
        add_121 = torch.ops.aten.add.Tensor(add_117, view_250);  add_117 = view_250 = None
        slice_162 = torch.ops.aten.slice.Tensor(add_121, 1, 1, 9223372036854775807);  add_121 = None
        view_251 = torch.ops.aten.view.default(slice_162, [8, 28, 28, 128]);  slice_162 = None
        permute_144 = torch.ops.aten.permute.default(view_251, [0, 3, 1, 2]);  view_251 = None
        clone_96 = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        convolution_54 = torch.ops.aten.convolution.default(clone_96, arg75_1, arg76_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_96 = arg75_1 = arg76_1 = None
        view_252 = torch.ops.aten.view.default(convolution_54, [8, 320, 196]);  convolution_54 = None
        permute_145 = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
        clone_97 = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(clone_97, [2], correction = 0, keepdim = True)
        getitem_134 = var_mean_31[0]
        getitem_135 = var_mean_31[1];  var_mean_31 = None
        add_122 = torch.ops.aten.add.Tensor(getitem_134, 1e-05);  getitem_134 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_43 = torch.ops.aten.sub.Tensor(clone_97, getitem_135);  clone_97 = getitem_135 = None
        mul_122 = torch.ops.aten.mul.Tensor(sub_43, rsqrt_31);  sub_43 = rsqrt_31 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_122, arg77_1);  mul_122 = arg77_1 = None
        add_123 = torch.ops.aten.add.Tensor(mul_123, arg78_1);  mul_123 = arg78_1 = None
        expand_54 = torch.ops.aten.expand.default(arg79_1, [8, -1, -1]);  arg79_1 = None
        cat_30 = torch.ops.aten.cat.default([expand_54, add_123], 1);  expand_54 = add_123 = None
        slice_165 = torch.ops.aten.slice.Tensor(cat_30, 1, 0, 1)
        slice_167 = torch.ops.aten.slice.Tensor(cat_30, 1, 1, 9223372036854775807);  cat_30 = None
        permute_146 = torch.ops.aten.permute.default(slice_167, [0, 2, 1]);  slice_167 = None
        view_253 = torch.ops.aten.view.default(permute_146, [8, 320, 14, 14]);  permute_146 = None
        convolution_55 = torch.ops.aten.convolution.default(view_253, arg80_1, arg81_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 320)
        add_124 = torch.ops.aten.add.Tensor(convolution_55, view_253);  convolution_55 = view_253 = None
        view_254 = torch.ops.aten.view.default(add_124, [8, 320, 196]);  add_124 = None
        permute_147 = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        cat_31 = torch.ops.aten.cat.default([slice_165, permute_147], 1);  slice_165 = permute_147 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(cat_31, [2], correction = 0, keepdim = True)
        getitem_136 = var_mean_32[0]
        getitem_137 = var_mean_32[1];  var_mean_32 = None
        add_125 = torch.ops.aten.add.Tensor(getitem_136, 1e-06);  getitem_136 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        sub_44 = torch.ops.aten.sub.Tensor(cat_31, getitem_137);  getitem_137 = None
        mul_124 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_32);  sub_44 = rsqrt_32 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, arg82_1);  mul_124 = arg82_1 = None
        add_126 = torch.ops.aten.add.Tensor(mul_125, arg83_1);  mul_125 = arg83_1 = None
        view_255 = torch.ops.aten.view.default(add_126, [1576, 320]);  add_126 = None
        permute_148 = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg85_1, view_255, permute_148);  arg85_1 = view_255 = permute_148 = None
        view_256 = torch.ops.aten.view.default(addmm_49, [8, 197, 960]);  addmm_49 = None
        view_257 = torch.ops.aten.view.default(view_256, [8, 197, 3, 8, 40]);  view_256 = None
        permute_149 = torch.ops.aten.permute.default(view_257, [2, 0, 3, 1, 4]);  view_257 = None
        unbind_12 = torch.ops.aten.unbind.int(permute_149);  permute_149 = None
        getitem_138 = unbind_12[0]
        getitem_139 = unbind_12[1]
        getitem_140 = unbind_12[2];  unbind_12 = None
        clone_98 = torch.ops.aten.clone.default(getitem_139, memory_format = torch.contiguous_format);  getitem_139 = None
        amax_12 = torch.ops.aten.amax.default(clone_98, [2], True)
        sub_45 = torch.ops.aten.sub.Tensor(clone_98, amax_12);  clone_98 = amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_45);  sub_45 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [2], True)
        div_12 = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        permute_150 = torch.ops.aten.permute.default(div_12, [0, 1, 3, 2]);  div_12 = None
        expand_55 = torch.ops.aten.expand.default(permute_150, [8, 8, 40, 197]);  permute_150 = None
        view_258 = torch.ops.aten.view.default(expand_55, [64, 40, 197]);  expand_55 = None
        expand_56 = torch.ops.aten.expand.default(getitem_140, [8, 8, 197, 40])
        clone_99 = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_259 = torch.ops.aten.view.default(clone_99, [64, 197, 40]);  clone_99 = None
        bmm_24 = torch.ops.aten.bmm.default(view_258, view_259);  view_258 = view_259 = None
        view_260 = torch.ops.aten.view.default(bmm_24, [8, 8, 40, 40]);  bmm_24 = None
        expand_57 = torch.ops.aten.expand.default(getitem_138, [8, 8, 197, 40])
        clone_100 = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_261 = torch.ops.aten.view.default(clone_100, [64, 197, 40]);  clone_100 = None
        expand_58 = torch.ops.aten.expand.default(view_260, [8, 8, 40, 40]);  view_260 = None
        view_262 = torch.ops.aten.view.default(expand_58, [64, 40, 40]);  expand_58 = None
        bmm_25 = torch.ops.aten.bmm.default(view_261, view_262);  view_261 = view_262 = None
        view_263 = torch.ops.aten.view.default(bmm_25, [8, 8, 197, 40]);  bmm_25 = None
        slice_170 = torch.ops.aten.slice.Tensor(getitem_138, 2, 1, 9223372036854775807);  getitem_138 = None
        slice_174 = torch.ops.aten.slice.Tensor(getitem_140, 2, 1, 9223372036854775807);  getitem_140 = None
        permute_151 = torch.ops.aten.permute.default(slice_174, [0, 1, 3, 2]);  slice_174 = None
        view_264 = torch.ops.aten.view.default(permute_151, [8, 320, 14, 14]);  permute_151 = None
        split_with_sizes_12 = torch.ops.aten.split_with_sizes.default(view_264, [80, 120, 120], 1);  view_264 = None
        getitem_141 = split_with_sizes_12[0]
        getitem_142 = split_with_sizes_12[1]
        getitem_143 = split_with_sizes_12[2];  split_with_sizes_12 = None
        convolution_56 = torch.ops.aten.convolution.default(getitem_141, arg86_1, arg87_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  getitem_141 = None
        convolution_57 = torch.ops.aten.convolution.default(getitem_142, arg88_1, arg89_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_142 = None
        convolution_58 = torch.ops.aten.convolution.default(getitem_143, arg90_1, arg91_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_143 = None
        cat_32 = torch.ops.aten.cat.default([convolution_56, convolution_57, convolution_58], 1);  convolution_56 = convolution_57 = convolution_58 = None
        view_265 = torch.ops.aten.view.default(cat_32, [8, 8, 40, 196]);  cat_32 = None
        permute_152 = torch.ops.aten.permute.default(view_265, [0, 1, 3, 2]);  view_265 = None
        mul_126 = torch.ops.aten.mul.Tensor(slice_170, permute_152);  slice_170 = permute_152 = None
        constant_pad_nd_12 = torch.ops.aten.constant_pad_nd.default(mul_126, [0, 0, 1, 0, 0, 0], 0.0);  mul_126 = None
        mul_127 = torch.ops.aten.mul.Tensor(view_263, 0.15811388300841897);  view_263 = None
        add_127 = torch.ops.aten.add.Tensor(mul_127, constant_pad_nd_12);  mul_127 = constant_pad_nd_12 = None
        permute_153 = torch.ops.aten.permute.default(add_127, [0, 2, 1, 3]);  add_127 = None
        clone_101 = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
        view_266 = torch.ops.aten.view.default(clone_101, [8, 197, 320]);  clone_101 = None
        view_267 = torch.ops.aten.view.default(view_266, [1576, 320]);  view_266 = None
        permute_154 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg93_1, view_267, permute_154);  arg93_1 = view_267 = permute_154 = None
        view_268 = torch.ops.aten.view.default(addmm_50, [8, 197, 320]);  addmm_50 = None
        add_128 = torch.ops.aten.add.Tensor(cat_31, view_268);  cat_31 = view_268 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
        getitem_144 = var_mean_33[0]
        getitem_145 = var_mean_33[1];  var_mean_33 = None
        add_129 = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        sub_46 = torch.ops.aten.sub.Tensor(add_128, getitem_145);  getitem_145 = None
        mul_128 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_33);  sub_46 = rsqrt_33 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_128, arg94_1);  mul_128 = arg94_1 = None
        add_130 = torch.ops.aten.add.Tensor(mul_129, arg95_1);  mul_129 = arg95_1 = None
        view_269 = torch.ops.aten.view.default(add_130, [1576, 320]);  add_130 = None
        permute_155 = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg97_1, view_269, permute_155);  arg97_1 = view_269 = permute_155 = None
        view_270 = torch.ops.aten.view.default(addmm_51, [8, 197, 1280]);  addmm_51 = None
        mul_130 = torch.ops.aten.mul.Tensor(view_270, 0.5)
        mul_131 = torch.ops.aten.mul.Tensor(view_270, 0.7071067811865476);  view_270 = None
        erf_12 = torch.ops.aten.erf.default(mul_131);  mul_131 = None
        add_131 = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_130, add_131);  mul_130 = add_131 = None
        view_271 = torch.ops.aten.view.default(mul_132, [1576, 1280]);  mul_132 = None
        permute_156 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg99_1, view_271, permute_156);  arg99_1 = view_271 = permute_156 = None
        view_272 = torch.ops.aten.view.default(addmm_52, [8, 197, 320]);  addmm_52 = None
        add_132 = torch.ops.aten.add.Tensor(add_128, view_272);  add_128 = view_272 = None
        slice_177 = torch.ops.aten.slice.Tensor(add_132, 1, 0, 1)
        slice_179 = torch.ops.aten.slice.Tensor(add_132, 1, 1, 9223372036854775807);  add_132 = None
        permute_157 = torch.ops.aten.permute.default(slice_179, [0, 2, 1]);  slice_179 = None
        view_273 = torch.ops.aten.view.default(permute_157, [8, 320, 14, 14]);  permute_157 = None
        convolution_59 = torch.ops.aten.convolution.default(view_273, arg80_1, arg81_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 320);  arg80_1 = arg81_1 = None
        add_133 = torch.ops.aten.add.Tensor(convolution_59, view_273);  convolution_59 = view_273 = None
        view_274 = torch.ops.aten.view.default(add_133, [8, 320, 196]);  add_133 = None
        permute_158 = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None
        cat_33 = torch.ops.aten.cat.default([slice_177, permute_158], 1);  slice_177 = permute_158 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(cat_33, [2], correction = 0, keepdim = True)
        getitem_146 = var_mean_34[0]
        getitem_147 = var_mean_34[1];  var_mean_34 = None
        add_134 = torch.ops.aten.add.Tensor(getitem_146, 1e-06);  getitem_146 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_47 = torch.ops.aten.sub.Tensor(cat_33, getitem_147);  getitem_147 = None
        mul_133 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_34);  sub_47 = rsqrt_34 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_133, arg100_1);  mul_133 = arg100_1 = None
        add_135 = torch.ops.aten.add.Tensor(mul_134, arg101_1);  mul_134 = arg101_1 = None
        view_275 = torch.ops.aten.view.default(add_135, [1576, 320]);  add_135 = None
        permute_159 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg103_1, view_275, permute_159);  arg103_1 = view_275 = permute_159 = None
        view_276 = torch.ops.aten.view.default(addmm_53, [8, 197, 960]);  addmm_53 = None
        view_277 = torch.ops.aten.view.default(view_276, [8, 197, 3, 8, 40]);  view_276 = None
        permute_160 = torch.ops.aten.permute.default(view_277, [2, 0, 3, 1, 4]);  view_277 = None
        unbind_13 = torch.ops.aten.unbind.int(permute_160);  permute_160 = None
        getitem_148 = unbind_13[0]
        getitem_149 = unbind_13[1]
        getitem_150 = unbind_13[2];  unbind_13 = None
        clone_105 = torch.ops.aten.clone.default(getitem_149, memory_format = torch.contiguous_format);  getitem_149 = None
        amax_13 = torch.ops.aten.amax.default(clone_105, [2], True)
        sub_48 = torch.ops.aten.sub.Tensor(clone_105, amax_13);  clone_105 = amax_13 = None
        exp_13 = torch.ops.aten.exp.default(sub_48);  sub_48 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(exp_13, [2], True)
        div_13 = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        permute_161 = torch.ops.aten.permute.default(div_13, [0, 1, 3, 2]);  div_13 = None
        expand_59 = torch.ops.aten.expand.default(permute_161, [8, 8, 40, 197]);  permute_161 = None
        view_278 = torch.ops.aten.view.default(expand_59, [64, 40, 197]);  expand_59 = None
        expand_60 = torch.ops.aten.expand.default(getitem_150, [8, 8, 197, 40])
        clone_106 = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_279 = torch.ops.aten.view.default(clone_106, [64, 197, 40]);  clone_106 = None
        bmm_26 = torch.ops.aten.bmm.default(view_278, view_279);  view_278 = view_279 = None
        view_280 = torch.ops.aten.view.default(bmm_26, [8, 8, 40, 40]);  bmm_26 = None
        expand_61 = torch.ops.aten.expand.default(getitem_148, [8, 8, 197, 40])
        clone_107 = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_281 = torch.ops.aten.view.default(clone_107, [64, 197, 40]);  clone_107 = None
        expand_62 = torch.ops.aten.expand.default(view_280, [8, 8, 40, 40]);  view_280 = None
        view_282 = torch.ops.aten.view.default(expand_62, [64, 40, 40]);  expand_62 = None
        bmm_27 = torch.ops.aten.bmm.default(view_281, view_282);  view_281 = view_282 = None
        view_283 = torch.ops.aten.view.default(bmm_27, [8, 8, 197, 40]);  bmm_27 = None
        slice_182 = torch.ops.aten.slice.Tensor(getitem_148, 2, 1, 9223372036854775807);  getitem_148 = None
        slice_186 = torch.ops.aten.slice.Tensor(getitem_150, 2, 1, 9223372036854775807);  getitem_150 = None
        permute_162 = torch.ops.aten.permute.default(slice_186, [0, 1, 3, 2]);  slice_186 = None
        view_284 = torch.ops.aten.view.default(permute_162, [8, 320, 14, 14]);  permute_162 = None
        split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(view_284, [80, 120, 120], 1);  view_284 = None
        getitem_151 = split_with_sizes_13[0]
        getitem_152 = split_with_sizes_13[1]
        getitem_153 = split_with_sizes_13[2];  split_with_sizes_13 = None
        convolution_60 = torch.ops.aten.convolution.default(getitem_151, arg86_1, arg87_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  getitem_151 = arg86_1 = arg87_1 = None
        convolution_61 = torch.ops.aten.convolution.default(getitem_152, arg88_1, arg89_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_152 = arg88_1 = arg89_1 = None
        convolution_62 = torch.ops.aten.convolution.default(getitem_153, arg90_1, arg91_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_153 = arg90_1 = arg91_1 = None
        cat_34 = torch.ops.aten.cat.default([convolution_60, convolution_61, convolution_62], 1);  convolution_60 = convolution_61 = convolution_62 = None
        view_285 = torch.ops.aten.view.default(cat_34, [8, 8, 40, 196]);  cat_34 = None
        permute_163 = torch.ops.aten.permute.default(view_285, [0, 1, 3, 2]);  view_285 = None
        mul_135 = torch.ops.aten.mul.Tensor(slice_182, permute_163);  slice_182 = permute_163 = None
        constant_pad_nd_13 = torch.ops.aten.constant_pad_nd.default(mul_135, [0, 0, 1, 0, 0, 0], 0.0);  mul_135 = None
        mul_136 = torch.ops.aten.mul.Tensor(view_283, 0.15811388300841897);  view_283 = None
        add_136 = torch.ops.aten.add.Tensor(mul_136, constant_pad_nd_13);  mul_136 = constant_pad_nd_13 = None
        permute_164 = torch.ops.aten.permute.default(add_136, [0, 2, 1, 3]);  add_136 = None
        clone_108 = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
        view_286 = torch.ops.aten.view.default(clone_108, [8, 197, 320]);  clone_108 = None
        view_287 = torch.ops.aten.view.default(view_286, [1576, 320]);  view_286 = None
        permute_165 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg105_1, view_287, permute_165);  arg105_1 = view_287 = permute_165 = None
        view_288 = torch.ops.aten.view.default(addmm_54, [8, 197, 320]);  addmm_54 = None
        add_137 = torch.ops.aten.add.Tensor(cat_33, view_288);  cat_33 = view_288 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_154 = var_mean_35[0]
        getitem_155 = var_mean_35[1];  var_mean_35 = None
        add_138 = torch.ops.aten.add.Tensor(getitem_154, 1e-06);  getitem_154 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_49 = torch.ops.aten.sub.Tensor(add_137, getitem_155);  getitem_155 = None
        mul_137 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_35);  sub_49 = rsqrt_35 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_137, arg106_1);  mul_137 = arg106_1 = None
        add_139 = torch.ops.aten.add.Tensor(mul_138, arg107_1);  mul_138 = arg107_1 = None
        view_289 = torch.ops.aten.view.default(add_139, [1576, 320]);  add_139 = None
        permute_166 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg109_1, view_289, permute_166);  arg109_1 = view_289 = permute_166 = None
        view_290 = torch.ops.aten.view.default(addmm_55, [8, 197, 1280]);  addmm_55 = None
        mul_139 = torch.ops.aten.mul.Tensor(view_290, 0.5)
        mul_140 = torch.ops.aten.mul.Tensor(view_290, 0.7071067811865476);  view_290 = None
        erf_13 = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_140 = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_139, add_140);  mul_139 = add_140 = None
        view_291 = torch.ops.aten.view.default(mul_141, [1576, 1280]);  mul_141 = None
        permute_167 = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg111_1, view_291, permute_167);  arg111_1 = view_291 = permute_167 = None
        view_292 = torch.ops.aten.view.default(addmm_56, [8, 197, 320]);  addmm_56 = None
        add_141 = torch.ops.aten.add.Tensor(add_137, view_292);  add_137 = view_292 = None
        slice_189 = torch.ops.aten.slice.Tensor(add_141, 1, 1, 9223372036854775807);  add_141 = None
        view_293 = torch.ops.aten.view.default(slice_189, [8, 14, 14, 320]);  slice_189 = None
        permute_168 = torch.ops.aten.permute.default(view_293, [0, 3, 1, 2]);  view_293 = None
        clone_112 = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        convolution_63 = torch.ops.aten.convolution.default(clone_112, arg112_1, arg113_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_112 = arg112_1 = arg113_1 = None
        view_294 = torch.ops.aten.view.default(convolution_63, [8, 512, 49]);  convolution_63 = None
        permute_169 = torch.ops.aten.permute.default(view_294, [0, 2, 1]);  view_294 = None
        clone_113 = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(clone_113, [2], correction = 0, keepdim = True)
        getitem_156 = var_mean_36[0]
        getitem_157 = var_mean_36[1];  var_mean_36 = None
        add_142 = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        sub_50 = torch.ops.aten.sub.Tensor(clone_113, getitem_157);  clone_113 = getitem_157 = None
        mul_142 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_36);  sub_50 = rsqrt_36 = None
        mul_143 = torch.ops.aten.mul.Tensor(mul_142, arg114_1);  mul_142 = arg114_1 = None
        add_143 = torch.ops.aten.add.Tensor(mul_143, arg115_1);  mul_143 = arg115_1 = None
        expand_63 = torch.ops.aten.expand.default(arg116_1, [8, -1, -1]);  arg116_1 = None
        cat_35 = torch.ops.aten.cat.default([expand_63, add_143], 1);  expand_63 = add_143 = None
        slice_192 = torch.ops.aten.slice.Tensor(cat_35, 1, 0, 1)
        slice_194 = torch.ops.aten.slice.Tensor(cat_35, 1, 1, 9223372036854775807);  cat_35 = None
        permute_170 = torch.ops.aten.permute.default(slice_194, [0, 2, 1]);  slice_194 = None
        view_295 = torch.ops.aten.view.default(permute_170, [8, 512, 7, 7]);  permute_170 = None
        convolution_64 = torch.ops.aten.convolution.default(view_295, arg117_1, arg118_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 512)
        add_144 = torch.ops.aten.add.Tensor(convolution_64, view_295);  convolution_64 = view_295 = None
        view_296 = torch.ops.aten.view.default(add_144, [8, 512, 49]);  add_144 = None
        permute_171 = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None
        cat_36 = torch.ops.aten.cat.default([slice_192, permute_171], 1);  slice_192 = permute_171 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(cat_36, [2], correction = 0, keepdim = True)
        getitem_158 = var_mean_37[0]
        getitem_159 = var_mean_37[1];  var_mean_37 = None
        add_145 = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        sub_51 = torch.ops.aten.sub.Tensor(cat_36, getitem_159);  getitem_159 = None
        mul_144 = torch.ops.aten.mul.Tensor(sub_51, rsqrt_37);  sub_51 = rsqrt_37 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_144, arg119_1);  mul_144 = arg119_1 = None
        add_146 = torch.ops.aten.add.Tensor(mul_145, arg120_1);  mul_145 = arg120_1 = None
        view_297 = torch.ops.aten.view.default(add_146, [400, 512]);  add_146 = None
        permute_172 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg122_1, view_297, permute_172);  arg122_1 = view_297 = permute_172 = None
        view_298 = torch.ops.aten.view.default(addmm_57, [8, 50, 1536]);  addmm_57 = None
        view_299 = torch.ops.aten.view.default(view_298, [8, 50, 3, 8, 64]);  view_298 = None
        permute_173 = torch.ops.aten.permute.default(view_299, [2, 0, 3, 1, 4]);  view_299 = None
        unbind_14 = torch.ops.aten.unbind.int(permute_173);  permute_173 = None
        getitem_160 = unbind_14[0]
        getitem_161 = unbind_14[1]
        getitem_162 = unbind_14[2];  unbind_14 = None
        clone_114 = torch.ops.aten.clone.default(getitem_161, memory_format = torch.contiguous_format);  getitem_161 = None
        amax_14 = torch.ops.aten.amax.default(clone_114, [2], True)
        sub_52 = torch.ops.aten.sub.Tensor(clone_114, amax_14);  clone_114 = amax_14 = None
        exp_14 = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_14, [2], True)
        div_14 = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        permute_174 = torch.ops.aten.permute.default(div_14, [0, 1, 3, 2]);  div_14 = None
        expand_64 = torch.ops.aten.expand.default(permute_174, [8, 8, 64, 50]);  permute_174 = None
        view_300 = torch.ops.aten.view.default(expand_64, [64, 64, 50]);  expand_64 = None
        expand_65 = torch.ops.aten.expand.default(getitem_162, [8, 8, 50, 64])
        clone_115 = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_301 = torch.ops.aten.view.default(clone_115, [64, 50, 64]);  clone_115 = None
        bmm_28 = torch.ops.aten.bmm.default(view_300, view_301);  view_300 = view_301 = None
        view_302 = torch.ops.aten.view.default(bmm_28, [8, 8, 64, 64]);  bmm_28 = None
        expand_66 = torch.ops.aten.expand.default(getitem_160, [8, 8, 50, 64])
        clone_116 = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_303 = torch.ops.aten.view.default(clone_116, [64, 50, 64]);  clone_116 = None
        expand_67 = torch.ops.aten.expand.default(view_302, [8, 8, 64, 64]);  view_302 = None
        view_304 = torch.ops.aten.view.default(expand_67, [64, 64, 64]);  expand_67 = None
        bmm_29 = torch.ops.aten.bmm.default(view_303, view_304);  view_303 = view_304 = None
        view_305 = torch.ops.aten.view.default(bmm_29, [8, 8, 50, 64]);  bmm_29 = None
        slice_197 = torch.ops.aten.slice.Tensor(getitem_160, 2, 1, 9223372036854775807);  getitem_160 = None
        slice_201 = torch.ops.aten.slice.Tensor(getitem_162, 2, 1, 9223372036854775807);  getitem_162 = None
        permute_175 = torch.ops.aten.permute.default(slice_201, [0, 1, 3, 2]);  slice_201 = None
        view_306 = torch.ops.aten.view.default(permute_175, [8, 512, 7, 7]);  permute_175 = None
        split_with_sizes_14 = torch.ops.aten.split_with_sizes.default(view_306, [128, 192, 192], 1);  view_306 = None
        getitem_163 = split_with_sizes_14[0]
        getitem_164 = split_with_sizes_14[1]
        getitem_165 = split_with_sizes_14[2];  split_with_sizes_14 = None
        convolution_65 = torch.ops.aten.convolution.default(getitem_163, arg123_1, arg124_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  getitem_163 = None
        convolution_66 = torch.ops.aten.convolution.default(getitem_164, arg125_1, arg126_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 192);  getitem_164 = None
        convolution_67 = torch.ops.aten.convolution.default(getitem_165, arg127_1, arg128_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 192);  getitem_165 = None
        cat_37 = torch.ops.aten.cat.default([convolution_65, convolution_66, convolution_67], 1);  convolution_65 = convolution_66 = convolution_67 = None
        view_307 = torch.ops.aten.view.default(cat_37, [8, 8, 64, 49]);  cat_37 = None
        permute_176 = torch.ops.aten.permute.default(view_307, [0, 1, 3, 2]);  view_307 = None
        mul_146 = torch.ops.aten.mul.Tensor(slice_197, permute_176);  slice_197 = permute_176 = None
        constant_pad_nd_14 = torch.ops.aten.constant_pad_nd.default(mul_146, [0, 0, 1, 0, 0, 0], 0.0);  mul_146 = None
        mul_147 = torch.ops.aten.mul.Tensor(view_305, 0.125);  view_305 = None
        add_147 = torch.ops.aten.add.Tensor(mul_147, constant_pad_nd_14);  mul_147 = constant_pad_nd_14 = None
        permute_177 = torch.ops.aten.permute.default(add_147, [0, 2, 1, 3]);  add_147 = None
        clone_117 = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_308 = torch.ops.aten.view.default(clone_117, [8, 50, 512]);  clone_117 = None
        view_309 = torch.ops.aten.view.default(view_308, [400, 512]);  view_308 = None
        permute_178 = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg130_1, view_309, permute_178);  arg130_1 = view_309 = permute_178 = None
        view_310 = torch.ops.aten.view.default(addmm_58, [8, 50, 512]);  addmm_58 = None
        add_148 = torch.ops.aten.add.Tensor(cat_36, view_310);  cat_36 = view_310 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_148, [2], correction = 0, keepdim = True)
        getitem_166 = var_mean_38[0]
        getitem_167 = var_mean_38[1];  var_mean_38 = None
        add_149 = torch.ops.aten.add.Tensor(getitem_166, 1e-06);  getitem_166 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_53 = torch.ops.aten.sub.Tensor(add_148, getitem_167);  getitem_167 = None
        mul_148 = torch.ops.aten.mul.Tensor(sub_53, rsqrt_38);  sub_53 = rsqrt_38 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, arg131_1);  mul_148 = arg131_1 = None
        add_150 = torch.ops.aten.add.Tensor(mul_149, arg132_1);  mul_149 = arg132_1 = None
        view_311 = torch.ops.aten.view.default(add_150, [400, 512]);  add_150 = None
        permute_179 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg134_1, view_311, permute_179);  arg134_1 = view_311 = permute_179 = None
        view_312 = torch.ops.aten.view.default(addmm_59, [8, 50, 2048]);  addmm_59 = None
        mul_150 = torch.ops.aten.mul.Tensor(view_312, 0.5)
        mul_151 = torch.ops.aten.mul.Tensor(view_312, 0.7071067811865476);  view_312 = None
        erf_14 = torch.ops.aten.erf.default(mul_151);  mul_151 = None
        add_151 = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_152 = torch.ops.aten.mul.Tensor(mul_150, add_151);  mul_150 = add_151 = None
        view_313 = torch.ops.aten.view.default(mul_152, [400, 2048]);  mul_152 = None
        permute_180 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg136_1, view_313, permute_180);  arg136_1 = view_313 = permute_180 = None
        view_314 = torch.ops.aten.view.default(addmm_60, [8, 50, 512]);  addmm_60 = None
        add_152 = torch.ops.aten.add.Tensor(add_148, view_314);  add_148 = view_314 = None
        slice_204 = torch.ops.aten.slice.Tensor(add_152, 1, 0, 1)
        slice_206 = torch.ops.aten.slice.Tensor(add_152, 1, 1, 9223372036854775807);  add_152 = None
        permute_181 = torch.ops.aten.permute.default(slice_206, [0, 2, 1]);  slice_206 = None
        view_315 = torch.ops.aten.view.default(permute_181, [8, 512, 7, 7]);  permute_181 = None
        convolution_68 = torch.ops.aten.convolution.default(view_315, arg117_1, arg118_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 512);  arg117_1 = arg118_1 = None
        add_153 = torch.ops.aten.add.Tensor(convolution_68, view_315);  convolution_68 = view_315 = None
        view_316 = torch.ops.aten.view.default(add_153, [8, 512, 49]);  add_153 = None
        permute_182 = torch.ops.aten.permute.default(view_316, [0, 2, 1]);  view_316 = None
        cat_38 = torch.ops.aten.cat.default([slice_204, permute_182], 1);  slice_204 = permute_182 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(cat_38, [2], correction = 0, keepdim = True)
        getitem_168 = var_mean_39[0]
        getitem_169 = var_mean_39[1];  var_mean_39 = None
        add_154 = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_54 = torch.ops.aten.sub.Tensor(cat_38, getitem_169);  getitem_169 = None
        mul_153 = torch.ops.aten.mul.Tensor(sub_54, rsqrt_39);  sub_54 = rsqrt_39 = None
        mul_154 = torch.ops.aten.mul.Tensor(mul_153, arg137_1);  mul_153 = arg137_1 = None
        add_155 = torch.ops.aten.add.Tensor(mul_154, arg138_1);  mul_154 = arg138_1 = None
        view_317 = torch.ops.aten.view.default(add_155, [400, 512]);  add_155 = None
        permute_183 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg140_1, view_317, permute_183);  arg140_1 = view_317 = permute_183 = None
        view_318 = torch.ops.aten.view.default(addmm_61, [8, 50, 1536]);  addmm_61 = None
        view_319 = torch.ops.aten.view.default(view_318, [8, 50, 3, 8, 64]);  view_318 = None
        permute_184 = torch.ops.aten.permute.default(view_319, [2, 0, 3, 1, 4]);  view_319 = None
        unbind_15 = torch.ops.aten.unbind.int(permute_184);  permute_184 = None
        getitem_170 = unbind_15[0]
        getitem_171 = unbind_15[1]
        getitem_172 = unbind_15[2];  unbind_15 = None
        clone_121 = torch.ops.aten.clone.default(getitem_171, memory_format = torch.contiguous_format);  getitem_171 = None
        amax_15 = torch.ops.aten.amax.default(clone_121, [2], True)
        sub_55 = torch.ops.aten.sub.Tensor(clone_121, amax_15);  clone_121 = amax_15 = None
        exp_15 = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_15, [2], True)
        div_15 = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        permute_185 = torch.ops.aten.permute.default(div_15, [0, 1, 3, 2]);  div_15 = None
        expand_68 = torch.ops.aten.expand.default(permute_185, [8, 8, 64, 50]);  permute_185 = None
        view_320 = torch.ops.aten.view.default(expand_68, [64, 64, 50]);  expand_68 = None
        expand_69 = torch.ops.aten.expand.default(getitem_172, [8, 8, 50, 64])
        clone_122 = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_321 = torch.ops.aten.view.default(clone_122, [64, 50, 64]);  clone_122 = None
        bmm_30 = torch.ops.aten.bmm.default(view_320, view_321);  view_320 = view_321 = None
        view_322 = torch.ops.aten.view.default(bmm_30, [8, 8, 64, 64]);  bmm_30 = None
        expand_70 = torch.ops.aten.expand.default(getitem_170, [8, 8, 50, 64])
        clone_123 = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_323 = torch.ops.aten.view.default(clone_123, [64, 50, 64]);  clone_123 = None
        expand_71 = torch.ops.aten.expand.default(view_322, [8, 8, 64, 64]);  view_322 = None
        view_324 = torch.ops.aten.view.default(expand_71, [64, 64, 64]);  expand_71 = None
        bmm_31 = torch.ops.aten.bmm.default(view_323, view_324);  view_323 = view_324 = None
        view_325 = torch.ops.aten.view.default(bmm_31, [8, 8, 50, 64]);  bmm_31 = None
        slice_209 = torch.ops.aten.slice.Tensor(getitem_170, 2, 1, 9223372036854775807);  getitem_170 = None
        slice_213 = torch.ops.aten.slice.Tensor(getitem_172, 2, 1, 9223372036854775807);  getitem_172 = None
        permute_186 = torch.ops.aten.permute.default(slice_213, [0, 1, 3, 2]);  slice_213 = None
        view_326 = torch.ops.aten.view.default(permute_186, [8, 512, 7, 7]);  permute_186 = None
        split_with_sizes_15 = torch.ops.aten.split_with_sizes.default(view_326, [128, 192, 192], 1);  view_326 = None
        getitem_173 = split_with_sizes_15[0]
        getitem_174 = split_with_sizes_15[1]
        getitem_175 = split_with_sizes_15[2];  split_with_sizes_15 = None
        convolution_69 = torch.ops.aten.convolution.default(getitem_173, arg123_1, arg124_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  getitem_173 = arg123_1 = arg124_1 = None
        convolution_70 = torch.ops.aten.convolution.default(getitem_174, arg125_1, arg126_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 192);  getitem_174 = arg125_1 = arg126_1 = None
        convolution_71 = torch.ops.aten.convolution.default(getitem_175, arg127_1, arg128_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 192);  getitem_175 = arg127_1 = arg128_1 = None
        cat_39 = torch.ops.aten.cat.default([convolution_69, convolution_70, convolution_71], 1);  convolution_69 = convolution_70 = convolution_71 = None
        view_327 = torch.ops.aten.view.default(cat_39, [8, 8, 64, 49]);  cat_39 = None
        permute_187 = torch.ops.aten.permute.default(view_327, [0, 1, 3, 2]);  view_327 = None
        mul_155 = torch.ops.aten.mul.Tensor(slice_209, permute_187);  slice_209 = permute_187 = None
        constant_pad_nd_15 = torch.ops.aten.constant_pad_nd.default(mul_155, [0, 0, 1, 0, 0, 0], 0.0);  mul_155 = None
        mul_156 = torch.ops.aten.mul.Tensor(view_325, 0.125);  view_325 = None
        add_156 = torch.ops.aten.add.Tensor(mul_156, constant_pad_nd_15);  mul_156 = constant_pad_nd_15 = None
        permute_188 = torch.ops.aten.permute.default(add_156, [0, 2, 1, 3]);  add_156 = None
        clone_124 = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_328 = torch.ops.aten.view.default(clone_124, [8, 50, 512]);  clone_124 = None
        view_329 = torch.ops.aten.view.default(view_328, [400, 512]);  view_328 = None
        permute_189 = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg142_1, view_329, permute_189);  arg142_1 = view_329 = permute_189 = None
        view_330 = torch.ops.aten.view.default(addmm_62, [8, 50, 512]);  addmm_62 = None
        add_157 = torch.ops.aten.add.Tensor(cat_38, view_330);  cat_38 = view_330 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_176 = var_mean_40[0]
        getitem_177 = var_mean_40[1];  var_mean_40 = None
        add_158 = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        sub_56 = torch.ops.aten.sub.Tensor(add_157, getitem_177);  getitem_177 = None
        mul_157 = torch.ops.aten.mul.Tensor(sub_56, rsqrt_40);  sub_56 = rsqrt_40 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_157, arg143_1);  mul_157 = arg143_1 = None
        add_159 = torch.ops.aten.add.Tensor(mul_158, arg144_1);  mul_158 = arg144_1 = None
        view_331 = torch.ops.aten.view.default(add_159, [400, 512]);  add_159 = None
        permute_190 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg146_1, view_331, permute_190);  arg146_1 = view_331 = permute_190 = None
        view_332 = torch.ops.aten.view.default(addmm_63, [8, 50, 2048]);  addmm_63 = None
        mul_159 = torch.ops.aten.mul.Tensor(view_332, 0.5)
        mul_160 = torch.ops.aten.mul.Tensor(view_332, 0.7071067811865476);  view_332 = None
        erf_15 = torch.ops.aten.erf.default(mul_160);  mul_160 = None
        add_160 = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_159, add_160);  mul_159 = add_160 = None
        view_333 = torch.ops.aten.view.default(mul_161, [400, 2048]);  mul_161 = None
        permute_191 = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg148_1, view_333, permute_191);  arg148_1 = view_333 = permute_191 = None
        view_334 = torch.ops.aten.view.default(addmm_64, [8, 50, 512]);  addmm_64 = None
        add_161 = torch.ops.aten.add.Tensor(add_157, view_334);  add_157 = view_334 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_178 = var_mean_41[0]
        getitem_179 = var_mean_41[1];  var_mean_41 = None
        add_162 = torch.ops.aten.add.Tensor(getitem_178, 1e-06);  getitem_178 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_57 = torch.ops.aten.sub.Tensor(add_161, getitem_179);  add_161 = getitem_179 = None
        mul_162 = torch.ops.aten.mul.Tensor(sub_57, rsqrt_41);  sub_57 = rsqrt_41 = None
        mul_163 = torch.ops.aten.mul.Tensor(mul_162, arg149_1);  mul_162 = arg149_1 = None
        add_163 = torch.ops.aten.add.Tensor(mul_163, arg150_1);  mul_163 = arg150_1 = None
        select_1 = torch.ops.aten.select.int(add_163, 1, 0);  add_163 = None
        clone_129 = torch.ops.aten.clone.default(select_1);  select_1 = None
        permute_193 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg152_1, clone_129, permute_193);  arg152_1 = clone_129 = permute_193 = None
        return (addmm_65,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 224, 224), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf1, (64, 3, 4, 4), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf3, (64,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf4, (64,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf5, (1, 1, 64), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf6, (64, 1, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf9, (64,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 49152, device=device(type='cuda', index=0))
    reader.tensor(buf10, (192, 64), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf11, (192,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf12, (16, 1, 3, 3), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf13, (16,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 2400, device=device(type='cuda', index=0))
    reader.tensor(buf14, (24, 1, 5, 5), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf15, (24,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 4704, device=device(type='cuda', index=0))
    reader.tensor(buf16, (24, 1, 7, 7), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf17, (24,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf18, (64, 64), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf19, (64,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf20, (64,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf21, (64,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (512, 64), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf23, (512,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (64, 512), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf25, (64,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf26, (64,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf27, (64,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 49152, device=device(type='cuda', index=0))
    reader.tensor(buf28, (192, 64), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf29, (192,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf30, (64, 64), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf31, (64,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf32, (64,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf33, (64,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (512, 64), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf35, (512,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf36, (64, 512), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf37, (64,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (128, 64, 2, 2), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf39, (128,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf40, (128,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf41, (128,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf42, (1, 1, 128), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf43, (128, 1, 3, 3), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf44, (128,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf45, (128,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf46, (128,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf47, (384, 128), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf48, (384,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf49, (32, 1, 3, 3), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf50, (32,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 4800, device=device(type='cuda', index=0))
    reader.tensor(buf51, (48, 1, 5, 5), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf52, (48,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 9408, device=device(type='cuda', index=0))
    reader.tensor(buf53, (48, 1, 7, 7), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf54, (48,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf55, (128, 128), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf56, (128,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf57, (128,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf58, (128,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf59, (1024, 128), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf60, (1024,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128, 1024), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf63, (128,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf64, (128,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf65, (384, 128), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf66, (384,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf67, (128, 128), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf68, (128,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf69, (128,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf70, (128,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf71, (1024, 128), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf72, (1024,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf73, (128, 1024), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf74, (128,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 655360, device=device(type='cuda', index=0))
    reader.tensor(buf75, (320, 128, 2, 2), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf76, (320,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf77, (320,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf78, (320,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf79, (1, 1, 320), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 11520, device=device(type='cuda', index=0))
    reader.tensor(buf80, (320, 1, 3, 3), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf81, (320,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf82, (320,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf83, (320,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf84, (960, 320), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf85, (960,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf86, (80, 1, 3, 3), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf87, (80,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 12000, device=device(type='cuda', index=0))
    reader.tensor(buf88, (120, 1, 5, 5), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf89, (120,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 23520, device=device(type='cuda', index=0))
    reader.tensor(buf90, (120, 1, 7, 7), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf91, (120,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf92, (320, 320), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf93, (320,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf94, (320,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf95, (320,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf96, (1280, 320), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf97, (1280,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf98, (320, 1280), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf99, (320,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf100, (320,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf101, (320,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf102, (960, 320), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf103, (960,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf104, (320, 320), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf105, (320,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf106, (320,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf107, (320,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf108, (1280, 320), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf109, (1280,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf110, (320, 1280), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf111, (320,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 2621440, device=device(type='cuda', index=0))
    reader.tensor(buf112, (512, 320, 2, 2), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf113, (512,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf114, (512,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf115, (512,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1, 1, 512), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf117, (512, 1, 3, 3), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf118, (512,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf119, (512,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf120, (512,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf121, (1536, 512), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf122, (1536,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf123, (128, 1, 3, 3), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf124, (128,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf125, (192, 1, 5, 5), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf126, (192,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 37632, device=device(type='cuda', index=0))
    reader.tensor(buf127, (192, 1, 7, 7), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf128, (192,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf129, (512, 512), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf130, (512,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf131, (512,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf132, (512,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf133, (2048, 512), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf134, (2048,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf135, (512, 2048), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf136, (512,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf137, (512,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf138, (512,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf139, (1536, 512), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf140, (1536,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf141, (512, 512), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf142, (512,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf143, (512,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf144, (512,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf145, (2048, 512), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf146, (2048,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf147, (512, 2048), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf148, (512,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf149, (512,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf150, (512,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf151, (1000, 512), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf152, (1000,), is_leaf=True)  # arg152_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)