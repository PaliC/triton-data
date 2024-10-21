
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1):
        convolution_3 = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [7, 7], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        add_96 = torch.ops.aten.add.Tensor(convolution_3, arg3_1);  convolution_3 = arg3_1 = None
        expand_1 = torch.ops.aten.expand.default(arg4_1, [8, -1, -1]);  arg4_1 = None
        view_140 = torch.ops.aten.view.default(add_96, [8, 256, 961]);  add_96 = None
        permute_87 = torch.ops.aten.permute.default(view_140, [0, 2, 1]);  view_140 = None
        cat_3 = torch.ops.aten.cat.default([expand_1, permute_87], 1);  expand_1 = permute_87 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
        getitem_145 = var_mean_27[0]
        getitem_146 = var_mean_27[1];  var_mean_27 = None
        add_97 = torch.ops.aten.add.Tensor(getitem_145, 1e-06);  getitem_145 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_27 = torch.ops.aten.sub.Tensor(cat_3, getitem_146);  getitem_146 = None
        mul_93 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
        mul_94 = torch.ops.aten.mul.Tensor(mul_93, arg5_1);  mul_93 = arg5_1 = None
        add_98 = torch.ops.aten.add.Tensor(mul_94, arg6_1);  mul_94 = arg6_1 = None
        view_141 = torch.ops.aten.view.default(add_98, [7696, 256]);  add_98 = None
        permute_88 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg8_1, view_141, permute_88);  arg8_1 = view_141 = permute_88 = None
        view_142 = torch.ops.aten.view.default(addmm_53, [8, 962, 768]);  addmm_53 = None
        view_143 = torch.ops.aten.view.default(view_142, [8, 962, 3, 4, 64]);  view_142 = None
        permute_89 = torch.ops.aten.permute.default(view_143, [2, 0, 3, 1, 4]);  view_143 = None
        unbind_13 = torch.ops.aten.unbind.int(permute_89);  permute_89 = None
        getitem_147 = unbind_13[0]
        getitem_148 = unbind_13[1]
        getitem_149 = unbind_13[2];  unbind_13 = None
        _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_147, getitem_148, getitem_149, None, False);  getitem_147 = getitem_148 = getitem_149 = None
        getitem_150 = _scaled_dot_product_efficient_attention_13[0];  _scaled_dot_product_efficient_attention_13 = None
        permute_90 = torch.ops.aten.permute.default(getitem_150, [0, 2, 1, 3]);  getitem_150 = None
        view_144 = torch.ops.aten.view.default(permute_90, [8, 962, 256]);  permute_90 = None
        view_145 = torch.ops.aten.view.default(view_144, [7696, 256]);  view_144 = None
        permute_91 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg10_1, view_145, permute_91);  arg10_1 = view_145 = permute_91 = None
        view_146 = torch.ops.aten.view.default(addmm_54, [8, 962, 256]);  addmm_54 = None
        add_99 = torch.ops.aten.add.Tensor(cat_3, view_146);  cat_3 = view_146 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_154 = var_mean_28[0]
        getitem_155 = var_mean_28[1];  var_mean_28 = None
        add_100 = torch.ops.aten.add.Tensor(getitem_154, 1e-06);  getitem_154 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_28 = torch.ops.aten.sub.Tensor(add_99, getitem_155);  getitem_155 = None
        mul_95 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
        mul_96 = torch.ops.aten.mul.Tensor(mul_95, arg11_1);  mul_95 = arg11_1 = None
        add_101 = torch.ops.aten.add.Tensor(mul_96, arg12_1);  mul_96 = arg12_1 = None
        view_147 = torch.ops.aten.view.default(add_101, [7696, 256]);  add_101 = None
        permute_92 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg14_1, view_147, permute_92);  arg14_1 = view_147 = permute_92 = None
        view_148 = torch.ops.aten.view.default(addmm_55, [8, 962, 1024]);  addmm_55 = None
        mul_97 = torch.ops.aten.mul.Tensor(view_148, 0.5)
        mul_98 = torch.ops.aten.mul.Tensor(view_148, 0.7071067811865476);  view_148 = None
        erf_13 = torch.ops.aten.erf.default(mul_98);  mul_98 = None
        add_102 = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_99 = torch.ops.aten.mul.Tensor(mul_97, add_102);  mul_97 = add_102 = None
        view_149 = torch.ops.aten.view.default(mul_99, [7696, 1024]);  mul_99 = None
        permute_93 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg16_1, view_149, permute_93);  arg16_1 = view_149 = permute_93 = None
        view_150 = torch.ops.aten.view.default(addmm_56, [8, 962, 256]);  addmm_56 = None
        add_103 = torch.ops.aten.add.Tensor(add_99, view_150);  add_99 = view_150 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
        getitem_156 = var_mean_29[0]
        getitem_157 = var_mean_29[1];  var_mean_29 = None
        add_104 = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_103, getitem_157);  getitem_157 = None
        mul_100 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, arg17_1);  mul_100 = arg17_1 = None
        add_105 = torch.ops.aten.add.Tensor(mul_101, arg18_1);  mul_101 = arg18_1 = None
        view_151 = torch.ops.aten.view.default(add_105, [7696, 256]);  add_105 = None
        permute_94 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg20_1, view_151, permute_94);  arg20_1 = view_151 = permute_94 = None
        view_152 = torch.ops.aten.view.default(addmm_57, [8, 962, 768]);  addmm_57 = None
        view_153 = torch.ops.aten.view.default(view_152, [8, 962, 3, 4, 64]);  view_152 = None
        permute_95 = torch.ops.aten.permute.default(view_153, [2, 0, 3, 1, 4]);  view_153 = None
        unbind_14 = torch.ops.aten.unbind.int(permute_95);  permute_95 = None
        getitem_158 = unbind_14[0]
        getitem_159 = unbind_14[1]
        getitem_160 = unbind_14[2];  unbind_14 = None
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_158, getitem_159, getitem_160, None, False);  getitem_158 = getitem_159 = getitem_160 = None
        getitem_161 = _scaled_dot_product_efficient_attention_14[0];  _scaled_dot_product_efficient_attention_14 = None
        permute_96 = torch.ops.aten.permute.default(getitem_161, [0, 2, 1, 3]);  getitem_161 = None
        view_154 = torch.ops.aten.view.default(permute_96, [8, 962, 256]);  permute_96 = None
        view_155 = torch.ops.aten.view.default(view_154, [7696, 256]);  view_154 = None
        permute_97 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg22_1, view_155, permute_97);  arg22_1 = view_155 = permute_97 = None
        view_156 = torch.ops.aten.view.default(addmm_58, [8, 962, 256]);  addmm_58 = None
        add_106 = torch.ops.aten.add.Tensor(add_103, view_156);  add_103 = view_156 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_106, [2], correction = 0, keepdim = True)
        getitem_165 = var_mean_30[0]
        getitem_166 = var_mean_30[1];  var_mean_30 = None
        add_107 = torch.ops.aten.add.Tensor(getitem_165, 1e-06);  getitem_165 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_106, getitem_166);  getitem_166 = None
        mul_102 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
        mul_103 = torch.ops.aten.mul.Tensor(mul_102, arg23_1);  mul_102 = arg23_1 = None
        add_108 = torch.ops.aten.add.Tensor(mul_103, arg24_1);  mul_103 = arg24_1 = None
        view_157 = torch.ops.aten.view.default(add_108, [7696, 256]);  add_108 = None
        permute_98 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg26_1, view_157, permute_98);  arg26_1 = view_157 = permute_98 = None
        view_158 = torch.ops.aten.view.default(addmm_59, [8, 962, 1024]);  addmm_59 = None
        mul_104 = torch.ops.aten.mul.Tensor(view_158, 0.5)
        mul_105 = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
        erf_14 = torch.ops.aten.erf.default(mul_105);  mul_105 = None
        add_109 = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_106 = torch.ops.aten.mul.Tensor(mul_104, add_109);  mul_104 = add_109 = None
        view_159 = torch.ops.aten.view.default(mul_106, [7696, 1024]);  mul_106 = None
        permute_99 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg28_1, view_159, permute_99);  arg28_1 = view_159 = permute_99 = None
        view_160 = torch.ops.aten.view.default(addmm_60, [8, 962, 256]);  addmm_60 = None
        add_110 = torch.ops.aten.add.Tensor(add_106, view_160);  add_106 = view_160 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
        getitem_167 = var_mean_31[0]
        getitem_168 = var_mean_31[1];  var_mean_31 = None
        add_111 = torch.ops.aten.add.Tensor(getitem_167, 1e-06);  getitem_167 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_31 = torch.ops.aten.sub.Tensor(add_110, getitem_168);  getitem_168 = None
        mul_107 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
        mul_108 = torch.ops.aten.mul.Tensor(mul_107, arg29_1);  mul_107 = arg29_1 = None
        add_112 = torch.ops.aten.add.Tensor(mul_108, arg30_1);  mul_108 = arg30_1 = None
        view_161 = torch.ops.aten.view.default(add_112, [7696, 256]);  add_112 = None
        permute_100 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg32_1, view_161, permute_100);  arg32_1 = view_161 = permute_100 = None
        view_162 = torch.ops.aten.view.default(addmm_61, [8, 962, 768]);  addmm_61 = None
        view_163 = torch.ops.aten.view.default(view_162, [8, 962, 3, 4, 64]);  view_162 = None
        permute_101 = torch.ops.aten.permute.default(view_163, [2, 0, 3, 1, 4]);  view_163 = None
        unbind_15 = torch.ops.aten.unbind.int(permute_101);  permute_101 = None
        getitem_169 = unbind_15[0]
        getitem_170 = unbind_15[1]
        getitem_171 = unbind_15[2];  unbind_15 = None
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_169, getitem_170, getitem_171, None, False);  getitem_169 = getitem_170 = getitem_171 = None
        getitem_172 = _scaled_dot_product_efficient_attention_15[0];  _scaled_dot_product_efficient_attention_15 = None
        permute_102 = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
        view_164 = torch.ops.aten.view.default(permute_102, [8, 962, 256]);  permute_102 = None
        view_165 = torch.ops.aten.view.default(view_164, [7696, 256]);  view_164 = None
        permute_103 = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg34_1, view_165, permute_103);  arg34_1 = view_165 = permute_103 = None
        view_166 = torch.ops.aten.view.default(addmm_62, [8, 962, 256]);  addmm_62 = None
        add_113 = torch.ops.aten.add.Tensor(add_110, view_166);  add_110 = view_166 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
        getitem_176 = var_mean_32[0]
        getitem_177 = var_mean_32[1];  var_mean_32 = None
        add_114 = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_32 = torch.ops.aten.sub.Tensor(add_113, getitem_177);  getitem_177 = None
        mul_109 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
        mul_110 = torch.ops.aten.mul.Tensor(mul_109, arg35_1);  mul_109 = arg35_1 = None
        add_115 = torch.ops.aten.add.Tensor(mul_110, arg36_1);  mul_110 = arg36_1 = None
        view_167 = torch.ops.aten.view.default(add_115, [7696, 256]);  add_115 = None
        permute_104 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg38_1, view_167, permute_104);  arg38_1 = view_167 = permute_104 = None
        view_168 = torch.ops.aten.view.default(addmm_63, [8, 962, 1024]);  addmm_63 = None
        mul_111 = torch.ops.aten.mul.Tensor(view_168, 0.5)
        mul_112 = torch.ops.aten.mul.Tensor(view_168, 0.7071067811865476);  view_168 = None
        erf_15 = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_116 = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_111, add_116);  mul_111 = add_116 = None
        view_169 = torch.ops.aten.view.default(mul_113, [7696, 1024]);  mul_113 = None
        permute_105 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg40_1, view_169, permute_105);  arg40_1 = view_169 = permute_105 = None
        view_170 = torch.ops.aten.view.default(addmm_64, [8, 962, 256]);  addmm_64 = None
        add_117 = torch.ops.aten.add.Tensor(add_113, view_170);  add_113 = view_170 = None
        slice_15 = torch.ops.aten.slice.Tensor(add_117, 1, 0, 1)
        slice_17 = torch.ops.aten.slice.Tensor(add_117, 1, 1, 9223372036854775807);  add_117 = None
        permute_106 = torch.ops.aten.permute.default(slice_17, [0, 2, 1]);  slice_17 = None
        view_171 = torch.ops.aten.view.default(permute_106, [8, 256, 31, 31]);  permute_106 = None
        convolution_4 = torch.ops.aten.convolution.default(view_171, arg41_1, arg42_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 256);  view_171 = arg41_1 = arg42_1 = None
        permute_107 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        view_172 = torch.ops.aten.view.default(slice_15, [8, 256]);  slice_15 = None
        mm_2 = torch.ops.aten.mm.default(view_172, permute_107);  view_172 = permute_107 = None
        view_173 = torch.ops.aten.view.default(mm_2, [8, 1, 512]);  mm_2 = None
        add_118 = torch.ops.aten.add.Tensor(view_173, arg44_1);  view_173 = arg44_1 = None
        view_174 = torch.ops.aten.view.default(convolution_4, [8, 512, 256]);  convolution_4 = None
        permute_108 = torch.ops.aten.permute.default(view_174, [0, 2, 1]);  view_174 = None
        cat_4 = torch.ops.aten.cat.default([add_118, permute_108], 1);  add_118 = permute_108 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(cat_4, [2], correction = 0, keepdim = True)
        getitem_178 = var_mean_33[0]
        getitem_179 = var_mean_33[1];  var_mean_33 = None
        add_119 = torch.ops.aten.add.Tensor(getitem_178, 1e-06);  getitem_178 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_33 = torch.ops.aten.sub.Tensor(cat_4, getitem_179);  getitem_179 = None
        mul_114 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
        mul_115 = torch.ops.aten.mul.Tensor(mul_114, arg45_1);  mul_114 = arg45_1 = None
        add_120 = torch.ops.aten.add.Tensor(mul_115, arg46_1);  mul_115 = arg46_1 = None
        view_175 = torch.ops.aten.view.default(add_120, [2056, 512]);  add_120 = None
        permute_109 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg48_1, view_175, permute_109);  arg48_1 = view_175 = permute_109 = None
        view_176 = torch.ops.aten.view.default(addmm_65, [8, 257, 1536]);  addmm_65 = None
        view_177 = torch.ops.aten.view.default(view_176, [8, 257, 3, 8, 64]);  view_176 = None
        permute_110 = torch.ops.aten.permute.default(view_177, [2, 0, 3, 1, 4]);  view_177 = None
        unbind_16 = torch.ops.aten.unbind.int(permute_110);  permute_110 = None
        getitem_180 = unbind_16[0]
        getitem_181 = unbind_16[1]
        getitem_182 = unbind_16[2];  unbind_16 = None
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_180, getitem_181, getitem_182, None, False);  getitem_180 = getitem_181 = getitem_182 = None
        getitem_183 = _scaled_dot_product_efficient_attention_16[0];  _scaled_dot_product_efficient_attention_16 = None
        permute_111 = torch.ops.aten.permute.default(getitem_183, [0, 2, 1, 3]);  getitem_183 = None
        view_178 = torch.ops.aten.view.default(permute_111, [8, 257, 512]);  permute_111 = None
        view_179 = torch.ops.aten.view.default(view_178, [2056, 512]);  view_178 = None
        permute_112 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg50_1, view_179, permute_112);  arg50_1 = view_179 = permute_112 = None
        view_180 = torch.ops.aten.view.default(addmm_66, [8, 257, 512]);  addmm_66 = None
        add_121 = torch.ops.aten.add.Tensor(cat_4, view_180);  cat_4 = view_180 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
        getitem_187 = var_mean_34[0]
        getitem_188 = var_mean_34[1];  var_mean_34 = None
        add_122 = torch.ops.aten.add.Tensor(getitem_187, 1e-06);  getitem_187 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_34 = torch.ops.aten.sub.Tensor(add_121, getitem_188);  getitem_188 = None
        mul_116 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_116, arg51_1);  mul_116 = arg51_1 = None
        add_123 = torch.ops.aten.add.Tensor(mul_117, arg52_1);  mul_117 = arg52_1 = None
        view_181 = torch.ops.aten.view.default(add_123, [2056, 512]);  add_123 = None
        permute_113 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg54_1, view_181, permute_113);  arg54_1 = view_181 = permute_113 = None
        view_182 = torch.ops.aten.view.default(addmm_67, [8, 257, 2048]);  addmm_67 = None
        mul_118 = torch.ops.aten.mul.Tensor(view_182, 0.5)
        mul_119 = torch.ops.aten.mul.Tensor(view_182, 0.7071067811865476);  view_182 = None
        erf_16 = torch.ops.aten.erf.default(mul_119);  mul_119 = None
        add_124 = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_120 = torch.ops.aten.mul.Tensor(mul_118, add_124);  mul_118 = add_124 = None
        view_183 = torch.ops.aten.view.default(mul_120, [2056, 2048]);  mul_120 = None
        permute_114 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg56_1, view_183, permute_114);  arg56_1 = view_183 = permute_114 = None
        view_184 = torch.ops.aten.view.default(addmm_68, [8, 257, 512]);  addmm_68 = None
        add_125 = torch.ops.aten.add.Tensor(add_121, view_184);  add_121 = view_184 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_189 = var_mean_35[0]
        getitem_190 = var_mean_35[1];  var_mean_35 = None
        add_126 = torch.ops.aten.add.Tensor(getitem_189, 1e-06);  getitem_189 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_35 = torch.ops.aten.sub.Tensor(add_125, getitem_190);  getitem_190 = None
        mul_121 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
        mul_122 = torch.ops.aten.mul.Tensor(mul_121, arg57_1);  mul_121 = arg57_1 = None
        add_127 = torch.ops.aten.add.Tensor(mul_122, arg58_1);  mul_122 = arg58_1 = None
        view_185 = torch.ops.aten.view.default(add_127, [2056, 512]);  add_127 = None
        permute_115 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg60_1, view_185, permute_115);  arg60_1 = view_185 = permute_115 = None
        view_186 = torch.ops.aten.view.default(addmm_69, [8, 257, 1536]);  addmm_69 = None
        view_187 = torch.ops.aten.view.default(view_186, [8, 257, 3, 8, 64]);  view_186 = None
        permute_116 = torch.ops.aten.permute.default(view_187, [2, 0, 3, 1, 4]);  view_187 = None
        unbind_17 = torch.ops.aten.unbind.int(permute_116);  permute_116 = None
        getitem_191 = unbind_17[0]
        getitem_192 = unbind_17[1]
        getitem_193 = unbind_17[2];  unbind_17 = None
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_191, getitem_192, getitem_193, None, False);  getitem_191 = getitem_192 = getitem_193 = None
        getitem_194 = _scaled_dot_product_efficient_attention_17[0];  _scaled_dot_product_efficient_attention_17 = None
        permute_117 = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3]);  getitem_194 = None
        view_188 = torch.ops.aten.view.default(permute_117, [8, 257, 512]);  permute_117 = None
        view_189 = torch.ops.aten.view.default(view_188, [2056, 512]);  view_188 = None
        permute_118 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg62_1, view_189, permute_118);  arg62_1 = view_189 = permute_118 = None
        view_190 = torch.ops.aten.view.default(addmm_70, [8, 257, 512]);  addmm_70 = None
        add_128 = torch.ops.aten.add.Tensor(add_125, view_190);  add_125 = view_190 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
        getitem_198 = var_mean_36[0]
        getitem_199 = var_mean_36[1];  var_mean_36 = None
        add_129 = torch.ops.aten.add.Tensor(getitem_198, 1e-06);  getitem_198 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_128, getitem_199);  getitem_199 = None
        mul_123 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
        mul_124 = torch.ops.aten.mul.Tensor(mul_123, arg63_1);  mul_123 = arg63_1 = None
        add_130 = torch.ops.aten.add.Tensor(mul_124, arg64_1);  mul_124 = arg64_1 = None
        view_191 = torch.ops.aten.view.default(add_130, [2056, 512]);  add_130 = None
        permute_119 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg66_1, view_191, permute_119);  arg66_1 = view_191 = permute_119 = None
        view_192 = torch.ops.aten.view.default(addmm_71, [8, 257, 2048]);  addmm_71 = None
        mul_125 = torch.ops.aten.mul.Tensor(view_192, 0.5)
        mul_126 = torch.ops.aten.mul.Tensor(view_192, 0.7071067811865476);  view_192 = None
        erf_17 = torch.ops.aten.erf.default(mul_126);  mul_126 = None
        add_131 = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_127 = torch.ops.aten.mul.Tensor(mul_125, add_131);  mul_125 = add_131 = None
        view_193 = torch.ops.aten.view.default(mul_127, [2056, 2048]);  mul_127 = None
        permute_120 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg68_1, view_193, permute_120);  arg68_1 = view_193 = permute_120 = None
        view_194 = torch.ops.aten.view.default(addmm_72, [8, 257, 512]);  addmm_72 = None
        add_132 = torch.ops.aten.add.Tensor(add_128, view_194);  add_128 = view_194 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_132, [2], correction = 0, keepdim = True)
        getitem_200 = var_mean_37[0]
        getitem_201 = var_mean_37[1];  var_mean_37 = None
        add_133 = torch.ops.aten.add.Tensor(getitem_200, 1e-06);  getitem_200 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_37 = torch.ops.aten.sub.Tensor(add_132, getitem_201);  getitem_201 = None
        mul_128 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_128, arg69_1);  mul_128 = arg69_1 = None
        add_134 = torch.ops.aten.add.Tensor(mul_129, arg70_1);  mul_129 = arg70_1 = None
        view_195 = torch.ops.aten.view.default(add_134, [2056, 512]);  add_134 = None
        permute_121 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg72_1, view_195, permute_121);  arg72_1 = view_195 = permute_121 = None
        view_196 = torch.ops.aten.view.default(addmm_73, [8, 257, 1536]);  addmm_73 = None
        view_197 = torch.ops.aten.view.default(view_196, [8, 257, 3, 8, 64]);  view_196 = None
        permute_122 = torch.ops.aten.permute.default(view_197, [2, 0, 3, 1, 4]);  view_197 = None
        unbind_18 = torch.ops.aten.unbind.int(permute_122);  permute_122 = None
        getitem_202 = unbind_18[0]
        getitem_203 = unbind_18[1]
        getitem_204 = unbind_18[2];  unbind_18 = None
        _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_202, getitem_203, getitem_204, None, False);  getitem_202 = getitem_203 = getitem_204 = None
        getitem_205 = _scaled_dot_product_efficient_attention_18[0];  _scaled_dot_product_efficient_attention_18 = None
        permute_123 = torch.ops.aten.permute.default(getitem_205, [0, 2, 1, 3]);  getitem_205 = None
        view_198 = torch.ops.aten.view.default(permute_123, [8, 257, 512]);  permute_123 = None
        view_199 = torch.ops.aten.view.default(view_198, [2056, 512]);  view_198 = None
        permute_124 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg74_1, view_199, permute_124);  arg74_1 = view_199 = permute_124 = None
        view_200 = torch.ops.aten.view.default(addmm_74, [8, 257, 512]);  addmm_74 = None
        add_135 = torch.ops.aten.add.Tensor(add_132, view_200);  add_132 = view_200 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_135, [2], correction = 0, keepdim = True)
        getitem_209 = var_mean_38[0]
        getitem_210 = var_mean_38[1];  var_mean_38 = None
        add_136 = torch.ops.aten.add.Tensor(getitem_209, 1e-06);  getitem_209 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        sub_38 = torch.ops.aten.sub.Tensor(add_135, getitem_210);  getitem_210 = None
        mul_130 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
        mul_131 = torch.ops.aten.mul.Tensor(mul_130, arg75_1);  mul_130 = arg75_1 = None
        add_137 = torch.ops.aten.add.Tensor(mul_131, arg76_1);  mul_131 = arg76_1 = None
        view_201 = torch.ops.aten.view.default(add_137, [2056, 512]);  add_137 = None
        permute_125 = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg78_1, view_201, permute_125);  arg78_1 = view_201 = permute_125 = None
        view_202 = torch.ops.aten.view.default(addmm_75, [8, 257, 2048]);  addmm_75 = None
        mul_132 = torch.ops.aten.mul.Tensor(view_202, 0.5)
        mul_133 = torch.ops.aten.mul.Tensor(view_202, 0.7071067811865476);  view_202 = None
        erf_18 = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_138 = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_132, add_138);  mul_132 = add_138 = None
        view_203 = torch.ops.aten.view.default(mul_134, [2056, 2048]);  mul_134 = None
        permute_126 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg80_1, view_203, permute_126);  arg80_1 = view_203 = permute_126 = None
        view_204 = torch.ops.aten.view.default(addmm_76, [8, 257, 512]);  addmm_76 = None
        add_139 = torch.ops.aten.add.Tensor(add_135, view_204);  add_135 = view_204 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
        getitem_211 = var_mean_39[0]
        getitem_212 = var_mean_39[1];  var_mean_39 = None
        add_140 = torch.ops.aten.add.Tensor(getitem_211, 1e-06);  getitem_211 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_139, getitem_212);  getitem_212 = None
        mul_135 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
        mul_136 = torch.ops.aten.mul.Tensor(mul_135, arg81_1);  mul_135 = arg81_1 = None
        add_141 = torch.ops.aten.add.Tensor(mul_136, arg82_1);  mul_136 = arg82_1 = None
        view_205 = torch.ops.aten.view.default(add_141, [2056, 512]);  add_141 = None
        permute_127 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg84_1, view_205, permute_127);  arg84_1 = view_205 = permute_127 = None
        view_206 = torch.ops.aten.view.default(addmm_77, [8, 257, 1536]);  addmm_77 = None
        view_207 = torch.ops.aten.view.default(view_206, [8, 257, 3, 8, 64]);  view_206 = None
        permute_128 = torch.ops.aten.permute.default(view_207, [2, 0, 3, 1, 4]);  view_207 = None
        unbind_19 = torch.ops.aten.unbind.int(permute_128);  permute_128 = None
        getitem_213 = unbind_19[0]
        getitem_214 = unbind_19[1]
        getitem_215 = unbind_19[2];  unbind_19 = None
        _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_213, getitem_214, getitem_215, None, False);  getitem_213 = getitem_214 = getitem_215 = None
        getitem_216 = _scaled_dot_product_efficient_attention_19[0];  _scaled_dot_product_efficient_attention_19 = None
        permute_129 = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None
        view_208 = torch.ops.aten.view.default(permute_129, [8, 257, 512]);  permute_129 = None
        view_209 = torch.ops.aten.view.default(view_208, [2056, 512]);  view_208 = None
        permute_130 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg86_1, view_209, permute_130);  arg86_1 = view_209 = permute_130 = None
        view_210 = torch.ops.aten.view.default(addmm_78, [8, 257, 512]);  addmm_78 = None
        add_142 = torch.ops.aten.add.Tensor(add_139, view_210);  add_139 = view_210 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_142, [2], correction = 0, keepdim = True)
        getitem_220 = var_mean_40[0]
        getitem_221 = var_mean_40[1];  var_mean_40 = None
        add_143 = torch.ops.aten.add.Tensor(getitem_220, 1e-06);  getitem_220 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        sub_40 = torch.ops.aten.sub.Tensor(add_142, getitem_221);  getitem_221 = None
        mul_137 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_137, arg87_1);  mul_137 = arg87_1 = None
        add_144 = torch.ops.aten.add.Tensor(mul_138, arg88_1);  mul_138 = arg88_1 = None
        view_211 = torch.ops.aten.view.default(add_144, [2056, 512]);  add_144 = None
        permute_131 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg90_1, view_211, permute_131);  arg90_1 = view_211 = permute_131 = None
        view_212 = torch.ops.aten.view.default(addmm_79, [8, 257, 2048]);  addmm_79 = None
        mul_139 = torch.ops.aten.mul.Tensor(view_212, 0.5)
        mul_140 = torch.ops.aten.mul.Tensor(view_212, 0.7071067811865476);  view_212 = None
        erf_19 = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_145 = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_139, add_145);  mul_139 = add_145 = None
        view_213 = torch.ops.aten.view.default(mul_141, [2056, 2048]);  mul_141 = None
        permute_132 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg92_1, view_213, permute_132);  arg92_1 = view_213 = permute_132 = None
        view_214 = torch.ops.aten.view.default(addmm_80, [8, 257, 512]);  addmm_80 = None
        add_146 = torch.ops.aten.add.Tensor(add_142, view_214);  add_142 = view_214 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_146, [2], correction = 0, keepdim = True)
        getitem_222 = var_mean_41[0]
        getitem_223 = var_mean_41[1];  var_mean_41 = None
        add_147 = torch.ops.aten.add.Tensor(getitem_222, 1e-06);  getitem_222 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        sub_41 = torch.ops.aten.sub.Tensor(add_146, getitem_223);  getitem_223 = None
        mul_142 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
        mul_143 = torch.ops.aten.mul.Tensor(mul_142, arg93_1);  mul_142 = arg93_1 = None
        add_148 = torch.ops.aten.add.Tensor(mul_143, arg94_1);  mul_143 = arg94_1 = None
        view_215 = torch.ops.aten.view.default(add_148, [2056, 512]);  add_148 = None
        permute_133 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg96_1, view_215, permute_133);  arg96_1 = view_215 = permute_133 = None
        view_216 = torch.ops.aten.view.default(addmm_81, [8, 257, 1536]);  addmm_81 = None
        view_217 = torch.ops.aten.view.default(view_216, [8, 257, 3, 8, 64]);  view_216 = None
        permute_134 = torch.ops.aten.permute.default(view_217, [2, 0, 3, 1, 4]);  view_217 = None
        unbind_20 = torch.ops.aten.unbind.int(permute_134);  permute_134 = None
        getitem_224 = unbind_20[0]
        getitem_225 = unbind_20[1]
        getitem_226 = unbind_20[2];  unbind_20 = None
        _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_224, getitem_225, getitem_226, None, False);  getitem_224 = getitem_225 = getitem_226 = None
        getitem_227 = _scaled_dot_product_efficient_attention_20[0];  _scaled_dot_product_efficient_attention_20 = None
        permute_135 = torch.ops.aten.permute.default(getitem_227, [0, 2, 1, 3]);  getitem_227 = None
        view_218 = torch.ops.aten.view.default(permute_135, [8, 257, 512]);  permute_135 = None
        view_219 = torch.ops.aten.view.default(view_218, [2056, 512]);  view_218 = None
        permute_136 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg98_1, view_219, permute_136);  arg98_1 = view_219 = permute_136 = None
        view_220 = torch.ops.aten.view.default(addmm_82, [8, 257, 512]);  addmm_82 = None
        add_149 = torch.ops.aten.add.Tensor(add_146, view_220);  add_146 = view_220 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(add_149, [2], correction = 0, keepdim = True)
        getitem_231 = var_mean_42[0]
        getitem_232 = var_mean_42[1];  var_mean_42 = None
        add_150 = torch.ops.aten.add.Tensor(getitem_231, 1e-06);  getitem_231 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        sub_42 = torch.ops.aten.sub.Tensor(add_149, getitem_232);  getitem_232 = None
        mul_144 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_144, arg99_1);  mul_144 = arg99_1 = None
        add_151 = torch.ops.aten.add.Tensor(mul_145, arg100_1);  mul_145 = arg100_1 = None
        view_221 = torch.ops.aten.view.default(add_151, [2056, 512]);  add_151 = None
        permute_137 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg102_1, view_221, permute_137);  arg102_1 = view_221 = permute_137 = None
        view_222 = torch.ops.aten.view.default(addmm_83, [8, 257, 2048]);  addmm_83 = None
        mul_146 = torch.ops.aten.mul.Tensor(view_222, 0.5)
        mul_147 = torch.ops.aten.mul.Tensor(view_222, 0.7071067811865476);  view_222 = None
        erf_20 = torch.ops.aten.erf.default(mul_147);  mul_147 = None
        add_152 = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_148 = torch.ops.aten.mul.Tensor(mul_146, add_152);  mul_146 = add_152 = None
        view_223 = torch.ops.aten.view.default(mul_148, [2056, 2048]);  mul_148 = None
        permute_138 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg104_1, view_223, permute_138);  arg104_1 = view_223 = permute_138 = None
        view_224 = torch.ops.aten.view.default(addmm_84, [8, 257, 512]);  addmm_84 = None
        add_153 = torch.ops.aten.add.Tensor(add_149, view_224);  add_149 = view_224 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
        getitem_233 = var_mean_43[0]
        getitem_234 = var_mean_43[1];  var_mean_43 = None
        add_154 = torch.ops.aten.add.Tensor(getitem_233, 1e-06);  getitem_233 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_43 = torch.ops.aten.sub.Tensor(add_153, getitem_234);  getitem_234 = None
        mul_149 = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
        mul_150 = torch.ops.aten.mul.Tensor(mul_149, arg105_1);  mul_149 = arg105_1 = None
        add_155 = torch.ops.aten.add.Tensor(mul_150, arg106_1);  mul_150 = arg106_1 = None
        view_225 = torch.ops.aten.view.default(add_155, [2056, 512]);  add_155 = None
        permute_139 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg108_1, view_225, permute_139);  arg108_1 = view_225 = permute_139 = None
        view_226 = torch.ops.aten.view.default(addmm_85, [8, 257, 1536]);  addmm_85 = None
        view_227 = torch.ops.aten.view.default(view_226, [8, 257, 3, 8, 64]);  view_226 = None
        permute_140 = torch.ops.aten.permute.default(view_227, [2, 0, 3, 1, 4]);  view_227 = None
        unbind_21 = torch.ops.aten.unbind.int(permute_140);  permute_140 = None
        getitem_235 = unbind_21[0]
        getitem_236 = unbind_21[1]
        getitem_237 = unbind_21[2];  unbind_21 = None
        _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_235, getitem_236, getitem_237, None, False);  getitem_235 = getitem_236 = getitem_237 = None
        getitem_238 = _scaled_dot_product_efficient_attention_21[0];  _scaled_dot_product_efficient_attention_21 = None
        permute_141 = torch.ops.aten.permute.default(getitem_238, [0, 2, 1, 3]);  getitem_238 = None
        view_228 = torch.ops.aten.view.default(permute_141, [8, 257, 512]);  permute_141 = None
        view_229 = torch.ops.aten.view.default(view_228, [2056, 512]);  view_228 = None
        permute_142 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg110_1, view_229, permute_142);  arg110_1 = view_229 = permute_142 = None
        view_230 = torch.ops.aten.view.default(addmm_86, [8, 257, 512]);  addmm_86 = None
        add_156 = torch.ops.aten.add.Tensor(add_153, view_230);  add_153 = view_230 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(add_156, [2], correction = 0, keepdim = True)
        getitem_242 = var_mean_44[0]
        getitem_243 = var_mean_44[1];  var_mean_44 = None
        add_157 = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        sub_44 = torch.ops.aten.sub.Tensor(add_156, getitem_243);  getitem_243 = None
        mul_151 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
        mul_152 = torch.ops.aten.mul.Tensor(mul_151, arg111_1);  mul_151 = arg111_1 = None
        add_158 = torch.ops.aten.add.Tensor(mul_152, arg112_1);  mul_152 = arg112_1 = None
        view_231 = torch.ops.aten.view.default(add_158, [2056, 512]);  add_158 = None
        permute_143 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg114_1, view_231, permute_143);  arg114_1 = view_231 = permute_143 = None
        view_232 = torch.ops.aten.view.default(addmm_87, [8, 257, 2048]);  addmm_87 = None
        mul_153 = torch.ops.aten.mul.Tensor(view_232, 0.5)
        mul_154 = torch.ops.aten.mul.Tensor(view_232, 0.7071067811865476);  view_232 = None
        erf_21 = torch.ops.aten.erf.default(mul_154);  mul_154 = None
        add_159 = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_155 = torch.ops.aten.mul.Tensor(mul_153, add_159);  mul_153 = add_159 = None
        view_233 = torch.ops.aten.view.default(mul_155, [2056, 2048]);  mul_155 = None
        permute_144 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg116_1, view_233, permute_144);  arg116_1 = view_233 = permute_144 = None
        view_234 = torch.ops.aten.view.default(addmm_88, [8, 257, 512]);  addmm_88 = None
        add_160 = torch.ops.aten.add.Tensor(add_156, view_234);  add_156 = view_234 = None
        slice_19 = torch.ops.aten.slice.Tensor(add_160, 1, 0, 1)
        slice_21 = torch.ops.aten.slice.Tensor(add_160, 1, 1, 9223372036854775807);  add_160 = None
        permute_145 = torch.ops.aten.permute.default(slice_21, [0, 2, 1]);  slice_21 = None
        view_235 = torch.ops.aten.view.default(permute_145, [8, 512, 16, 16]);  permute_145 = None
        convolution_5 = torch.ops.aten.convolution.default(view_235, arg117_1, arg118_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 512);  view_235 = arg117_1 = arg118_1 = None
        permute_146 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        view_236 = torch.ops.aten.view.default(slice_19, [8, 512]);  slice_19 = None
        mm_3 = torch.ops.aten.mm.default(view_236, permute_146);  view_236 = permute_146 = None
        view_237 = torch.ops.aten.view.default(mm_3, [8, 1, 1024]);  mm_3 = None
        add_161 = torch.ops.aten.add.Tensor(view_237, arg120_1);  view_237 = arg120_1 = None
        view_238 = torch.ops.aten.view.default(convolution_5, [8, 1024, 64]);  convolution_5 = None
        permute_147 = torch.ops.aten.permute.default(view_238, [0, 2, 1]);  view_238 = None
        cat_5 = torch.ops.aten.cat.default([add_161, permute_147], 1);  add_161 = permute_147 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(cat_5, [2], correction = 0, keepdim = True)
        getitem_244 = var_mean_45[0]
        getitem_245 = var_mean_45[1];  var_mean_45 = None
        add_162 = torch.ops.aten.add.Tensor(getitem_244, 1e-06);  getitem_244 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_45 = torch.ops.aten.sub.Tensor(cat_5, getitem_245);  getitem_245 = None
        mul_156 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_156, arg121_1);  mul_156 = arg121_1 = None
        add_163 = torch.ops.aten.add.Tensor(mul_157, arg122_1);  mul_157 = arg122_1 = None
        view_239 = torch.ops.aten.view.default(add_163, [520, 1024]);  add_163 = None
        permute_148 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg124_1, view_239, permute_148);  arg124_1 = view_239 = permute_148 = None
        view_240 = torch.ops.aten.view.default(addmm_89, [8, 65, 3072]);  addmm_89 = None
        view_241 = torch.ops.aten.view.default(view_240, [8, 65, 3, 16, 64]);  view_240 = None
        permute_149 = torch.ops.aten.permute.default(view_241, [2, 0, 3, 1, 4]);  view_241 = None
        unbind_22 = torch.ops.aten.unbind.int(permute_149);  permute_149 = None
        getitem_246 = unbind_22[0]
        getitem_247 = unbind_22[1]
        getitem_248 = unbind_22[2];  unbind_22 = None
        _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_246, getitem_247, getitem_248, None, False);  getitem_246 = getitem_247 = getitem_248 = None
        getitem_249 = _scaled_dot_product_efficient_attention_22[0];  _scaled_dot_product_efficient_attention_22 = None
        permute_150 = torch.ops.aten.permute.default(getitem_249, [0, 2, 1, 3]);  getitem_249 = None
        view_242 = torch.ops.aten.view.default(permute_150, [8, 65, 1024]);  permute_150 = None
        view_243 = torch.ops.aten.view.default(view_242, [520, 1024]);  view_242 = None
        permute_151 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg126_1, view_243, permute_151);  arg126_1 = view_243 = permute_151 = None
        view_244 = torch.ops.aten.view.default(addmm_90, [8, 65, 1024]);  addmm_90 = None
        add_164 = torch.ops.aten.add.Tensor(cat_5, view_244);  cat_5 = view_244 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
        getitem_253 = var_mean_46[0]
        getitem_254 = var_mean_46[1];  var_mean_46 = None
        add_165 = torch.ops.aten.add.Tensor(getitem_253, 1e-06);  getitem_253 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_46 = torch.ops.aten.sub.Tensor(add_164, getitem_254);  getitem_254 = None
        mul_158 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
        mul_159 = torch.ops.aten.mul.Tensor(mul_158, arg127_1);  mul_158 = arg127_1 = None
        add_166 = torch.ops.aten.add.Tensor(mul_159, arg128_1);  mul_159 = arg128_1 = None
        view_245 = torch.ops.aten.view.default(add_166, [520, 1024]);  add_166 = None
        permute_152 = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg130_1, view_245, permute_152);  arg130_1 = view_245 = permute_152 = None
        view_246 = torch.ops.aten.view.default(addmm_91, [8, 65, 4096]);  addmm_91 = None
        mul_160 = torch.ops.aten.mul.Tensor(view_246, 0.5)
        mul_161 = torch.ops.aten.mul.Tensor(view_246, 0.7071067811865476);  view_246 = None
        erf_22 = torch.ops.aten.erf.default(mul_161);  mul_161 = None
        add_167 = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_162 = torch.ops.aten.mul.Tensor(mul_160, add_167);  mul_160 = add_167 = None
        view_247 = torch.ops.aten.view.default(mul_162, [520, 4096]);  mul_162 = None
        permute_153 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg132_1, view_247, permute_153);  arg132_1 = view_247 = permute_153 = None
        view_248 = torch.ops.aten.view.default(addmm_92, [8, 65, 1024]);  addmm_92 = None
        add_168 = torch.ops.aten.add.Tensor(add_164, view_248);  add_164 = view_248 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_168, [2], correction = 0, keepdim = True)
        getitem_255 = var_mean_47[0]
        getitem_256 = var_mean_47[1];  var_mean_47 = None
        add_169 = torch.ops.aten.add.Tensor(getitem_255, 1e-06);  getitem_255 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
        sub_47 = torch.ops.aten.sub.Tensor(add_168, getitem_256);  getitem_256 = None
        mul_163 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
        mul_164 = torch.ops.aten.mul.Tensor(mul_163, arg133_1);  mul_163 = arg133_1 = None
        add_170 = torch.ops.aten.add.Tensor(mul_164, arg134_1);  mul_164 = arg134_1 = None
        view_249 = torch.ops.aten.view.default(add_170, [520, 1024]);  add_170 = None
        permute_154 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg136_1, view_249, permute_154);  arg136_1 = view_249 = permute_154 = None
        view_250 = torch.ops.aten.view.default(addmm_93, [8, 65, 3072]);  addmm_93 = None
        view_251 = torch.ops.aten.view.default(view_250, [8, 65, 3, 16, 64]);  view_250 = None
        permute_155 = torch.ops.aten.permute.default(view_251, [2, 0, 3, 1, 4]);  view_251 = None
        unbind_23 = torch.ops.aten.unbind.int(permute_155);  permute_155 = None
        getitem_257 = unbind_23[0]
        getitem_258 = unbind_23[1]
        getitem_259 = unbind_23[2];  unbind_23 = None
        _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_257, getitem_258, getitem_259, None, False);  getitem_257 = getitem_258 = getitem_259 = None
        getitem_260 = _scaled_dot_product_efficient_attention_23[0];  _scaled_dot_product_efficient_attention_23 = None
        permute_156 = torch.ops.aten.permute.default(getitem_260, [0, 2, 1, 3]);  getitem_260 = None
        view_252 = torch.ops.aten.view.default(permute_156, [8, 65, 1024]);  permute_156 = None
        view_253 = torch.ops.aten.view.default(view_252, [520, 1024]);  view_252 = None
        permute_157 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg138_1, view_253, permute_157);  arg138_1 = view_253 = permute_157 = None
        view_254 = torch.ops.aten.view.default(addmm_94, [8, 65, 1024]);  addmm_94 = None
        add_171 = torch.ops.aten.add.Tensor(add_168, view_254);  add_168 = view_254 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(add_171, [2], correction = 0, keepdim = True)
        getitem_264 = var_mean_48[0]
        getitem_265 = var_mean_48[1];  var_mean_48 = None
        add_172 = torch.ops.aten.add.Tensor(getitem_264, 1e-06);  getitem_264 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        sub_48 = torch.ops.aten.sub.Tensor(add_171, getitem_265);  getitem_265 = None
        mul_165 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
        mul_166 = torch.ops.aten.mul.Tensor(mul_165, arg139_1);  mul_165 = arg139_1 = None
        add_173 = torch.ops.aten.add.Tensor(mul_166, arg140_1);  mul_166 = arg140_1 = None
        view_255 = torch.ops.aten.view.default(add_173, [520, 1024]);  add_173 = None
        permute_158 = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg142_1, view_255, permute_158);  arg142_1 = view_255 = permute_158 = None
        view_256 = torch.ops.aten.view.default(addmm_95, [8, 65, 4096]);  addmm_95 = None
        mul_167 = torch.ops.aten.mul.Tensor(view_256, 0.5)
        mul_168 = torch.ops.aten.mul.Tensor(view_256, 0.7071067811865476);  view_256 = None
        erf_23 = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_174 = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_167, add_174);  mul_167 = add_174 = None
        view_257 = torch.ops.aten.view.default(mul_169, [520, 4096]);  mul_169 = None
        permute_159 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg144_1, view_257, permute_159);  arg144_1 = view_257 = permute_159 = None
        view_258 = torch.ops.aten.view.default(addmm_96, [8, 65, 1024]);  addmm_96 = None
        add_175 = torch.ops.aten.add.Tensor(add_171, view_258);  add_171 = view_258 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(add_175, [2], correction = 0, keepdim = True)
        getitem_266 = var_mean_49[0]
        getitem_267 = var_mean_49[1];  var_mean_49 = None
        add_176 = torch.ops.aten.add.Tensor(getitem_266, 1e-06);  getitem_266 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        sub_49 = torch.ops.aten.sub.Tensor(add_175, getitem_267);  getitem_267 = None
        mul_170 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
        mul_171 = torch.ops.aten.mul.Tensor(mul_170, arg145_1);  mul_170 = arg145_1 = None
        add_177 = torch.ops.aten.add.Tensor(mul_171, arg146_1);  mul_171 = arg146_1 = None
        view_259 = torch.ops.aten.view.default(add_177, [520, 1024]);  add_177 = None
        permute_160 = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg148_1, view_259, permute_160);  arg148_1 = view_259 = permute_160 = None
        view_260 = torch.ops.aten.view.default(addmm_97, [8, 65, 3072]);  addmm_97 = None
        view_261 = torch.ops.aten.view.default(view_260, [8, 65, 3, 16, 64]);  view_260 = None
        permute_161 = torch.ops.aten.permute.default(view_261, [2, 0, 3, 1, 4]);  view_261 = None
        unbind_24 = torch.ops.aten.unbind.int(permute_161);  permute_161 = None
        getitem_268 = unbind_24[0]
        getitem_269 = unbind_24[1]
        getitem_270 = unbind_24[2];  unbind_24 = None
        _scaled_dot_product_efficient_attention_24 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_268, getitem_269, getitem_270, None, False);  getitem_268 = getitem_269 = getitem_270 = None
        getitem_271 = _scaled_dot_product_efficient_attention_24[0];  _scaled_dot_product_efficient_attention_24 = None
        permute_162 = torch.ops.aten.permute.default(getitem_271, [0, 2, 1, 3]);  getitem_271 = None
        view_262 = torch.ops.aten.view.default(permute_162, [8, 65, 1024]);  permute_162 = None
        view_263 = torch.ops.aten.view.default(view_262, [520, 1024]);  view_262 = None
        permute_163 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg150_1, view_263, permute_163);  arg150_1 = view_263 = permute_163 = None
        view_264 = torch.ops.aten.view.default(addmm_98, [8, 65, 1024]);  addmm_98 = None
        add_178 = torch.ops.aten.add.Tensor(add_175, view_264);  add_175 = view_264 = None
        var_mean_50 = torch.ops.aten.var_mean.correction(add_178, [2], correction = 0, keepdim = True)
        getitem_275 = var_mean_50[0]
        getitem_276 = var_mean_50[1];  var_mean_50 = None
        add_179 = torch.ops.aten.add.Tensor(getitem_275, 1e-06);  getitem_275 = None
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
        sub_50 = torch.ops.aten.sub.Tensor(add_178, getitem_276);  getitem_276 = None
        mul_172 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_172, arg151_1);  mul_172 = arg151_1 = None
        add_180 = torch.ops.aten.add.Tensor(mul_173, arg152_1);  mul_173 = arg152_1 = None
        view_265 = torch.ops.aten.view.default(add_180, [520, 1024]);  add_180 = None
        permute_164 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg154_1, view_265, permute_164);  arg154_1 = view_265 = permute_164 = None
        view_266 = torch.ops.aten.view.default(addmm_99, [8, 65, 4096]);  addmm_99 = None
        mul_174 = torch.ops.aten.mul.Tensor(view_266, 0.5)
        mul_175 = torch.ops.aten.mul.Tensor(view_266, 0.7071067811865476);  view_266 = None
        erf_24 = torch.ops.aten.erf.default(mul_175);  mul_175 = None
        add_181 = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_174, add_181);  mul_174 = add_181 = None
        view_267 = torch.ops.aten.view.default(mul_176, [520, 4096]);  mul_176 = None
        permute_165 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg156_1, view_267, permute_165);  arg156_1 = view_267 = permute_165 = None
        view_268 = torch.ops.aten.view.default(addmm_100, [8, 65, 1024]);  addmm_100 = None
        add_182 = torch.ops.aten.add.Tensor(add_178, view_268);  add_178 = view_268 = None
        var_mean_51 = torch.ops.aten.var_mean.correction(add_182, [2], correction = 0, keepdim = True)
        getitem_277 = var_mean_51[0]
        getitem_278 = var_mean_51[1];  var_mean_51 = None
        add_183 = torch.ops.aten.add.Tensor(getitem_277, 1e-06);  getitem_277 = None
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
        sub_51 = torch.ops.aten.sub.Tensor(add_182, getitem_278);  getitem_278 = None
        mul_177 = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None
        mul_178 = torch.ops.aten.mul.Tensor(mul_177, arg157_1);  mul_177 = arg157_1 = None
        add_184 = torch.ops.aten.add.Tensor(mul_178, arg158_1);  mul_178 = arg158_1 = None
        view_269 = torch.ops.aten.view.default(add_184, [520, 1024]);  add_184 = None
        permute_166 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg160_1, view_269, permute_166);  arg160_1 = view_269 = permute_166 = None
        view_270 = torch.ops.aten.view.default(addmm_101, [8, 65, 3072]);  addmm_101 = None
        view_271 = torch.ops.aten.view.default(view_270, [8, 65, 3, 16, 64]);  view_270 = None
        permute_167 = torch.ops.aten.permute.default(view_271, [2, 0, 3, 1, 4]);  view_271 = None
        unbind_25 = torch.ops.aten.unbind.int(permute_167);  permute_167 = None
        getitem_279 = unbind_25[0]
        getitem_280 = unbind_25[1]
        getitem_281 = unbind_25[2];  unbind_25 = None
        _scaled_dot_product_efficient_attention_25 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_279, getitem_280, getitem_281, None, False);  getitem_279 = getitem_280 = getitem_281 = None
        getitem_282 = _scaled_dot_product_efficient_attention_25[0];  _scaled_dot_product_efficient_attention_25 = None
        permute_168 = torch.ops.aten.permute.default(getitem_282, [0, 2, 1, 3]);  getitem_282 = None
        view_272 = torch.ops.aten.view.default(permute_168, [8, 65, 1024]);  permute_168 = None
        view_273 = torch.ops.aten.view.default(view_272, [520, 1024]);  view_272 = None
        permute_169 = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg162_1, view_273, permute_169);  arg162_1 = view_273 = permute_169 = None
        view_274 = torch.ops.aten.view.default(addmm_102, [8, 65, 1024]);  addmm_102 = None
        add_185 = torch.ops.aten.add.Tensor(add_182, view_274);  add_182 = view_274 = None
        var_mean_52 = torch.ops.aten.var_mean.correction(add_185, [2], correction = 0, keepdim = True)
        getitem_286 = var_mean_52[0]
        getitem_287 = var_mean_52[1];  var_mean_52 = None
        add_186 = torch.ops.aten.add.Tensor(getitem_286, 1e-06);  getitem_286 = None
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_52 = torch.ops.aten.sub.Tensor(add_185, getitem_287);  getitem_287 = None
        mul_179 = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None
        mul_180 = torch.ops.aten.mul.Tensor(mul_179, arg163_1);  mul_179 = arg163_1 = None
        add_187 = torch.ops.aten.add.Tensor(mul_180, arg164_1);  mul_180 = arg164_1 = None
        view_275 = torch.ops.aten.view.default(add_187, [520, 1024]);  add_187 = None
        permute_170 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg166_1, view_275, permute_170);  arg166_1 = view_275 = permute_170 = None
        view_276 = torch.ops.aten.view.default(addmm_103, [8, 65, 4096]);  addmm_103 = None
        mul_181 = torch.ops.aten.mul.Tensor(view_276, 0.5)
        mul_182 = torch.ops.aten.mul.Tensor(view_276, 0.7071067811865476);  view_276 = None
        erf_25 = torch.ops.aten.erf.default(mul_182);  mul_182 = None
        add_188 = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_183 = torch.ops.aten.mul.Tensor(mul_181, add_188);  mul_181 = add_188 = None
        view_277 = torch.ops.aten.view.default(mul_183, [520, 4096]);  mul_183 = None
        permute_171 = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg168_1, view_277, permute_171);  arg168_1 = view_277 = permute_171 = None
        view_278 = torch.ops.aten.view.default(addmm_104, [8, 65, 1024]);  addmm_104 = None
        add_189 = torch.ops.aten.add.Tensor(add_185, view_278);  add_185 = view_278 = None
        slice_23 = torch.ops.aten.slice.Tensor(add_189, 1, 0, 1);  add_189 = None
        clone_82 = torch.ops.aten.clone.default(slice_23, memory_format = torch.contiguous_format);  slice_23 = None
        var_mean_53 = torch.ops.aten.var_mean.correction(clone_82, [2], correction = 0, keepdim = True)
        getitem_288 = var_mean_53[0]
        getitem_289 = var_mean_53[1];  var_mean_53 = None
        add_190 = torch.ops.aten.add.Tensor(getitem_288, 1e-06);  getitem_288 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        sub_53 = torch.ops.aten.sub.Tensor(clone_82, getitem_289);  clone_82 = getitem_289 = None
        mul_184 = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = rsqrt_53 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_184, arg169_1);  mul_184 = arg169_1 = None
        add_191 = torch.ops.aten.add.Tensor(mul_185, arg170_1);  mul_185 = arg170_1 = None
        select_1 = torch.ops.aten.select.int(add_191, 1, 0);  add_191 = None
        permute_173 = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg172_1, select_1, permute_173);  arg172_1 = select_1 = permute_173 = None
        return (addmm_105,)
        
def load_args(reader):
    buf0 = reader.storage(None, 602112, device=device(type='cuda', index=0))
    reader.tensor(buf0, (256, 3, 14, 14), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1, (256,), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf2, (8, 3, 224, 224), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 984064, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1, 256, 31, 31), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1, 1, 256), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf5, (256,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf6, (256,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768, 256), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf9, (256, 256), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf10, (256,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf11, (256,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf12, (256,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf13, (1024, 256), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf14, (1024,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf15, (256, 1024), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf16, (256,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf17, (256,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf18, (256,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768, 256), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf21, (256, 256), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf22, (256,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf23, (256,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf24, (256,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf25, (1024, 256), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf26, (1024,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf27, (256, 1024), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf28, (256,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf29, (256,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf30, (256,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768, 256), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf33, (256, 256), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf35, (256,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf36, (256,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf37, (1024, 256), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf38, (1024,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf39, (256, 1024), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf40, (256,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf41, (512, 1, 3, 3), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf42, (512,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf43, (512, 256), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf44, (512,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf45, (512,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf46, (512,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf47, (1536, 512), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf48, (1536,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf49, (512, 512), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf50, (512,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf51, (512,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf52, (512,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf53, (2048, 512), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf54, (2048,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf55, (512, 2048), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf56, (512,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf57, (512,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf58, (512,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf59, (1536, 512), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf60, (1536,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf61, (512, 512), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf62, (512,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf63, (512,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf64, (512,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf65, (2048, 512), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf66, (2048,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf67, (512, 2048), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf68, (512,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf69, (512,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf70, (512,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf71, (1536, 512), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf72, (1536,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512, 512), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf74, (512,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf76, (512,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf77, (2048, 512), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf78, (2048,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf79, (512, 2048), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf80, (512,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf81, (512,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf82, (512,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf83, (1536, 512), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf84, (1536,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf85, (512, 512), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf86, (512,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf87, (512,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf88, (512,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf89, (2048, 512), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf90, (2048,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf91, (512, 2048), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf92, (512,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf93, (512,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf94, (512,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf95, (1536, 512), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf96, (1536,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf97, (512, 512), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf98, (512,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf99, (512,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf100, (512,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf101, (2048, 512), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf102, (2048,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf103, (512, 2048), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf104, (512,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf105, (512,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf106, (512,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf107, (1536, 512), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf108, (1536,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf109, (512, 512), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf110, (512,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf111, (512,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf112, (512,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf113, (2048, 512), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf114, (2048,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf115, (512, 2048), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf116, (512,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf117, (1024, 1, 3, 3), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf118, (1024,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf119, (1024, 512), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf120, (1024,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf121, (1024,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf122, (1024,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 12582912, device=device(type='cuda', index=0))
    reader.tensor(buf123, (3072, 1024), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf124, (3072,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf125, (1024, 1024), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf126, (1024,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf127, (1024,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf128, (1024,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf129, (4096, 1024), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf130, (4096,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf131, (1024, 4096), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1024,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf133, (1024,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf134, (1024,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 12582912, device=device(type='cuda', index=0))
    reader.tensor(buf135, (3072, 1024), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf136, (3072,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf137, (1024, 1024), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf138, (1024,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf139, (1024,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf140, (1024,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf141, (4096, 1024), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf142, (4096,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf143, (1024, 4096), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf144, (1024,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf145, (1024,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf146, (1024,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 12582912, device=device(type='cuda', index=0))
    reader.tensor(buf147, (3072, 1024), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf148, (3072,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf149, (1024, 1024), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf150, (1024,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf151, (1024,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf152, (1024,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf153, (4096, 1024), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf154, (4096,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf155, (1024, 4096), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf156, (1024,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf157, (1024,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf158, (1024,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 12582912, device=device(type='cuda', index=0))
    reader.tensor(buf159, (3072, 1024), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf160, (3072,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf161, (1024, 1024), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf162, (1024,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf163, (1024,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf164, (1024,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf165, (4096, 1024), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf166, (4096,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf167, (1024, 4096), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf168, (1024,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf169, (1024,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf170, (1024,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 4096000, device=device(type='cuda', index=0))
    reader.tensor(buf171, (1000, 1024), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf172, (1000,), is_leaf=True)  # arg172_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)