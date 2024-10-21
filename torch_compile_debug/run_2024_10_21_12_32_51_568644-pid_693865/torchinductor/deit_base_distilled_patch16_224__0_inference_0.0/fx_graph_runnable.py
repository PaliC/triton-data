
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1):
        convolution_1 = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        view_121 = torch.ops.aten.view.default(convolution_1, [8, 768, 196]);  convolution_1 = None
        permute_75 = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
        expand_2 = torch.ops.aten.expand.default(arg4_1, [8, -1, -1]);  arg4_1 = None
        expand_3 = torch.ops.aten.expand.default(arg5_1, [8, -1, -1]);  arg5_1 = None
        cat_1 = torch.ops.aten.cat.default([expand_2, expand_3, permute_75], 1);  expand_2 = expand_3 = permute_75 = None
        add_88 = torch.ops.aten.add.Tensor(cat_1, arg3_1);  cat_1 = arg3_1 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_134 = var_mean_25[0]
        getitem_135 = var_mean_25[1];  var_mean_25 = None
        add_89 = torch.ops.aten.add.Tensor(getitem_134, 1e-06);  getitem_134 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_88, getitem_135);  getitem_135 = None
        mul_86 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_86, arg6_1);  mul_86 = arg6_1 = None
        add_90 = torch.ops.aten.add.Tensor(mul_87, arg7_1);  mul_87 = arg7_1 = None
        view_122 = torch.ops.aten.view.default(add_90, [1584, 768]);  add_90 = None
        permute_76 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg9_1, view_122, permute_76);  arg9_1 = view_122 = permute_76 = None
        view_123 = torch.ops.aten.view.default(addmm_50, [8, 198, 2304]);  addmm_50 = None
        view_124 = torch.ops.aten.view.default(view_123, [8, 198, 3, 12, 64]);  view_123 = None
        permute_77 = torch.ops.aten.permute.default(view_124, [2, 0, 3, 1, 4]);  view_124 = None
        unbind_12 = torch.ops.aten.unbind.int(permute_77);  permute_77 = None
        getitem_136 = unbind_12[0]
        getitem_137 = unbind_12[1]
        getitem_138 = unbind_12[2];  unbind_12 = None
        _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_136, getitem_137, getitem_138, None, False);  getitem_136 = getitem_137 = getitem_138 = None
        getitem_139 = _scaled_dot_product_efficient_attention_12[0];  _scaled_dot_product_efficient_attention_12 = None
        permute_78 = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
        view_125 = torch.ops.aten.view.default(permute_78, [8, 198, 768]);  permute_78 = None
        view_126 = torch.ops.aten.view.default(view_125, [1584, 768]);  view_125 = None
        permute_79 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg11_1, view_126, permute_79);  arg11_1 = view_126 = permute_79 = None
        view_127 = torch.ops.aten.view.default(addmm_51, [8, 198, 768]);  addmm_51 = None
        add_91 = torch.ops.aten.add.Tensor(add_88, view_127);  add_88 = view_127 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_143 = var_mean_26[0]
        getitem_144 = var_mean_26[1];  var_mean_26 = None
        add_92 = torch.ops.aten.add.Tensor(getitem_143, 1e-06);  getitem_143 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_26 = torch.ops.aten.sub.Tensor(add_91, getitem_144);  getitem_144 = None
        mul_88 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, arg12_1);  mul_88 = arg12_1 = None
        add_93 = torch.ops.aten.add.Tensor(mul_89, arg13_1);  mul_89 = arg13_1 = None
        view_128 = torch.ops.aten.view.default(add_93, [1584, 768]);  add_93 = None
        permute_80 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg15_1, view_128, permute_80);  arg15_1 = view_128 = permute_80 = None
        view_129 = torch.ops.aten.view.default(addmm_52, [8, 198, 3072]);  addmm_52 = None
        mul_90 = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_91 = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_12 = torch.ops.aten.erf.default(mul_91);  mul_91 = None
        add_94 = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_92 = torch.ops.aten.mul.Tensor(mul_90, add_94);  mul_90 = add_94 = None
        view_130 = torch.ops.aten.view.default(mul_92, [1584, 3072]);  mul_92 = None
        permute_81 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg17_1, view_130, permute_81);  arg17_1 = view_130 = permute_81 = None
        view_131 = torch.ops.aten.view.default(addmm_53, [8, 198, 768]);  addmm_53 = None
        add_95 = torch.ops.aten.add.Tensor(add_91, view_131);  add_91 = view_131 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_145 = var_mean_27[0]
        getitem_146 = var_mean_27[1];  var_mean_27 = None
        add_96 = torch.ops.aten.add.Tensor(getitem_145, 1e-06);  getitem_145 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_95, getitem_146);  getitem_146 = None
        mul_93 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
        mul_94 = torch.ops.aten.mul.Tensor(mul_93, arg18_1);  mul_93 = arg18_1 = None
        add_97 = torch.ops.aten.add.Tensor(mul_94, arg19_1);  mul_94 = arg19_1 = None
        view_132 = torch.ops.aten.view.default(add_97, [1584, 768]);  add_97 = None
        permute_82 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg21_1, view_132, permute_82);  arg21_1 = view_132 = permute_82 = None
        view_133 = torch.ops.aten.view.default(addmm_54, [8, 198, 2304]);  addmm_54 = None
        view_134 = torch.ops.aten.view.default(view_133, [8, 198, 3, 12, 64]);  view_133 = None
        permute_83 = torch.ops.aten.permute.default(view_134, [2, 0, 3, 1, 4]);  view_134 = None
        unbind_13 = torch.ops.aten.unbind.int(permute_83);  permute_83 = None
        getitem_147 = unbind_13[0]
        getitem_148 = unbind_13[1]
        getitem_149 = unbind_13[2];  unbind_13 = None
        _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_147, getitem_148, getitem_149, None, False);  getitem_147 = getitem_148 = getitem_149 = None
        getitem_150 = _scaled_dot_product_efficient_attention_13[0];  _scaled_dot_product_efficient_attention_13 = None
        permute_84 = torch.ops.aten.permute.default(getitem_150, [0, 2, 1, 3]);  getitem_150 = None
        view_135 = torch.ops.aten.view.default(permute_84, [8, 198, 768]);  permute_84 = None
        view_136 = torch.ops.aten.view.default(view_135, [1584, 768]);  view_135 = None
        permute_85 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg23_1, view_136, permute_85);  arg23_1 = view_136 = permute_85 = None
        view_137 = torch.ops.aten.view.default(addmm_55, [8, 198, 768]);  addmm_55 = None
        add_98 = torch.ops.aten.add.Tensor(add_95, view_137);  add_95 = view_137 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
        getitem_154 = var_mean_28[0]
        getitem_155 = var_mean_28[1];  var_mean_28 = None
        add_99 = torch.ops.aten.add.Tensor(getitem_154, 1e-06);  getitem_154 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_28 = torch.ops.aten.sub.Tensor(add_98, getitem_155);  getitem_155 = None
        mul_95 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
        mul_96 = torch.ops.aten.mul.Tensor(mul_95, arg24_1);  mul_95 = arg24_1 = None
        add_100 = torch.ops.aten.add.Tensor(mul_96, arg25_1);  mul_96 = arg25_1 = None
        view_138 = torch.ops.aten.view.default(add_100, [1584, 768]);  add_100 = None
        permute_86 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg27_1, view_138, permute_86);  arg27_1 = view_138 = permute_86 = None
        view_139 = torch.ops.aten.view.default(addmm_56, [8, 198, 3072]);  addmm_56 = None
        mul_97 = torch.ops.aten.mul.Tensor(view_139, 0.5)
        mul_98 = torch.ops.aten.mul.Tensor(view_139, 0.7071067811865476);  view_139 = None
        erf_13 = torch.ops.aten.erf.default(mul_98);  mul_98 = None
        add_101 = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_99 = torch.ops.aten.mul.Tensor(mul_97, add_101);  mul_97 = add_101 = None
        view_140 = torch.ops.aten.view.default(mul_99, [1584, 3072]);  mul_99 = None
        permute_87 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg29_1, view_140, permute_87);  arg29_1 = view_140 = permute_87 = None
        view_141 = torch.ops.aten.view.default(addmm_57, [8, 198, 768]);  addmm_57 = None
        add_102 = torch.ops.aten.add.Tensor(add_98, view_141);  add_98 = view_141 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_102, [2], correction = 0, keepdim = True)
        getitem_156 = var_mean_29[0]
        getitem_157 = var_mean_29[1];  var_mean_29 = None
        add_103 = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_102, getitem_157);  getitem_157 = None
        mul_100 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, arg30_1);  mul_100 = arg30_1 = None
        add_104 = torch.ops.aten.add.Tensor(mul_101, arg31_1);  mul_101 = arg31_1 = None
        view_142 = torch.ops.aten.view.default(add_104, [1584, 768]);  add_104 = None
        permute_88 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg33_1, view_142, permute_88);  arg33_1 = view_142 = permute_88 = None
        view_143 = torch.ops.aten.view.default(addmm_58, [8, 198, 2304]);  addmm_58 = None
        view_144 = torch.ops.aten.view.default(view_143, [8, 198, 3, 12, 64]);  view_143 = None
        permute_89 = torch.ops.aten.permute.default(view_144, [2, 0, 3, 1, 4]);  view_144 = None
        unbind_14 = torch.ops.aten.unbind.int(permute_89);  permute_89 = None
        getitem_158 = unbind_14[0]
        getitem_159 = unbind_14[1]
        getitem_160 = unbind_14[2];  unbind_14 = None
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_158, getitem_159, getitem_160, None, False);  getitem_158 = getitem_159 = getitem_160 = None
        getitem_161 = _scaled_dot_product_efficient_attention_14[0];  _scaled_dot_product_efficient_attention_14 = None
        permute_90 = torch.ops.aten.permute.default(getitem_161, [0, 2, 1, 3]);  getitem_161 = None
        view_145 = torch.ops.aten.view.default(permute_90, [8, 198, 768]);  permute_90 = None
        view_146 = torch.ops.aten.view.default(view_145, [1584, 768]);  view_145 = None
        permute_91 = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg35_1, view_146, permute_91);  arg35_1 = view_146 = permute_91 = None
        view_147 = torch.ops.aten.view.default(addmm_59, [8, 198, 768]);  addmm_59 = None
        add_105 = torch.ops.aten.add.Tensor(add_102, view_147);  add_102 = view_147 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_165 = var_mean_30[0]
        getitem_166 = var_mean_30[1];  var_mean_30 = None
        add_106 = torch.ops.aten.add.Tensor(getitem_165, 1e-06);  getitem_165 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_105, getitem_166);  getitem_166 = None
        mul_102 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
        mul_103 = torch.ops.aten.mul.Tensor(mul_102, arg36_1);  mul_102 = arg36_1 = None
        add_107 = torch.ops.aten.add.Tensor(mul_103, arg37_1);  mul_103 = arg37_1 = None
        view_148 = torch.ops.aten.view.default(add_107, [1584, 768]);  add_107 = None
        permute_92 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg39_1, view_148, permute_92);  arg39_1 = view_148 = permute_92 = None
        view_149 = torch.ops.aten.view.default(addmm_60, [8, 198, 3072]);  addmm_60 = None
        mul_104 = torch.ops.aten.mul.Tensor(view_149, 0.5)
        mul_105 = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476);  view_149 = None
        erf_14 = torch.ops.aten.erf.default(mul_105);  mul_105 = None
        add_108 = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_106 = torch.ops.aten.mul.Tensor(mul_104, add_108);  mul_104 = add_108 = None
        view_150 = torch.ops.aten.view.default(mul_106, [1584, 3072]);  mul_106 = None
        permute_93 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg41_1, view_150, permute_93);  arg41_1 = view_150 = permute_93 = None
        view_151 = torch.ops.aten.view.default(addmm_61, [8, 198, 768]);  addmm_61 = None
        add_109 = torch.ops.aten.add.Tensor(add_105, view_151);  add_105 = view_151 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
        getitem_167 = var_mean_31[0]
        getitem_168 = var_mean_31[1];  var_mean_31 = None
        add_110 = torch.ops.aten.add.Tensor(getitem_167, 1e-06);  getitem_167 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        sub_31 = torch.ops.aten.sub.Tensor(add_109, getitem_168);  getitem_168 = None
        mul_107 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
        mul_108 = torch.ops.aten.mul.Tensor(mul_107, arg42_1);  mul_107 = arg42_1 = None
        add_111 = torch.ops.aten.add.Tensor(mul_108, arg43_1);  mul_108 = arg43_1 = None
        view_152 = torch.ops.aten.view.default(add_111, [1584, 768]);  add_111 = None
        permute_94 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg45_1, view_152, permute_94);  arg45_1 = view_152 = permute_94 = None
        view_153 = torch.ops.aten.view.default(addmm_62, [8, 198, 2304]);  addmm_62 = None
        view_154 = torch.ops.aten.view.default(view_153, [8, 198, 3, 12, 64]);  view_153 = None
        permute_95 = torch.ops.aten.permute.default(view_154, [2, 0, 3, 1, 4]);  view_154 = None
        unbind_15 = torch.ops.aten.unbind.int(permute_95);  permute_95 = None
        getitem_169 = unbind_15[0]
        getitem_170 = unbind_15[1]
        getitem_171 = unbind_15[2];  unbind_15 = None
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_169, getitem_170, getitem_171, None, False);  getitem_169 = getitem_170 = getitem_171 = None
        getitem_172 = _scaled_dot_product_efficient_attention_15[0];  _scaled_dot_product_efficient_attention_15 = None
        permute_96 = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
        view_155 = torch.ops.aten.view.default(permute_96, [8, 198, 768]);  permute_96 = None
        view_156 = torch.ops.aten.view.default(view_155, [1584, 768]);  view_155 = None
        permute_97 = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg47_1, view_156, permute_97);  arg47_1 = view_156 = permute_97 = None
        view_157 = torch.ops.aten.view.default(addmm_63, [8, 198, 768]);  addmm_63 = None
        add_112 = torch.ops.aten.add.Tensor(add_109, view_157);  add_109 = view_157 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
        getitem_176 = var_mean_32[0]
        getitem_177 = var_mean_32[1];  var_mean_32 = None
        add_113 = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        sub_32 = torch.ops.aten.sub.Tensor(add_112, getitem_177);  getitem_177 = None
        mul_109 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
        mul_110 = torch.ops.aten.mul.Tensor(mul_109, arg48_1);  mul_109 = arg48_1 = None
        add_114 = torch.ops.aten.add.Tensor(mul_110, arg49_1);  mul_110 = arg49_1 = None
        view_158 = torch.ops.aten.view.default(add_114, [1584, 768]);  add_114 = None
        permute_98 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg51_1, view_158, permute_98);  arg51_1 = view_158 = permute_98 = None
        view_159 = torch.ops.aten.view.default(addmm_64, [8, 198, 3072]);  addmm_64 = None
        mul_111 = torch.ops.aten.mul.Tensor(view_159, 0.5)
        mul_112 = torch.ops.aten.mul.Tensor(view_159, 0.7071067811865476);  view_159 = None
        erf_15 = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_115 = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_111, add_115);  mul_111 = add_115 = None
        view_160 = torch.ops.aten.view.default(mul_113, [1584, 3072]);  mul_113 = None
        permute_99 = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg53_1, view_160, permute_99);  arg53_1 = view_160 = permute_99 = None
        view_161 = torch.ops.aten.view.default(addmm_65, [8, 198, 768]);  addmm_65 = None
        add_116 = torch.ops.aten.add.Tensor(add_112, view_161);  add_112 = view_161 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_116, [2], correction = 0, keepdim = True)
        getitem_178 = var_mean_33[0]
        getitem_179 = var_mean_33[1];  var_mean_33 = None
        add_117 = torch.ops.aten.add.Tensor(getitem_178, 1e-06);  getitem_178 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_116, getitem_179);  getitem_179 = None
        mul_114 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
        mul_115 = torch.ops.aten.mul.Tensor(mul_114, arg54_1);  mul_114 = arg54_1 = None
        add_118 = torch.ops.aten.add.Tensor(mul_115, arg55_1);  mul_115 = arg55_1 = None
        view_162 = torch.ops.aten.view.default(add_118, [1584, 768]);  add_118 = None
        permute_100 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg57_1, view_162, permute_100);  arg57_1 = view_162 = permute_100 = None
        view_163 = torch.ops.aten.view.default(addmm_66, [8, 198, 2304]);  addmm_66 = None
        view_164 = torch.ops.aten.view.default(view_163, [8, 198, 3, 12, 64]);  view_163 = None
        permute_101 = torch.ops.aten.permute.default(view_164, [2, 0, 3, 1, 4]);  view_164 = None
        unbind_16 = torch.ops.aten.unbind.int(permute_101);  permute_101 = None
        getitem_180 = unbind_16[0]
        getitem_181 = unbind_16[1]
        getitem_182 = unbind_16[2];  unbind_16 = None
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_180, getitem_181, getitem_182, None, False);  getitem_180 = getitem_181 = getitem_182 = None
        getitem_183 = _scaled_dot_product_efficient_attention_16[0];  _scaled_dot_product_efficient_attention_16 = None
        permute_102 = torch.ops.aten.permute.default(getitem_183, [0, 2, 1, 3]);  getitem_183 = None
        view_165 = torch.ops.aten.view.default(permute_102, [8, 198, 768]);  permute_102 = None
        view_166 = torch.ops.aten.view.default(view_165, [1584, 768]);  view_165 = None
        permute_103 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg59_1, view_166, permute_103);  arg59_1 = view_166 = permute_103 = None
        view_167 = torch.ops.aten.view.default(addmm_67, [8, 198, 768]);  addmm_67 = None
        add_119 = torch.ops.aten.add.Tensor(add_116, view_167);  add_116 = view_167 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
        getitem_187 = var_mean_34[0]
        getitem_188 = var_mean_34[1];  var_mean_34 = None
        add_120 = torch.ops.aten.add.Tensor(getitem_187, 1e-06);  getitem_187 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        sub_34 = torch.ops.aten.sub.Tensor(add_119, getitem_188);  getitem_188 = None
        mul_116 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_116, arg60_1);  mul_116 = arg60_1 = None
        add_121 = torch.ops.aten.add.Tensor(mul_117, arg61_1);  mul_117 = arg61_1 = None
        view_168 = torch.ops.aten.view.default(add_121, [1584, 768]);  add_121 = None
        permute_104 = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg63_1, view_168, permute_104);  arg63_1 = view_168 = permute_104 = None
        view_169 = torch.ops.aten.view.default(addmm_68, [8, 198, 3072]);  addmm_68 = None
        mul_118 = torch.ops.aten.mul.Tensor(view_169, 0.5)
        mul_119 = torch.ops.aten.mul.Tensor(view_169, 0.7071067811865476);  view_169 = None
        erf_16 = torch.ops.aten.erf.default(mul_119);  mul_119 = None
        add_122 = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_120 = torch.ops.aten.mul.Tensor(mul_118, add_122);  mul_118 = add_122 = None
        view_170 = torch.ops.aten.view.default(mul_120, [1584, 3072]);  mul_120 = None
        permute_105 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg65_1, view_170, permute_105);  arg65_1 = view_170 = permute_105 = None
        view_171 = torch.ops.aten.view.default(addmm_69, [8, 198, 768]);  addmm_69 = None
        add_123 = torch.ops.aten.add.Tensor(add_119, view_171);  add_119 = view_171 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_123, [2], correction = 0, keepdim = True)
        getitem_189 = var_mean_35[0]
        getitem_190 = var_mean_35[1];  var_mean_35 = None
        add_124 = torch.ops.aten.add.Tensor(getitem_189, 1e-06);  getitem_189 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        sub_35 = torch.ops.aten.sub.Tensor(add_123, getitem_190);  getitem_190 = None
        mul_121 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
        mul_122 = torch.ops.aten.mul.Tensor(mul_121, arg66_1);  mul_121 = arg66_1 = None
        add_125 = torch.ops.aten.add.Tensor(mul_122, arg67_1);  mul_122 = arg67_1 = None
        view_172 = torch.ops.aten.view.default(add_125, [1584, 768]);  add_125 = None
        permute_106 = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg69_1, view_172, permute_106);  arg69_1 = view_172 = permute_106 = None
        view_173 = torch.ops.aten.view.default(addmm_70, [8, 198, 2304]);  addmm_70 = None
        view_174 = torch.ops.aten.view.default(view_173, [8, 198, 3, 12, 64]);  view_173 = None
        permute_107 = torch.ops.aten.permute.default(view_174, [2, 0, 3, 1, 4]);  view_174 = None
        unbind_17 = torch.ops.aten.unbind.int(permute_107);  permute_107 = None
        getitem_191 = unbind_17[0]
        getitem_192 = unbind_17[1]
        getitem_193 = unbind_17[2];  unbind_17 = None
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_191, getitem_192, getitem_193, None, False);  getitem_191 = getitem_192 = getitem_193 = None
        getitem_194 = _scaled_dot_product_efficient_attention_17[0];  _scaled_dot_product_efficient_attention_17 = None
        permute_108 = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3]);  getitem_194 = None
        view_175 = torch.ops.aten.view.default(permute_108, [8, 198, 768]);  permute_108 = None
        view_176 = torch.ops.aten.view.default(view_175, [1584, 768]);  view_175 = None
        permute_109 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg71_1, view_176, permute_109);  arg71_1 = view_176 = permute_109 = None
        view_177 = torch.ops.aten.view.default(addmm_71, [8, 198, 768]);  addmm_71 = None
        add_126 = torch.ops.aten.add.Tensor(add_123, view_177);  add_123 = view_177 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_126, [2], correction = 0, keepdim = True)
        getitem_198 = var_mean_36[0]
        getitem_199 = var_mean_36[1];  var_mean_36 = None
        add_127 = torch.ops.aten.add.Tensor(getitem_198, 1e-06);  getitem_198 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_126, getitem_199);  getitem_199 = None
        mul_123 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
        mul_124 = torch.ops.aten.mul.Tensor(mul_123, arg72_1);  mul_123 = arg72_1 = None
        add_128 = torch.ops.aten.add.Tensor(mul_124, arg73_1);  mul_124 = arg73_1 = None
        view_178 = torch.ops.aten.view.default(add_128, [1584, 768]);  add_128 = None
        permute_110 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg75_1, view_178, permute_110);  arg75_1 = view_178 = permute_110 = None
        view_179 = torch.ops.aten.view.default(addmm_72, [8, 198, 3072]);  addmm_72 = None
        mul_125 = torch.ops.aten.mul.Tensor(view_179, 0.5)
        mul_126 = torch.ops.aten.mul.Tensor(view_179, 0.7071067811865476);  view_179 = None
        erf_17 = torch.ops.aten.erf.default(mul_126);  mul_126 = None
        add_129 = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_127 = torch.ops.aten.mul.Tensor(mul_125, add_129);  mul_125 = add_129 = None
        view_180 = torch.ops.aten.view.default(mul_127, [1584, 3072]);  mul_127 = None
        permute_111 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg77_1, view_180, permute_111);  arg77_1 = view_180 = permute_111 = None
        view_181 = torch.ops.aten.view.default(addmm_73, [8, 198, 768]);  addmm_73 = None
        add_130 = torch.ops.aten.add.Tensor(add_126, view_181);  add_126 = view_181 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_130, [2], correction = 0, keepdim = True)
        getitem_200 = var_mean_37[0]
        getitem_201 = var_mean_37[1];  var_mean_37 = None
        add_131 = torch.ops.aten.add.Tensor(getitem_200, 1e-06);  getitem_200 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        sub_37 = torch.ops.aten.sub.Tensor(add_130, getitem_201);  getitem_201 = None
        mul_128 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_128, arg78_1);  mul_128 = arg78_1 = None
        add_132 = torch.ops.aten.add.Tensor(mul_129, arg79_1);  mul_129 = arg79_1 = None
        view_182 = torch.ops.aten.view.default(add_132, [1584, 768]);  add_132 = None
        permute_112 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg81_1, view_182, permute_112);  arg81_1 = view_182 = permute_112 = None
        view_183 = torch.ops.aten.view.default(addmm_74, [8, 198, 2304]);  addmm_74 = None
        view_184 = torch.ops.aten.view.default(view_183, [8, 198, 3, 12, 64]);  view_183 = None
        permute_113 = torch.ops.aten.permute.default(view_184, [2, 0, 3, 1, 4]);  view_184 = None
        unbind_18 = torch.ops.aten.unbind.int(permute_113);  permute_113 = None
        getitem_202 = unbind_18[0]
        getitem_203 = unbind_18[1]
        getitem_204 = unbind_18[2];  unbind_18 = None
        _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_202, getitem_203, getitem_204, None, False);  getitem_202 = getitem_203 = getitem_204 = None
        getitem_205 = _scaled_dot_product_efficient_attention_18[0];  _scaled_dot_product_efficient_attention_18 = None
        permute_114 = torch.ops.aten.permute.default(getitem_205, [0, 2, 1, 3]);  getitem_205 = None
        view_185 = torch.ops.aten.view.default(permute_114, [8, 198, 768]);  permute_114 = None
        view_186 = torch.ops.aten.view.default(view_185, [1584, 768]);  view_185 = None
        permute_115 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg83_1, view_186, permute_115);  arg83_1 = view_186 = permute_115 = None
        view_187 = torch.ops.aten.view.default(addmm_75, [8, 198, 768]);  addmm_75 = None
        add_133 = torch.ops.aten.add.Tensor(add_130, view_187);  add_130 = view_187 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_209 = var_mean_38[0]
        getitem_210 = var_mean_38[1];  var_mean_38 = None
        add_134 = torch.ops.aten.add.Tensor(getitem_209, 1e-06);  getitem_209 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_38 = torch.ops.aten.sub.Tensor(add_133, getitem_210);  getitem_210 = None
        mul_130 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
        mul_131 = torch.ops.aten.mul.Tensor(mul_130, arg84_1);  mul_130 = arg84_1 = None
        add_135 = torch.ops.aten.add.Tensor(mul_131, arg85_1);  mul_131 = arg85_1 = None
        view_188 = torch.ops.aten.view.default(add_135, [1584, 768]);  add_135 = None
        permute_116 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg87_1, view_188, permute_116);  arg87_1 = view_188 = permute_116 = None
        view_189 = torch.ops.aten.view.default(addmm_76, [8, 198, 3072]);  addmm_76 = None
        mul_132 = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_133 = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_18 = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_136 = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_132, add_136);  mul_132 = add_136 = None
        view_190 = torch.ops.aten.view.default(mul_134, [1584, 3072]);  mul_134 = None
        permute_117 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg89_1, view_190, permute_117);  arg89_1 = view_190 = permute_117 = None
        view_191 = torch.ops.aten.view.default(addmm_77, [8, 198, 768]);  addmm_77 = None
        add_137 = torch.ops.aten.add.Tensor(add_133, view_191);  add_133 = view_191 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_211 = var_mean_39[0]
        getitem_212 = var_mean_39[1];  var_mean_39 = None
        add_138 = torch.ops.aten.add.Tensor(getitem_211, 1e-06);  getitem_211 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_137, getitem_212);  getitem_212 = None
        mul_135 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
        mul_136 = torch.ops.aten.mul.Tensor(mul_135, arg90_1);  mul_135 = arg90_1 = None
        add_139 = torch.ops.aten.add.Tensor(mul_136, arg91_1);  mul_136 = arg91_1 = None
        view_192 = torch.ops.aten.view.default(add_139, [1584, 768]);  add_139 = None
        permute_118 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg93_1, view_192, permute_118);  arg93_1 = view_192 = permute_118 = None
        view_193 = torch.ops.aten.view.default(addmm_78, [8, 198, 2304]);  addmm_78 = None
        view_194 = torch.ops.aten.view.default(view_193, [8, 198, 3, 12, 64]);  view_193 = None
        permute_119 = torch.ops.aten.permute.default(view_194, [2, 0, 3, 1, 4]);  view_194 = None
        unbind_19 = torch.ops.aten.unbind.int(permute_119);  permute_119 = None
        getitem_213 = unbind_19[0]
        getitem_214 = unbind_19[1]
        getitem_215 = unbind_19[2];  unbind_19 = None
        _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_213, getitem_214, getitem_215, None, False);  getitem_213 = getitem_214 = getitem_215 = None
        getitem_216 = _scaled_dot_product_efficient_attention_19[0];  _scaled_dot_product_efficient_attention_19 = None
        permute_120 = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None
        view_195 = torch.ops.aten.view.default(permute_120, [8, 198, 768]);  permute_120 = None
        view_196 = torch.ops.aten.view.default(view_195, [1584, 768]);  view_195 = None
        permute_121 = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg95_1, view_196, permute_121);  arg95_1 = view_196 = permute_121 = None
        view_197 = torch.ops.aten.view.default(addmm_79, [8, 198, 768]);  addmm_79 = None
        add_140 = torch.ops.aten.add.Tensor(add_137, view_197);  add_137 = view_197 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
        getitem_220 = var_mean_40[0]
        getitem_221 = var_mean_40[1];  var_mean_40 = None
        add_141 = torch.ops.aten.add.Tensor(getitem_220, 1e-06);  getitem_220 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_40 = torch.ops.aten.sub.Tensor(add_140, getitem_221);  getitem_221 = None
        mul_137 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_137, arg96_1);  mul_137 = arg96_1 = None
        add_142 = torch.ops.aten.add.Tensor(mul_138, arg97_1);  mul_138 = arg97_1 = None
        view_198 = torch.ops.aten.view.default(add_142, [1584, 768]);  add_142 = None
        permute_122 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg99_1, view_198, permute_122);  arg99_1 = view_198 = permute_122 = None
        view_199 = torch.ops.aten.view.default(addmm_80, [8, 198, 3072]);  addmm_80 = None
        mul_139 = torch.ops.aten.mul.Tensor(view_199, 0.5)
        mul_140 = torch.ops.aten.mul.Tensor(view_199, 0.7071067811865476);  view_199 = None
        erf_19 = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_143 = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_139, add_143);  mul_139 = add_143 = None
        view_200 = torch.ops.aten.view.default(mul_141, [1584, 3072]);  mul_141 = None
        permute_123 = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg101_1, view_200, permute_123);  arg101_1 = view_200 = permute_123 = None
        view_201 = torch.ops.aten.view.default(addmm_81, [8, 198, 768]);  addmm_81 = None
        add_144 = torch.ops.aten.add.Tensor(add_140, view_201);  add_140 = view_201 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_144, [2], correction = 0, keepdim = True)
        getitem_222 = var_mean_41[0]
        getitem_223 = var_mean_41[1];  var_mean_41 = None
        add_145 = torch.ops.aten.add.Tensor(getitem_222, 1e-06);  getitem_222 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        sub_41 = torch.ops.aten.sub.Tensor(add_144, getitem_223);  getitem_223 = None
        mul_142 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
        mul_143 = torch.ops.aten.mul.Tensor(mul_142, arg102_1);  mul_142 = arg102_1 = None
        add_146 = torch.ops.aten.add.Tensor(mul_143, arg103_1);  mul_143 = arg103_1 = None
        view_202 = torch.ops.aten.view.default(add_146, [1584, 768]);  add_146 = None
        permute_124 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg105_1, view_202, permute_124);  arg105_1 = view_202 = permute_124 = None
        view_203 = torch.ops.aten.view.default(addmm_82, [8, 198, 2304]);  addmm_82 = None
        view_204 = torch.ops.aten.view.default(view_203, [8, 198, 3, 12, 64]);  view_203 = None
        permute_125 = torch.ops.aten.permute.default(view_204, [2, 0, 3, 1, 4]);  view_204 = None
        unbind_20 = torch.ops.aten.unbind.int(permute_125);  permute_125 = None
        getitem_224 = unbind_20[0]
        getitem_225 = unbind_20[1]
        getitem_226 = unbind_20[2];  unbind_20 = None
        _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_224, getitem_225, getitem_226, None, False);  getitem_224 = getitem_225 = getitem_226 = None
        getitem_227 = _scaled_dot_product_efficient_attention_20[0];  _scaled_dot_product_efficient_attention_20 = None
        permute_126 = torch.ops.aten.permute.default(getitem_227, [0, 2, 1, 3]);  getitem_227 = None
        view_205 = torch.ops.aten.view.default(permute_126, [8, 198, 768]);  permute_126 = None
        view_206 = torch.ops.aten.view.default(view_205, [1584, 768]);  view_205 = None
        permute_127 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg107_1, view_206, permute_127);  arg107_1 = view_206 = permute_127 = None
        view_207 = torch.ops.aten.view.default(addmm_83, [8, 198, 768]);  addmm_83 = None
        add_147 = torch.ops.aten.add.Tensor(add_144, view_207);  add_144 = view_207 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_231 = var_mean_42[0]
        getitem_232 = var_mean_42[1];  var_mean_42 = None
        add_148 = torch.ops.aten.add.Tensor(getitem_231, 1e-06);  getitem_231 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_42 = torch.ops.aten.sub.Tensor(add_147, getitem_232);  getitem_232 = None
        mul_144 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_144, arg108_1);  mul_144 = arg108_1 = None
        add_149 = torch.ops.aten.add.Tensor(mul_145, arg109_1);  mul_145 = arg109_1 = None
        view_208 = torch.ops.aten.view.default(add_149, [1584, 768]);  add_149 = None
        permute_128 = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg111_1, view_208, permute_128);  arg111_1 = view_208 = permute_128 = None
        view_209 = torch.ops.aten.view.default(addmm_84, [8, 198, 3072]);  addmm_84 = None
        mul_146 = torch.ops.aten.mul.Tensor(view_209, 0.5)
        mul_147 = torch.ops.aten.mul.Tensor(view_209, 0.7071067811865476);  view_209 = None
        erf_20 = torch.ops.aten.erf.default(mul_147);  mul_147 = None
        add_150 = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_148 = torch.ops.aten.mul.Tensor(mul_146, add_150);  mul_146 = add_150 = None
        view_210 = torch.ops.aten.view.default(mul_148, [1584, 3072]);  mul_148 = None
        permute_129 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg113_1, view_210, permute_129);  arg113_1 = view_210 = permute_129 = None
        view_211 = torch.ops.aten.view.default(addmm_85, [8, 198, 768]);  addmm_85 = None
        add_151 = torch.ops.aten.add.Tensor(add_147, view_211);  add_147 = view_211 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(add_151, [2], correction = 0, keepdim = True)
        getitem_233 = var_mean_43[0]
        getitem_234 = var_mean_43[1];  var_mean_43 = None
        add_152 = torch.ops.aten.add.Tensor(getitem_233, 1e-06);  getitem_233 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        sub_43 = torch.ops.aten.sub.Tensor(add_151, getitem_234);  getitem_234 = None
        mul_149 = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
        mul_150 = torch.ops.aten.mul.Tensor(mul_149, arg114_1);  mul_149 = arg114_1 = None
        add_153 = torch.ops.aten.add.Tensor(mul_150, arg115_1);  mul_150 = arg115_1 = None
        view_212 = torch.ops.aten.view.default(add_153, [1584, 768]);  add_153 = None
        permute_130 = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg117_1, view_212, permute_130);  arg117_1 = view_212 = permute_130 = None
        view_213 = torch.ops.aten.view.default(addmm_86, [8, 198, 2304]);  addmm_86 = None
        view_214 = torch.ops.aten.view.default(view_213, [8, 198, 3, 12, 64]);  view_213 = None
        permute_131 = torch.ops.aten.permute.default(view_214, [2, 0, 3, 1, 4]);  view_214 = None
        unbind_21 = torch.ops.aten.unbind.int(permute_131);  permute_131 = None
        getitem_235 = unbind_21[0]
        getitem_236 = unbind_21[1]
        getitem_237 = unbind_21[2];  unbind_21 = None
        _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_235, getitem_236, getitem_237, None, False);  getitem_235 = getitem_236 = getitem_237 = None
        getitem_238 = _scaled_dot_product_efficient_attention_21[0];  _scaled_dot_product_efficient_attention_21 = None
        permute_132 = torch.ops.aten.permute.default(getitem_238, [0, 2, 1, 3]);  getitem_238 = None
        view_215 = torch.ops.aten.view.default(permute_132, [8, 198, 768]);  permute_132 = None
        view_216 = torch.ops.aten.view.default(view_215, [1584, 768]);  view_215 = None
        permute_133 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg119_1, view_216, permute_133);  arg119_1 = view_216 = permute_133 = None
        view_217 = torch.ops.aten.view.default(addmm_87, [8, 198, 768]);  addmm_87 = None
        add_154 = torch.ops.aten.add.Tensor(add_151, view_217);  add_151 = view_217 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
        getitem_242 = var_mean_44[0]
        getitem_243 = var_mean_44[1];  var_mean_44 = None
        add_155 = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        sub_44 = torch.ops.aten.sub.Tensor(add_154, getitem_243);  getitem_243 = None
        mul_151 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
        mul_152 = torch.ops.aten.mul.Tensor(mul_151, arg120_1);  mul_151 = arg120_1 = None
        add_156 = torch.ops.aten.add.Tensor(mul_152, arg121_1);  mul_152 = arg121_1 = None
        view_218 = torch.ops.aten.view.default(add_156, [1584, 768]);  add_156 = None
        permute_134 = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg123_1, view_218, permute_134);  arg123_1 = view_218 = permute_134 = None
        view_219 = torch.ops.aten.view.default(addmm_88, [8, 198, 3072]);  addmm_88 = None
        mul_153 = torch.ops.aten.mul.Tensor(view_219, 0.5)
        mul_154 = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
        erf_21 = torch.ops.aten.erf.default(mul_154);  mul_154 = None
        add_157 = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_155 = torch.ops.aten.mul.Tensor(mul_153, add_157);  mul_153 = add_157 = None
        view_220 = torch.ops.aten.view.default(mul_155, [1584, 3072]);  mul_155 = None
        permute_135 = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg125_1, view_220, permute_135);  arg125_1 = view_220 = permute_135 = None
        view_221 = torch.ops.aten.view.default(addmm_89, [8, 198, 768]);  addmm_89 = None
        add_158 = torch.ops.aten.add.Tensor(add_154, view_221);  add_154 = view_221 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
        getitem_244 = var_mean_45[0]
        getitem_245 = var_mean_45[1];  var_mean_45 = None
        add_159 = torch.ops.aten.add.Tensor(getitem_244, 1e-06);  getitem_244 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_45 = torch.ops.aten.sub.Tensor(add_158, getitem_245);  getitem_245 = None
        mul_156 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_156, arg126_1);  mul_156 = arg126_1 = None
        add_160 = torch.ops.aten.add.Tensor(mul_157, arg127_1);  mul_157 = arg127_1 = None
        view_222 = torch.ops.aten.view.default(add_160, [1584, 768]);  add_160 = None
        permute_136 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg129_1, view_222, permute_136);  arg129_1 = view_222 = permute_136 = None
        view_223 = torch.ops.aten.view.default(addmm_90, [8, 198, 2304]);  addmm_90 = None
        view_224 = torch.ops.aten.view.default(view_223, [8, 198, 3, 12, 64]);  view_223 = None
        permute_137 = torch.ops.aten.permute.default(view_224, [2, 0, 3, 1, 4]);  view_224 = None
        unbind_22 = torch.ops.aten.unbind.int(permute_137);  permute_137 = None
        getitem_246 = unbind_22[0]
        getitem_247 = unbind_22[1]
        getitem_248 = unbind_22[2];  unbind_22 = None
        _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_246, getitem_247, getitem_248, None, False);  getitem_246 = getitem_247 = getitem_248 = None
        getitem_249 = _scaled_dot_product_efficient_attention_22[0];  _scaled_dot_product_efficient_attention_22 = None
        permute_138 = torch.ops.aten.permute.default(getitem_249, [0, 2, 1, 3]);  getitem_249 = None
        view_225 = torch.ops.aten.view.default(permute_138, [8, 198, 768]);  permute_138 = None
        view_226 = torch.ops.aten.view.default(view_225, [1584, 768]);  view_225 = None
        permute_139 = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg131_1, view_226, permute_139);  arg131_1 = view_226 = permute_139 = None
        view_227 = torch.ops.aten.view.default(addmm_91, [8, 198, 768]);  addmm_91 = None
        add_161 = torch.ops.aten.add.Tensor(add_158, view_227);  add_158 = view_227 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_253 = var_mean_46[0]
        getitem_254 = var_mean_46[1];  var_mean_46 = None
        add_162 = torch.ops.aten.add.Tensor(getitem_253, 1e-06);  getitem_253 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_46 = torch.ops.aten.sub.Tensor(add_161, getitem_254);  getitem_254 = None
        mul_158 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
        mul_159 = torch.ops.aten.mul.Tensor(mul_158, arg132_1);  mul_158 = arg132_1 = None
        add_163 = torch.ops.aten.add.Tensor(mul_159, arg133_1);  mul_159 = arg133_1 = None
        view_228 = torch.ops.aten.view.default(add_163, [1584, 768]);  add_163 = None
        permute_140 = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg135_1, view_228, permute_140);  arg135_1 = view_228 = permute_140 = None
        view_229 = torch.ops.aten.view.default(addmm_92, [8, 198, 3072]);  addmm_92 = None
        mul_160 = torch.ops.aten.mul.Tensor(view_229, 0.5)
        mul_161 = torch.ops.aten.mul.Tensor(view_229, 0.7071067811865476);  view_229 = None
        erf_22 = torch.ops.aten.erf.default(mul_161);  mul_161 = None
        add_164 = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_162 = torch.ops.aten.mul.Tensor(mul_160, add_164);  mul_160 = add_164 = None
        view_230 = torch.ops.aten.view.default(mul_162, [1584, 3072]);  mul_162 = None
        permute_141 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg137_1, view_230, permute_141);  arg137_1 = view_230 = permute_141 = None
        view_231 = torch.ops.aten.view.default(addmm_93, [8, 198, 768]);  addmm_93 = None
        add_165 = torch.ops.aten.add.Tensor(add_161, view_231);  add_161 = view_231 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
        getitem_255 = var_mean_47[0]
        getitem_256 = var_mean_47[1];  var_mean_47 = None
        add_166 = torch.ops.aten.add.Tensor(getitem_255, 1e-06);  getitem_255 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        sub_47 = torch.ops.aten.sub.Tensor(add_165, getitem_256);  getitem_256 = None
        mul_163 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
        mul_164 = torch.ops.aten.mul.Tensor(mul_163, arg138_1);  mul_163 = arg138_1 = None
        add_167 = torch.ops.aten.add.Tensor(mul_164, arg139_1);  mul_164 = arg139_1 = None
        view_232 = torch.ops.aten.view.default(add_167, [1584, 768]);  add_167 = None
        permute_142 = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg141_1, view_232, permute_142);  arg141_1 = view_232 = permute_142 = None
        view_233 = torch.ops.aten.view.default(addmm_94, [8, 198, 2304]);  addmm_94 = None
        view_234 = torch.ops.aten.view.default(view_233, [8, 198, 3, 12, 64]);  view_233 = None
        permute_143 = torch.ops.aten.permute.default(view_234, [2, 0, 3, 1, 4]);  view_234 = None
        unbind_23 = torch.ops.aten.unbind.int(permute_143);  permute_143 = None
        getitem_257 = unbind_23[0]
        getitem_258 = unbind_23[1]
        getitem_259 = unbind_23[2];  unbind_23 = None
        _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_257, getitem_258, getitem_259, None, False);  getitem_257 = getitem_258 = getitem_259 = None
        getitem_260 = _scaled_dot_product_efficient_attention_23[0];  _scaled_dot_product_efficient_attention_23 = None
        permute_144 = torch.ops.aten.permute.default(getitem_260, [0, 2, 1, 3]);  getitem_260 = None
        view_235 = torch.ops.aten.view.default(permute_144, [8, 198, 768]);  permute_144 = None
        view_236 = torch.ops.aten.view.default(view_235, [1584, 768]);  view_235 = None
        permute_145 = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg143_1, view_236, permute_145);  arg143_1 = view_236 = permute_145 = None
        view_237 = torch.ops.aten.view.default(addmm_95, [8, 198, 768]);  addmm_95 = None
        add_168 = torch.ops.aten.add.Tensor(add_165, view_237);  add_165 = view_237 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(add_168, [2], correction = 0, keepdim = True)
        getitem_264 = var_mean_48[0]
        getitem_265 = var_mean_48[1];  var_mean_48 = None
        add_169 = torch.ops.aten.add.Tensor(getitem_264, 1e-06);  getitem_264 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
        sub_48 = torch.ops.aten.sub.Tensor(add_168, getitem_265);  getitem_265 = None
        mul_165 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
        mul_166 = torch.ops.aten.mul.Tensor(mul_165, arg144_1);  mul_165 = arg144_1 = None
        add_170 = torch.ops.aten.add.Tensor(mul_166, arg145_1);  mul_166 = arg145_1 = None
        view_238 = torch.ops.aten.view.default(add_170, [1584, 768]);  add_170 = None
        permute_146 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg147_1, view_238, permute_146);  arg147_1 = view_238 = permute_146 = None
        view_239 = torch.ops.aten.view.default(addmm_96, [8, 198, 3072]);  addmm_96 = None
        mul_167 = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_168 = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_23 = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_171 = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_167, add_171);  mul_167 = add_171 = None
        view_240 = torch.ops.aten.view.default(mul_169, [1584, 3072]);  mul_169 = None
        permute_147 = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg149_1, view_240, permute_147);  arg149_1 = view_240 = permute_147 = None
        view_241 = torch.ops.aten.view.default(addmm_97, [8, 198, 768]);  addmm_97 = None
        add_172 = torch.ops.aten.add.Tensor(add_168, view_241);  add_168 = view_241 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(add_172, [2], correction = 0, keepdim = True)
        getitem_266 = var_mean_49[0]
        getitem_267 = var_mean_49[1];  var_mean_49 = None
        add_173 = torch.ops.aten.add.Tensor(getitem_266, 1e-06);  getitem_266 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        sub_49 = torch.ops.aten.sub.Tensor(add_172, getitem_267);  add_172 = getitem_267 = None
        mul_170 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
        mul_171 = torch.ops.aten.mul.Tensor(mul_170, arg150_1);  mul_170 = arg150_1 = None
        add_174 = torch.ops.aten.add.Tensor(mul_171, arg151_1);  mul_171 = arg151_1 = None
        select_2 = torch.ops.aten.select.int(add_174, 1, 0)
        select_3 = torch.ops.aten.select.int(add_174, 1, 1);  add_174 = None
        permute_148 = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg153_1, select_2, permute_148);  arg153_1 = select_2 = permute_148 = None
        permute_149 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg155_1, select_3, permute_149);  arg155_1 = select_3 = permute_149 = None
        add_175 = torch.ops.aten.add.Tensor(addmm_98, addmm_99);  addmm_98 = addmm_99 = None
        div_1 = torch.ops.aten.div.Tensor(add_175, 2);  add_175 = None
        return (div_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 224, 224), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf1, (768, 3, 16, 16), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (768,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 608256, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1, 198, 768), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1, 1, 768), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (1, 1, 768), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf8, (2304, 768), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf9, (2304,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768, 768), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf14, (3072, 768), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf15, (3072,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768, 3072), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf20, (2304, 768), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf21, (2304,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768, 768), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf26, (3072, 768), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf27, (3072,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768, 3072), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf29, (768,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf32, (2304, 768), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf33, (2304,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768, 768), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf38, (3072, 768), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf39, (3072,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768, 3072), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf44, (2304, 768), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf45, (2304,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768, 768), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf50, (3072, 768), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf51, (3072,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768, 3072), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf56, (2304, 768), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf57, (2304,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768, 768), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf62, (3072, 768), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf63, (3072,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768, 3072), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf65, (768,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf68, (2304, 768), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf69, (2304,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768, 768), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf74, (3072, 768), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf75, (3072,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768, 3072), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (768,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf79, (768,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf80, (2304, 768), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf81, (2304,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768, 768), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf84, (768,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf86, (3072, 768), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf87, (3072,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768, 3072), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf89, (768,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf92, (2304, 768), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf93, (2304,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768, 768), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf98, (3072, 768), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf99, (3072,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf100, (768, 3072), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf104, (2304, 768), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf105, (2304,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768, 768), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf110, (3072, 768), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf111, (3072,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768, 3072), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf116, (2304, 768), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf117, (2304,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768, 768), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf122, (3072, 768), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf123, (3072,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768, 3072), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf125, (768,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf127, (768,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf128, (2304, 768), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf129, (2304,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768, 768), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf134, (3072, 768), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf135, (3072,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768, 3072), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf140, (2304, 768), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf141, (2304,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768, 768), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf143, (768,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf144, (768,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf146, (3072, 768), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf147, (3072,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768, 3072), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf149, (768,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf150, (768,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf151, (768,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 3072000, device=device(type='cuda', index=0))
    reader.tensor(buf152, (1000, 768), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf153, (1000,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 3072000, device=device(type='cuda', index=0))
    reader.tensor(buf154, (1000, 768), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf155, (1000,), is_leaf=True)  # arg155_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)