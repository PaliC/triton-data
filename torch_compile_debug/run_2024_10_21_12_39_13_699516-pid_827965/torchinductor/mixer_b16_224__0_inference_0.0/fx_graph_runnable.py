
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1):
        convolution_1 = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        view_97 = torch.ops.aten.view.default(convolution_1, [8, 768, 196]);  convolution_1 = None
        permute_74 = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        clone_86 = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format)
        var_mean_25 = torch.ops.aten.var_mean.correction(clone_86, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_110 = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        sub_25 = torch.ops.aten.sub.Tensor(clone_86, getitem_51);  clone_86 = getitem_51 = None
        mul_122 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_122, arg3_1);  mul_122 = arg3_1 = None
        add_111 = torch.ops.aten.add.Tensor(mul_123, arg4_1);  mul_123 = arg4_1 = None
        permute_75 = torch.ops.aten.permute.default(add_111, [0, 2, 1]);  add_111 = None
        permute_76 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        clone_87 = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        view_98 = torch.ops.aten.view.default(clone_87, [6144, 196]);  clone_87 = None
        mm_12 = torch.ops.aten.mm.default(view_98, permute_76);  view_98 = permute_76 = None
        view_99 = torch.ops.aten.view.default(mm_12, [8, 768, 384]);  mm_12 = None
        add_112 = torch.ops.aten.add.Tensor(view_99, arg6_1);  view_99 = arg6_1 = None
        mul_124 = torch.ops.aten.mul.Tensor(add_112, 0.5)
        mul_125 = torch.ops.aten.mul.Tensor(add_112, 0.7071067811865476);  add_112 = None
        erf_24 = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_113 = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_126 = torch.ops.aten.mul.Tensor(mul_124, add_113);  mul_124 = add_113 = None
        view_100 = torch.ops.aten.view.default(mul_126, [6144, 384]);  mul_126 = None
        permute_77 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg8_1, view_100, permute_77);  arg8_1 = view_100 = permute_77 = None
        view_101 = torch.ops.aten.view.default(addmm_37, [8, 768, 196]);  addmm_37 = None
        permute_78 = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
        add_114 = torch.ops.aten.add.Tensor(permute_74, permute_78);  permute_74 = permute_78 = None
        clone_90 = torch.ops.aten.clone.default(add_114, memory_format = torch.contiguous_format)
        var_mean_26 = torch.ops.aten.var_mean.correction(clone_90, [2], correction = 0, keepdim = True)
        getitem_52 = var_mean_26[0]
        getitem_53 = var_mean_26[1];  var_mean_26 = None
        add_115 = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        sub_26 = torch.ops.aten.sub.Tensor(clone_90, getitem_53);  clone_90 = getitem_53 = None
        mul_127 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
        mul_128 = torch.ops.aten.mul.Tensor(mul_127, arg9_1);  mul_127 = arg9_1 = None
        add_116 = torch.ops.aten.add.Tensor(mul_128, arg10_1);  mul_128 = arg10_1 = None
        view_102 = torch.ops.aten.view.default(add_116, [1568, 768]);  add_116 = None
        permute_79 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg12_1, view_102, permute_79);  arg12_1 = view_102 = permute_79 = None
        view_103 = torch.ops.aten.view.default(addmm_38, [8, 196, 3072]);  addmm_38 = None
        mul_129 = torch.ops.aten.mul.Tensor(view_103, 0.5)
        mul_130 = torch.ops.aten.mul.Tensor(view_103, 0.7071067811865476);  view_103 = None
        erf_25 = torch.ops.aten.erf.default(mul_130);  mul_130 = None
        add_117 = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_131 = torch.ops.aten.mul.Tensor(mul_129, add_117);  mul_129 = add_117 = None
        view_104 = torch.ops.aten.view.default(mul_131, [1568, 3072]);  mul_131 = None
        permute_80 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg14_1, view_104, permute_80);  arg14_1 = view_104 = permute_80 = None
        view_105 = torch.ops.aten.view.default(addmm_39, [8, 196, 768]);  addmm_39 = None
        add_118 = torch.ops.aten.add.Tensor(add_114, view_105);  add_114 = view_105 = None
        clone_93 = torch.ops.aten.clone.default(add_118, memory_format = torch.contiguous_format)
        var_mean_27 = torch.ops.aten.var_mean.correction(clone_93, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_27[0]
        getitem_55 = var_mean_27[1];  var_mean_27 = None
        add_119 = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_27 = torch.ops.aten.sub.Tensor(clone_93, getitem_55);  clone_93 = getitem_55 = None
        mul_132 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
        mul_133 = torch.ops.aten.mul.Tensor(mul_132, arg15_1);  mul_132 = arg15_1 = None
        add_120 = torch.ops.aten.add.Tensor(mul_133, arg16_1);  mul_133 = arg16_1 = None
        permute_81 = torch.ops.aten.permute.default(add_120, [0, 2, 1]);  add_120 = None
        permute_82 = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        clone_94 = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_106 = torch.ops.aten.view.default(clone_94, [6144, 196]);  clone_94 = None
        mm_13 = torch.ops.aten.mm.default(view_106, permute_82);  view_106 = permute_82 = None
        view_107 = torch.ops.aten.view.default(mm_13, [8, 768, 384]);  mm_13 = None
        add_121 = torch.ops.aten.add.Tensor(view_107, arg18_1);  view_107 = arg18_1 = None
        mul_134 = torch.ops.aten.mul.Tensor(add_121, 0.5)
        mul_135 = torch.ops.aten.mul.Tensor(add_121, 0.7071067811865476);  add_121 = None
        erf_26 = torch.ops.aten.erf.default(mul_135);  mul_135 = None
        add_122 = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_136 = torch.ops.aten.mul.Tensor(mul_134, add_122);  mul_134 = add_122 = None
        view_108 = torch.ops.aten.view.default(mul_136, [6144, 384]);  mul_136 = None
        permute_83 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg20_1, view_108, permute_83);  arg20_1 = view_108 = permute_83 = None
        view_109 = torch.ops.aten.view.default(addmm_40, [8, 768, 196]);  addmm_40 = None
        permute_84 = torch.ops.aten.permute.default(view_109, [0, 2, 1]);  view_109 = None
        add_123 = torch.ops.aten.add.Tensor(add_118, permute_84);  add_118 = permute_84 = None
        clone_97 = torch.ops.aten.clone.default(add_123, memory_format = torch.contiguous_format)
        var_mean_28 = torch.ops.aten.var_mean.correction(clone_97, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_28[0]
        getitem_57 = var_mean_28[1];  var_mean_28 = None
        add_124 = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        sub_28 = torch.ops.aten.sub.Tensor(clone_97, getitem_57);  clone_97 = getitem_57 = None
        mul_137 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_137, arg21_1);  mul_137 = arg21_1 = None
        add_125 = torch.ops.aten.add.Tensor(mul_138, arg22_1);  mul_138 = arg22_1 = None
        view_110 = torch.ops.aten.view.default(add_125, [1568, 768]);  add_125 = None
        permute_85 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg24_1, view_110, permute_85);  arg24_1 = view_110 = permute_85 = None
        view_111 = torch.ops.aten.view.default(addmm_41, [8, 196, 3072]);  addmm_41 = None
        mul_139 = torch.ops.aten.mul.Tensor(view_111, 0.5)
        mul_140 = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476);  view_111 = None
        erf_27 = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_126 = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_139, add_126);  mul_139 = add_126 = None
        view_112 = torch.ops.aten.view.default(mul_141, [1568, 3072]);  mul_141 = None
        permute_86 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg26_1, view_112, permute_86);  arg26_1 = view_112 = permute_86 = None
        view_113 = torch.ops.aten.view.default(addmm_42, [8, 196, 768]);  addmm_42 = None
        add_127 = torch.ops.aten.add.Tensor(add_123, view_113);  add_123 = view_113 = None
        clone_100 = torch.ops.aten.clone.default(add_127, memory_format = torch.contiguous_format)
        var_mean_29 = torch.ops.aten.var_mean.correction(clone_100, [2], correction = 0, keepdim = True)
        getitem_58 = var_mean_29[0]
        getitem_59 = var_mean_29[1];  var_mean_29 = None
        add_128 = torch.ops.aten.add.Tensor(getitem_58, 1e-06);  getitem_58 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_29 = torch.ops.aten.sub.Tensor(clone_100, getitem_59);  clone_100 = getitem_59 = None
        mul_142 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
        mul_143 = torch.ops.aten.mul.Tensor(mul_142, arg27_1);  mul_142 = arg27_1 = None
        add_129 = torch.ops.aten.add.Tensor(mul_143, arg28_1);  mul_143 = arg28_1 = None
        permute_87 = torch.ops.aten.permute.default(add_129, [0, 2, 1]);  add_129 = None
        permute_88 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        clone_101 = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
        view_114 = torch.ops.aten.view.default(clone_101, [6144, 196]);  clone_101 = None
        mm_14 = torch.ops.aten.mm.default(view_114, permute_88);  view_114 = permute_88 = None
        view_115 = torch.ops.aten.view.default(mm_14, [8, 768, 384]);  mm_14 = None
        add_130 = torch.ops.aten.add.Tensor(view_115, arg30_1);  view_115 = arg30_1 = None
        mul_144 = torch.ops.aten.mul.Tensor(add_130, 0.5)
        mul_145 = torch.ops.aten.mul.Tensor(add_130, 0.7071067811865476);  add_130 = None
        erf_28 = torch.ops.aten.erf.default(mul_145);  mul_145 = None
        add_131 = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_144, add_131);  mul_144 = add_131 = None
        view_116 = torch.ops.aten.view.default(mul_146, [6144, 384]);  mul_146 = None
        permute_89 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg32_1, view_116, permute_89);  arg32_1 = view_116 = permute_89 = None
        view_117 = torch.ops.aten.view.default(addmm_43, [8, 768, 196]);  addmm_43 = None
        permute_90 = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
        add_132 = torch.ops.aten.add.Tensor(add_127, permute_90);  add_127 = permute_90 = None
        clone_104 = torch.ops.aten.clone.default(add_132, memory_format = torch.contiguous_format)
        var_mean_30 = torch.ops.aten.var_mean.correction(clone_104, [2], correction = 0, keepdim = True)
        getitem_60 = var_mean_30[0]
        getitem_61 = var_mean_30[1];  var_mean_30 = None
        add_133 = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_30 = torch.ops.aten.sub.Tensor(clone_104, getitem_61);  clone_104 = getitem_61 = None
        mul_147 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
        mul_148 = torch.ops.aten.mul.Tensor(mul_147, arg33_1);  mul_147 = arg33_1 = None
        add_134 = torch.ops.aten.add.Tensor(mul_148, arg34_1);  mul_148 = arg34_1 = None
        view_118 = torch.ops.aten.view.default(add_134, [1568, 768]);  add_134 = None
        permute_91 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg36_1, view_118, permute_91);  arg36_1 = view_118 = permute_91 = None
        view_119 = torch.ops.aten.view.default(addmm_44, [8, 196, 3072]);  addmm_44 = None
        mul_149 = torch.ops.aten.mul.Tensor(view_119, 0.5)
        mul_150 = torch.ops.aten.mul.Tensor(view_119, 0.7071067811865476);  view_119 = None
        erf_29 = torch.ops.aten.erf.default(mul_150);  mul_150 = None
        add_135 = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_151 = torch.ops.aten.mul.Tensor(mul_149, add_135);  mul_149 = add_135 = None
        view_120 = torch.ops.aten.view.default(mul_151, [1568, 3072]);  mul_151 = None
        permute_92 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg38_1, view_120, permute_92);  arg38_1 = view_120 = permute_92 = None
        view_121 = torch.ops.aten.view.default(addmm_45, [8, 196, 768]);  addmm_45 = None
        add_136 = torch.ops.aten.add.Tensor(add_132, view_121);  add_132 = view_121 = None
        clone_107 = torch.ops.aten.clone.default(add_136, memory_format = torch.contiguous_format)
        var_mean_31 = torch.ops.aten.var_mean.correction(clone_107, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_31[0]
        getitem_63 = var_mean_31[1];  var_mean_31 = None
        add_137 = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        sub_31 = torch.ops.aten.sub.Tensor(clone_107, getitem_63);  clone_107 = getitem_63 = None
        mul_152 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_152, arg39_1);  mul_152 = arg39_1 = None
        add_138 = torch.ops.aten.add.Tensor(mul_153, arg40_1);  mul_153 = arg40_1 = None
        permute_93 = torch.ops.aten.permute.default(add_138, [0, 2, 1]);  add_138 = None
        permute_94 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        clone_108 = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_122 = torch.ops.aten.view.default(clone_108, [6144, 196]);  clone_108 = None
        mm_15 = torch.ops.aten.mm.default(view_122, permute_94);  view_122 = permute_94 = None
        view_123 = torch.ops.aten.view.default(mm_15, [8, 768, 384]);  mm_15 = None
        add_139 = torch.ops.aten.add.Tensor(view_123, arg42_1);  view_123 = arg42_1 = None
        mul_154 = torch.ops.aten.mul.Tensor(add_139, 0.5)
        mul_155 = torch.ops.aten.mul.Tensor(add_139, 0.7071067811865476);  add_139 = None
        erf_30 = torch.ops.aten.erf.default(mul_155);  mul_155 = None
        add_140 = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_156 = torch.ops.aten.mul.Tensor(mul_154, add_140);  mul_154 = add_140 = None
        view_124 = torch.ops.aten.view.default(mul_156, [6144, 384]);  mul_156 = None
        permute_95 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg44_1, view_124, permute_95);  arg44_1 = view_124 = permute_95 = None
        view_125 = torch.ops.aten.view.default(addmm_46, [8, 768, 196]);  addmm_46 = None
        permute_96 = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
        add_141 = torch.ops.aten.add.Tensor(add_136, permute_96);  add_136 = permute_96 = None
        clone_111 = torch.ops.aten.clone.default(add_141, memory_format = torch.contiguous_format)
        var_mean_32 = torch.ops.aten.var_mean.correction(clone_111, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_32[0]
        getitem_65 = var_mean_32[1];  var_mean_32 = None
        add_142 = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        sub_32 = torch.ops.aten.sub.Tensor(clone_111, getitem_65);  clone_111 = getitem_65 = None
        mul_157 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_157, arg45_1);  mul_157 = arg45_1 = None
        add_143 = torch.ops.aten.add.Tensor(mul_158, arg46_1);  mul_158 = arg46_1 = None
        view_126 = torch.ops.aten.view.default(add_143, [1568, 768]);  add_143 = None
        permute_97 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg48_1, view_126, permute_97);  arg48_1 = view_126 = permute_97 = None
        view_127 = torch.ops.aten.view.default(addmm_47, [8, 196, 3072]);  addmm_47 = None
        mul_159 = torch.ops.aten.mul.Tensor(view_127, 0.5)
        mul_160 = torch.ops.aten.mul.Tensor(view_127, 0.7071067811865476);  view_127 = None
        erf_31 = torch.ops.aten.erf.default(mul_160);  mul_160 = None
        add_144 = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_159, add_144);  mul_159 = add_144 = None
        view_128 = torch.ops.aten.view.default(mul_161, [1568, 3072]);  mul_161 = None
        permute_98 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg50_1, view_128, permute_98);  arg50_1 = view_128 = permute_98 = None
        view_129 = torch.ops.aten.view.default(addmm_48, [8, 196, 768]);  addmm_48 = None
        add_145 = torch.ops.aten.add.Tensor(add_141, view_129);  add_141 = view_129 = None
        clone_114 = torch.ops.aten.clone.default(add_145, memory_format = torch.contiguous_format)
        var_mean_33 = torch.ops.aten.var_mean.correction(clone_114, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_33[0]
        getitem_67 = var_mean_33[1];  var_mean_33 = None
        add_146 = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_33 = torch.ops.aten.sub.Tensor(clone_114, getitem_67);  clone_114 = getitem_67 = None
        mul_162 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
        mul_163 = torch.ops.aten.mul.Tensor(mul_162, arg51_1);  mul_162 = arg51_1 = None
        add_147 = torch.ops.aten.add.Tensor(mul_163, arg52_1);  mul_163 = arg52_1 = None
        permute_99 = torch.ops.aten.permute.default(add_147, [0, 2, 1]);  add_147 = None
        permute_100 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        clone_115 = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
        view_130 = torch.ops.aten.view.default(clone_115, [6144, 196]);  clone_115 = None
        mm_16 = torch.ops.aten.mm.default(view_130, permute_100);  view_130 = permute_100 = None
        view_131 = torch.ops.aten.view.default(mm_16, [8, 768, 384]);  mm_16 = None
        add_148 = torch.ops.aten.add.Tensor(view_131, arg54_1);  view_131 = arg54_1 = None
        mul_164 = torch.ops.aten.mul.Tensor(add_148, 0.5)
        mul_165 = torch.ops.aten.mul.Tensor(add_148, 0.7071067811865476);  add_148 = None
        erf_32 = torch.ops.aten.erf.default(mul_165);  mul_165 = None
        add_149 = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_166 = torch.ops.aten.mul.Tensor(mul_164, add_149);  mul_164 = add_149 = None
        view_132 = torch.ops.aten.view.default(mul_166, [6144, 384]);  mul_166 = None
        permute_101 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg56_1, view_132, permute_101);  arg56_1 = view_132 = permute_101 = None
        view_133 = torch.ops.aten.view.default(addmm_49, [8, 768, 196]);  addmm_49 = None
        permute_102 = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
        add_150 = torch.ops.aten.add.Tensor(add_145, permute_102);  add_145 = permute_102 = None
        clone_118 = torch.ops.aten.clone.default(add_150, memory_format = torch.contiguous_format)
        var_mean_34 = torch.ops.aten.var_mean.correction(clone_118, [2], correction = 0, keepdim = True)
        getitem_68 = var_mean_34[0]
        getitem_69 = var_mean_34[1];  var_mean_34 = None
        add_151 = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        sub_34 = torch.ops.aten.sub.Tensor(clone_118, getitem_69);  clone_118 = getitem_69 = None
        mul_167 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
        mul_168 = torch.ops.aten.mul.Tensor(mul_167, arg57_1);  mul_167 = arg57_1 = None
        add_152 = torch.ops.aten.add.Tensor(mul_168, arg58_1);  mul_168 = arg58_1 = None
        view_134 = torch.ops.aten.view.default(add_152, [1568, 768]);  add_152 = None
        permute_103 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg60_1, view_134, permute_103);  arg60_1 = view_134 = permute_103 = None
        view_135 = torch.ops.aten.view.default(addmm_50, [8, 196, 3072]);  addmm_50 = None
        mul_169 = torch.ops.aten.mul.Tensor(view_135, 0.5)
        mul_170 = torch.ops.aten.mul.Tensor(view_135, 0.7071067811865476);  view_135 = None
        erf_33 = torch.ops.aten.erf.default(mul_170);  mul_170 = None
        add_153 = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_171 = torch.ops.aten.mul.Tensor(mul_169, add_153);  mul_169 = add_153 = None
        view_136 = torch.ops.aten.view.default(mul_171, [1568, 3072]);  mul_171 = None
        permute_104 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg62_1, view_136, permute_104);  arg62_1 = view_136 = permute_104 = None
        view_137 = torch.ops.aten.view.default(addmm_51, [8, 196, 768]);  addmm_51 = None
        add_154 = torch.ops.aten.add.Tensor(add_150, view_137);  add_150 = view_137 = None
        clone_121 = torch.ops.aten.clone.default(add_154, memory_format = torch.contiguous_format)
        var_mean_35 = torch.ops.aten.var_mean.correction(clone_121, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_35[0]
        getitem_71 = var_mean_35[1];  var_mean_35 = None
        add_155 = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        sub_35 = torch.ops.aten.sub.Tensor(clone_121, getitem_71);  clone_121 = getitem_71 = None
        mul_172 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_172, arg63_1);  mul_172 = arg63_1 = None
        add_156 = torch.ops.aten.add.Tensor(mul_173, arg64_1);  mul_173 = arg64_1 = None
        permute_105 = torch.ops.aten.permute.default(add_156, [0, 2, 1]);  add_156 = None
        permute_106 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        clone_122 = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_138 = torch.ops.aten.view.default(clone_122, [6144, 196]);  clone_122 = None
        mm_17 = torch.ops.aten.mm.default(view_138, permute_106);  view_138 = permute_106 = None
        view_139 = torch.ops.aten.view.default(mm_17, [8, 768, 384]);  mm_17 = None
        add_157 = torch.ops.aten.add.Tensor(view_139, arg66_1);  view_139 = arg66_1 = None
        mul_174 = torch.ops.aten.mul.Tensor(add_157, 0.5)
        mul_175 = torch.ops.aten.mul.Tensor(add_157, 0.7071067811865476);  add_157 = None
        erf_34 = torch.ops.aten.erf.default(mul_175);  mul_175 = None
        add_158 = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_174, add_158);  mul_174 = add_158 = None
        view_140 = torch.ops.aten.view.default(mul_176, [6144, 384]);  mul_176 = None
        permute_107 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg68_1, view_140, permute_107);  arg68_1 = view_140 = permute_107 = None
        view_141 = torch.ops.aten.view.default(addmm_52, [8, 768, 196]);  addmm_52 = None
        permute_108 = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        add_159 = torch.ops.aten.add.Tensor(add_154, permute_108);  add_154 = permute_108 = None
        clone_125 = torch.ops.aten.clone.default(add_159, memory_format = torch.contiguous_format)
        var_mean_36 = torch.ops.aten.var_mean.correction(clone_125, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_36[0]
        getitem_73 = var_mean_36[1];  var_mean_36 = None
        add_160 = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        sub_36 = torch.ops.aten.sub.Tensor(clone_125, getitem_73);  clone_125 = getitem_73 = None
        mul_177 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
        mul_178 = torch.ops.aten.mul.Tensor(mul_177, arg69_1);  mul_177 = arg69_1 = None
        add_161 = torch.ops.aten.add.Tensor(mul_178, arg70_1);  mul_178 = arg70_1 = None
        view_142 = torch.ops.aten.view.default(add_161, [1568, 768]);  add_161 = None
        permute_109 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg72_1, view_142, permute_109);  arg72_1 = view_142 = permute_109 = None
        view_143 = torch.ops.aten.view.default(addmm_53, [8, 196, 3072]);  addmm_53 = None
        mul_179 = torch.ops.aten.mul.Tensor(view_143, 0.5)
        mul_180 = torch.ops.aten.mul.Tensor(view_143, 0.7071067811865476);  view_143 = None
        erf_35 = torch.ops.aten.erf.default(mul_180);  mul_180 = None
        add_162 = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_179, add_162);  mul_179 = add_162 = None
        view_144 = torch.ops.aten.view.default(mul_181, [1568, 3072]);  mul_181 = None
        permute_110 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg74_1, view_144, permute_110);  arg74_1 = view_144 = permute_110 = None
        view_145 = torch.ops.aten.view.default(addmm_54, [8, 196, 768]);  addmm_54 = None
        add_163 = torch.ops.aten.add.Tensor(add_159, view_145);  add_159 = view_145 = None
        clone_128 = torch.ops.aten.clone.default(add_163, memory_format = torch.contiguous_format)
        var_mean_37 = torch.ops.aten.var_mean.correction(clone_128, [2], correction = 0, keepdim = True)
        getitem_74 = var_mean_37[0]
        getitem_75 = var_mean_37[1];  var_mean_37 = None
        add_164 = torch.ops.aten.add.Tensor(getitem_74, 1e-06);  getitem_74 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        sub_37 = torch.ops.aten.sub.Tensor(clone_128, getitem_75);  clone_128 = getitem_75 = None
        mul_182 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
        mul_183 = torch.ops.aten.mul.Tensor(mul_182, arg75_1);  mul_182 = arg75_1 = None
        add_165 = torch.ops.aten.add.Tensor(mul_183, arg76_1);  mul_183 = arg76_1 = None
        permute_111 = torch.ops.aten.permute.default(add_165, [0, 2, 1]);  add_165 = None
        permute_112 = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        clone_129 = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        view_146 = torch.ops.aten.view.default(clone_129, [6144, 196]);  clone_129 = None
        mm_18 = torch.ops.aten.mm.default(view_146, permute_112);  view_146 = permute_112 = None
        view_147 = torch.ops.aten.view.default(mm_18, [8, 768, 384]);  mm_18 = None
        add_166 = torch.ops.aten.add.Tensor(view_147, arg78_1);  view_147 = arg78_1 = None
        mul_184 = torch.ops.aten.mul.Tensor(add_166, 0.5)
        mul_185 = torch.ops.aten.mul.Tensor(add_166, 0.7071067811865476);  add_166 = None
        erf_36 = torch.ops.aten.erf.default(mul_185);  mul_185 = None
        add_167 = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_186 = torch.ops.aten.mul.Tensor(mul_184, add_167);  mul_184 = add_167 = None
        view_148 = torch.ops.aten.view.default(mul_186, [6144, 384]);  mul_186 = None
        permute_113 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg80_1, view_148, permute_113);  arg80_1 = view_148 = permute_113 = None
        view_149 = torch.ops.aten.view.default(addmm_55, [8, 768, 196]);  addmm_55 = None
        permute_114 = torch.ops.aten.permute.default(view_149, [0, 2, 1]);  view_149 = None
        add_168 = torch.ops.aten.add.Tensor(add_163, permute_114);  add_163 = permute_114 = None
        clone_132 = torch.ops.aten.clone.default(add_168, memory_format = torch.contiguous_format)
        var_mean_38 = torch.ops.aten.var_mean.correction(clone_132, [2], correction = 0, keepdim = True)
        getitem_76 = var_mean_38[0]
        getitem_77 = var_mean_38[1];  var_mean_38 = None
        add_169 = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
        sub_38 = torch.ops.aten.sub.Tensor(clone_132, getitem_77);  clone_132 = getitem_77 = None
        mul_187 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
        mul_188 = torch.ops.aten.mul.Tensor(mul_187, arg81_1);  mul_187 = arg81_1 = None
        add_170 = torch.ops.aten.add.Tensor(mul_188, arg82_1);  mul_188 = arg82_1 = None
        view_150 = torch.ops.aten.view.default(add_170, [1568, 768]);  add_170 = None
        permute_115 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg84_1, view_150, permute_115);  arg84_1 = view_150 = permute_115 = None
        view_151 = torch.ops.aten.view.default(addmm_56, [8, 196, 3072]);  addmm_56 = None
        mul_189 = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_190 = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
        erf_37 = torch.ops.aten.erf.default(mul_190);  mul_190 = None
        add_171 = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_191 = torch.ops.aten.mul.Tensor(mul_189, add_171);  mul_189 = add_171 = None
        view_152 = torch.ops.aten.view.default(mul_191, [1568, 3072]);  mul_191 = None
        permute_116 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg86_1, view_152, permute_116);  arg86_1 = view_152 = permute_116 = None
        view_153 = torch.ops.aten.view.default(addmm_57, [8, 196, 768]);  addmm_57 = None
        add_172 = torch.ops.aten.add.Tensor(add_168, view_153);  add_168 = view_153 = None
        clone_135 = torch.ops.aten.clone.default(add_172, memory_format = torch.contiguous_format)
        var_mean_39 = torch.ops.aten.var_mean.correction(clone_135, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_39[0]
        getitem_79 = var_mean_39[1];  var_mean_39 = None
        add_173 = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        sub_39 = torch.ops.aten.sub.Tensor(clone_135, getitem_79);  clone_135 = getitem_79 = None
        mul_192 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_192, arg87_1);  mul_192 = arg87_1 = None
        add_174 = torch.ops.aten.add.Tensor(mul_193, arg88_1);  mul_193 = arg88_1 = None
        permute_117 = torch.ops.aten.permute.default(add_174, [0, 2, 1]);  add_174 = None
        permute_118 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        clone_136 = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_154 = torch.ops.aten.view.default(clone_136, [6144, 196]);  clone_136 = None
        mm_19 = torch.ops.aten.mm.default(view_154, permute_118);  view_154 = permute_118 = None
        view_155 = torch.ops.aten.view.default(mm_19, [8, 768, 384]);  mm_19 = None
        add_175 = torch.ops.aten.add.Tensor(view_155, arg90_1);  view_155 = arg90_1 = None
        mul_194 = torch.ops.aten.mul.Tensor(add_175, 0.5)
        mul_195 = torch.ops.aten.mul.Tensor(add_175, 0.7071067811865476);  add_175 = None
        erf_38 = torch.ops.aten.erf.default(mul_195);  mul_195 = None
        add_176 = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_196 = torch.ops.aten.mul.Tensor(mul_194, add_176);  mul_194 = add_176 = None
        view_156 = torch.ops.aten.view.default(mul_196, [6144, 384]);  mul_196 = None
        permute_119 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg92_1, view_156, permute_119);  arg92_1 = view_156 = permute_119 = None
        view_157 = torch.ops.aten.view.default(addmm_58, [8, 768, 196]);  addmm_58 = None
        permute_120 = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
        add_177 = torch.ops.aten.add.Tensor(add_172, permute_120);  add_172 = permute_120 = None
        clone_139 = torch.ops.aten.clone.default(add_177, memory_format = torch.contiguous_format)
        var_mean_40 = torch.ops.aten.var_mean.correction(clone_139, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_40[0]
        getitem_81 = var_mean_40[1];  var_mean_40 = None
        add_178 = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_40 = torch.ops.aten.sub.Tensor(clone_139, getitem_81);  clone_139 = getitem_81 = None
        mul_197 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
        mul_198 = torch.ops.aten.mul.Tensor(mul_197, arg93_1);  mul_197 = arg93_1 = None
        add_179 = torch.ops.aten.add.Tensor(mul_198, arg94_1);  mul_198 = arg94_1 = None
        view_158 = torch.ops.aten.view.default(add_179, [1568, 768]);  add_179 = None
        permute_121 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg96_1, view_158, permute_121);  arg96_1 = view_158 = permute_121 = None
        view_159 = torch.ops.aten.view.default(addmm_59, [8, 196, 3072]);  addmm_59 = None
        mul_199 = torch.ops.aten.mul.Tensor(view_159, 0.5)
        mul_200 = torch.ops.aten.mul.Tensor(view_159, 0.7071067811865476);  view_159 = None
        erf_39 = torch.ops.aten.erf.default(mul_200);  mul_200 = None
        add_180 = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_201 = torch.ops.aten.mul.Tensor(mul_199, add_180);  mul_199 = add_180 = None
        view_160 = torch.ops.aten.view.default(mul_201, [1568, 3072]);  mul_201 = None
        permute_122 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg98_1, view_160, permute_122);  arg98_1 = view_160 = permute_122 = None
        view_161 = torch.ops.aten.view.default(addmm_60, [8, 196, 768]);  addmm_60 = None
        add_181 = torch.ops.aten.add.Tensor(add_177, view_161);  add_177 = view_161 = None
        clone_142 = torch.ops.aten.clone.default(add_181, memory_format = torch.contiguous_format)
        var_mean_41 = torch.ops.aten.var_mean.correction(clone_142, [2], correction = 0, keepdim = True)
        getitem_82 = var_mean_41[0]
        getitem_83 = var_mean_41[1];  var_mean_41 = None
        add_182 = torch.ops.aten.add.Tensor(getitem_82, 1e-06);  getitem_82 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        sub_41 = torch.ops.aten.sub.Tensor(clone_142, getitem_83);  clone_142 = getitem_83 = None
        mul_202 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
        mul_203 = torch.ops.aten.mul.Tensor(mul_202, arg99_1);  mul_202 = arg99_1 = None
        add_183 = torch.ops.aten.add.Tensor(mul_203, arg100_1);  mul_203 = arg100_1 = None
        permute_123 = torch.ops.aten.permute.default(add_183, [0, 2, 1]);  add_183 = None
        permute_124 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        clone_143 = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
        view_162 = torch.ops.aten.view.default(clone_143, [6144, 196]);  clone_143 = None
        mm_20 = torch.ops.aten.mm.default(view_162, permute_124);  view_162 = permute_124 = None
        view_163 = torch.ops.aten.view.default(mm_20, [8, 768, 384]);  mm_20 = None
        add_184 = torch.ops.aten.add.Tensor(view_163, arg102_1);  view_163 = arg102_1 = None
        mul_204 = torch.ops.aten.mul.Tensor(add_184, 0.5)
        mul_205 = torch.ops.aten.mul.Tensor(add_184, 0.7071067811865476);  add_184 = None
        erf_40 = torch.ops.aten.erf.default(mul_205);  mul_205 = None
        add_185 = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_206 = torch.ops.aten.mul.Tensor(mul_204, add_185);  mul_204 = add_185 = None
        view_164 = torch.ops.aten.view.default(mul_206, [6144, 384]);  mul_206 = None
        permute_125 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg104_1, view_164, permute_125);  arg104_1 = view_164 = permute_125 = None
        view_165 = torch.ops.aten.view.default(addmm_61, [8, 768, 196]);  addmm_61 = None
        permute_126 = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
        add_186 = torch.ops.aten.add.Tensor(add_181, permute_126);  add_181 = permute_126 = None
        clone_146 = torch.ops.aten.clone.default(add_186, memory_format = torch.contiguous_format)
        var_mean_42 = torch.ops.aten.var_mean.correction(clone_146, [2], correction = 0, keepdim = True)
        getitem_84 = var_mean_42[0]
        getitem_85 = var_mean_42[1];  var_mean_42 = None
        add_187 = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
        sub_42 = torch.ops.aten.sub.Tensor(clone_146, getitem_85);  clone_146 = getitem_85 = None
        mul_207 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
        mul_208 = torch.ops.aten.mul.Tensor(mul_207, arg105_1);  mul_207 = arg105_1 = None
        add_188 = torch.ops.aten.add.Tensor(mul_208, arg106_1);  mul_208 = arg106_1 = None
        view_166 = torch.ops.aten.view.default(add_188, [1568, 768]);  add_188 = None
        permute_127 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg108_1, view_166, permute_127);  arg108_1 = view_166 = permute_127 = None
        view_167 = torch.ops.aten.view.default(addmm_62, [8, 196, 3072]);  addmm_62 = None
        mul_209 = torch.ops.aten.mul.Tensor(view_167, 0.5)
        mul_210 = torch.ops.aten.mul.Tensor(view_167, 0.7071067811865476);  view_167 = None
        erf_41 = torch.ops.aten.erf.default(mul_210);  mul_210 = None
        add_189 = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_211 = torch.ops.aten.mul.Tensor(mul_209, add_189);  mul_209 = add_189 = None
        view_168 = torch.ops.aten.view.default(mul_211, [1568, 3072]);  mul_211 = None
        permute_128 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg110_1, view_168, permute_128);  arg110_1 = view_168 = permute_128 = None
        view_169 = torch.ops.aten.view.default(addmm_63, [8, 196, 768]);  addmm_63 = None
        add_190 = torch.ops.aten.add.Tensor(add_186, view_169);  add_186 = view_169 = None
        clone_149 = torch.ops.aten.clone.default(add_190, memory_format = torch.contiguous_format)
        var_mean_43 = torch.ops.aten.var_mean.correction(clone_149, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_43[0]
        getitem_87 = var_mean_43[1];  var_mean_43 = None
        add_191 = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_43 = torch.ops.aten.sub.Tensor(clone_149, getitem_87);  clone_149 = getitem_87 = None
        mul_212 = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
        mul_213 = torch.ops.aten.mul.Tensor(mul_212, arg111_1);  mul_212 = arg111_1 = None
        add_192 = torch.ops.aten.add.Tensor(mul_213, arg112_1);  mul_213 = arg112_1 = None
        permute_129 = torch.ops.aten.permute.default(add_192, [0, 2, 1]);  add_192 = None
        permute_130 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        clone_150 = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_170 = torch.ops.aten.view.default(clone_150, [6144, 196]);  clone_150 = None
        mm_21 = torch.ops.aten.mm.default(view_170, permute_130);  view_170 = permute_130 = None
        view_171 = torch.ops.aten.view.default(mm_21, [8, 768, 384]);  mm_21 = None
        add_193 = torch.ops.aten.add.Tensor(view_171, arg114_1);  view_171 = arg114_1 = None
        mul_214 = torch.ops.aten.mul.Tensor(add_193, 0.5)
        mul_215 = torch.ops.aten.mul.Tensor(add_193, 0.7071067811865476);  add_193 = None
        erf_42 = torch.ops.aten.erf.default(mul_215);  mul_215 = None
        add_194 = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_216 = torch.ops.aten.mul.Tensor(mul_214, add_194);  mul_214 = add_194 = None
        view_172 = torch.ops.aten.view.default(mul_216, [6144, 384]);  mul_216 = None
        permute_131 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg116_1, view_172, permute_131);  arg116_1 = view_172 = permute_131 = None
        view_173 = torch.ops.aten.view.default(addmm_64, [8, 768, 196]);  addmm_64 = None
        permute_132 = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
        add_195 = torch.ops.aten.add.Tensor(add_190, permute_132);  add_190 = permute_132 = None
        clone_153 = torch.ops.aten.clone.default(add_195, memory_format = torch.contiguous_format)
        var_mean_44 = torch.ops.aten.var_mean.correction(clone_153, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_44[0]
        getitem_89 = var_mean_44[1];  var_mean_44 = None
        add_196 = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        sub_44 = torch.ops.aten.sub.Tensor(clone_153, getitem_89);  clone_153 = getitem_89 = None
        mul_217 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
        mul_218 = torch.ops.aten.mul.Tensor(mul_217, arg117_1);  mul_217 = arg117_1 = None
        add_197 = torch.ops.aten.add.Tensor(mul_218, arg118_1);  mul_218 = arg118_1 = None
        view_174 = torch.ops.aten.view.default(add_197, [1568, 768]);  add_197 = None
        permute_133 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg120_1, view_174, permute_133);  arg120_1 = view_174 = permute_133 = None
        view_175 = torch.ops.aten.view.default(addmm_65, [8, 196, 3072]);  addmm_65 = None
        mul_219 = torch.ops.aten.mul.Tensor(view_175, 0.5)
        mul_220 = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
        erf_43 = torch.ops.aten.erf.default(mul_220);  mul_220 = None
        add_198 = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_219, add_198);  mul_219 = add_198 = None
        view_176 = torch.ops.aten.view.default(mul_221, [1568, 3072]);  mul_221 = None
        permute_134 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg122_1, view_176, permute_134);  arg122_1 = view_176 = permute_134 = None
        view_177 = torch.ops.aten.view.default(addmm_66, [8, 196, 768]);  addmm_66 = None
        add_199 = torch.ops.aten.add.Tensor(add_195, view_177);  add_195 = view_177 = None
        clone_156 = torch.ops.aten.clone.default(add_199, memory_format = torch.contiguous_format)
        var_mean_45 = torch.ops.aten.var_mean.correction(clone_156, [2], correction = 0, keepdim = True)
        getitem_90 = var_mean_45[0]
        getitem_91 = var_mean_45[1];  var_mean_45 = None
        add_200 = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
        sub_45 = torch.ops.aten.sub.Tensor(clone_156, getitem_91);  clone_156 = getitem_91 = None
        mul_222 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_222, arg123_1);  mul_222 = arg123_1 = None
        add_201 = torch.ops.aten.add.Tensor(mul_223, arg124_1);  mul_223 = arg124_1 = None
        permute_135 = torch.ops.aten.permute.default(add_201, [0, 2, 1]);  add_201 = None
        permute_136 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        clone_157 = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_178 = torch.ops.aten.view.default(clone_157, [6144, 196]);  clone_157 = None
        mm_22 = torch.ops.aten.mm.default(view_178, permute_136);  view_178 = permute_136 = None
        view_179 = torch.ops.aten.view.default(mm_22, [8, 768, 384]);  mm_22 = None
        add_202 = torch.ops.aten.add.Tensor(view_179, arg126_1);  view_179 = arg126_1 = None
        mul_224 = torch.ops.aten.mul.Tensor(add_202, 0.5)
        mul_225 = torch.ops.aten.mul.Tensor(add_202, 0.7071067811865476);  add_202 = None
        erf_44 = torch.ops.aten.erf.default(mul_225);  mul_225 = None
        add_203 = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_226 = torch.ops.aten.mul.Tensor(mul_224, add_203);  mul_224 = add_203 = None
        view_180 = torch.ops.aten.view.default(mul_226, [6144, 384]);  mul_226 = None
        permute_137 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg128_1, view_180, permute_137);  arg128_1 = view_180 = permute_137 = None
        view_181 = torch.ops.aten.view.default(addmm_67, [8, 768, 196]);  addmm_67 = None
        permute_138 = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
        add_204 = torch.ops.aten.add.Tensor(add_199, permute_138);  add_199 = permute_138 = None
        clone_160 = torch.ops.aten.clone.default(add_204, memory_format = torch.contiguous_format)
        var_mean_46 = torch.ops.aten.var_mean.correction(clone_160, [2], correction = 0, keepdim = True)
        getitem_92 = var_mean_46[0]
        getitem_93 = var_mean_46[1];  var_mean_46 = None
        add_205 = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
        sub_46 = torch.ops.aten.sub.Tensor(clone_160, getitem_93);  clone_160 = getitem_93 = None
        mul_227 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
        mul_228 = torch.ops.aten.mul.Tensor(mul_227, arg129_1);  mul_227 = arg129_1 = None
        add_206 = torch.ops.aten.add.Tensor(mul_228, arg130_1);  mul_228 = arg130_1 = None
        view_182 = torch.ops.aten.view.default(add_206, [1568, 768]);  add_206 = None
        permute_139 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg132_1, view_182, permute_139);  arg132_1 = view_182 = permute_139 = None
        view_183 = torch.ops.aten.view.default(addmm_68, [8, 196, 3072]);  addmm_68 = None
        mul_229 = torch.ops.aten.mul.Tensor(view_183, 0.5)
        mul_230 = torch.ops.aten.mul.Tensor(view_183, 0.7071067811865476);  view_183 = None
        erf_45 = torch.ops.aten.erf.default(mul_230);  mul_230 = None
        add_207 = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_231 = torch.ops.aten.mul.Tensor(mul_229, add_207);  mul_229 = add_207 = None
        view_184 = torch.ops.aten.view.default(mul_231, [1568, 3072]);  mul_231 = None
        permute_140 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg134_1, view_184, permute_140);  arg134_1 = view_184 = permute_140 = None
        view_185 = torch.ops.aten.view.default(addmm_69, [8, 196, 768]);  addmm_69 = None
        add_208 = torch.ops.aten.add.Tensor(add_204, view_185);  add_204 = view_185 = None
        clone_163 = torch.ops.aten.clone.default(add_208, memory_format = torch.contiguous_format)
        var_mean_47 = torch.ops.aten.var_mean.correction(clone_163, [2], correction = 0, keepdim = True)
        getitem_94 = var_mean_47[0]
        getitem_95 = var_mean_47[1];  var_mean_47 = None
        add_209 = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
        sub_47 = torch.ops.aten.sub.Tensor(clone_163, getitem_95);  clone_163 = getitem_95 = None
        mul_232 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_232, arg135_1);  mul_232 = arg135_1 = None
        add_210 = torch.ops.aten.add.Tensor(mul_233, arg136_1);  mul_233 = arg136_1 = None
        permute_141 = torch.ops.aten.permute.default(add_210, [0, 2, 1]);  add_210 = None
        permute_142 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        clone_164 = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
        view_186 = torch.ops.aten.view.default(clone_164, [6144, 196]);  clone_164 = None
        mm_23 = torch.ops.aten.mm.default(view_186, permute_142);  view_186 = permute_142 = None
        view_187 = torch.ops.aten.view.default(mm_23, [8, 768, 384]);  mm_23 = None
        add_211 = torch.ops.aten.add.Tensor(view_187, arg138_1);  view_187 = arg138_1 = None
        mul_234 = torch.ops.aten.mul.Tensor(add_211, 0.5)
        mul_235 = torch.ops.aten.mul.Tensor(add_211, 0.7071067811865476);  add_211 = None
        erf_46 = torch.ops.aten.erf.default(mul_235);  mul_235 = None
        add_212 = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_236 = torch.ops.aten.mul.Tensor(mul_234, add_212);  mul_234 = add_212 = None
        view_188 = torch.ops.aten.view.default(mul_236, [6144, 384]);  mul_236 = None
        permute_143 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg140_1, view_188, permute_143);  arg140_1 = view_188 = permute_143 = None
        view_189 = torch.ops.aten.view.default(addmm_70, [8, 768, 196]);  addmm_70 = None
        permute_144 = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
        add_213 = torch.ops.aten.add.Tensor(add_208, permute_144);  add_208 = permute_144 = None
        clone_167 = torch.ops.aten.clone.default(add_213, memory_format = torch.contiguous_format)
        var_mean_48 = torch.ops.aten.var_mean.correction(clone_167, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_48[0]
        getitem_97 = var_mean_48[1];  var_mean_48 = None
        add_214 = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        sub_48 = torch.ops.aten.sub.Tensor(clone_167, getitem_97);  clone_167 = getitem_97 = None
        mul_237 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
        mul_238 = torch.ops.aten.mul.Tensor(mul_237, arg141_1);  mul_237 = arg141_1 = None
        add_215 = torch.ops.aten.add.Tensor(mul_238, arg142_1);  mul_238 = arg142_1 = None
        view_190 = torch.ops.aten.view.default(add_215, [1568, 768]);  add_215 = None
        permute_145 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg144_1, view_190, permute_145);  arg144_1 = view_190 = permute_145 = None
        view_191 = torch.ops.aten.view.default(addmm_71, [8, 196, 3072]);  addmm_71 = None
        mul_239 = torch.ops.aten.mul.Tensor(view_191, 0.5)
        mul_240 = torch.ops.aten.mul.Tensor(view_191, 0.7071067811865476);  view_191 = None
        erf_47 = torch.ops.aten.erf.default(mul_240);  mul_240 = None
        add_216 = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_239, add_216);  mul_239 = add_216 = None
        view_192 = torch.ops.aten.view.default(mul_241, [1568, 3072]);  mul_241 = None
        permute_146 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg146_1, view_192, permute_146);  arg146_1 = view_192 = permute_146 = None
        view_193 = torch.ops.aten.view.default(addmm_72, [8, 196, 768]);  addmm_72 = None
        add_217 = torch.ops.aten.add.Tensor(add_213, view_193);  add_213 = view_193 = None
        clone_170 = torch.ops.aten.clone.default(add_217, memory_format = torch.contiguous_format);  add_217 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(clone_170, [2], correction = 0, keepdim = True)
        getitem_98 = var_mean_49[0]
        getitem_99 = var_mean_49[1];  var_mean_49 = None
        add_218 = torch.ops.aten.add.Tensor(getitem_98, 1e-06);  getitem_98 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_49 = torch.ops.aten.sub.Tensor(clone_170, getitem_99);  clone_170 = getitem_99 = None
        mul_242 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
        mul_243 = torch.ops.aten.mul.Tensor(mul_242, arg147_1);  mul_242 = arg147_1 = None
        add_219 = torch.ops.aten.add.Tensor(mul_243, arg148_1);  mul_243 = arg148_1 = None
        mean_1 = torch.ops.aten.mean.dim(add_219, [1]);  add_219 = None
        permute_147 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg150_1, mean_1, permute_147);  arg150_1 = mean_1 = permute_147 = None
        return (addmm_73,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 224, 224), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf1, (768, 3, 16, 16), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (768,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf5, (384, 196), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf6, (384,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf7, (196, 384), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf8, (196,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf11, (3072, 768), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf12, (3072,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768, 3072), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf17, (384, 196), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf18, (384,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf19, (196, 384), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf20, (196,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf23, (3072, 768), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf24, (3072,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768, 3072), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf29, (384, 196), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf30, (384,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf31, (196, 384), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf32, (196,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf35, (3072, 768), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf36, (3072,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768, 3072), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf41, (384, 196), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf42, (384,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf43, (196, 384), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf44, (196,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf47, (3072, 768), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf48, (3072,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768, 3072), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf53, (384, 196), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf54, (384,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf55, (196, 384), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf56, (196,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf59, (3072, 768), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf60, (3072,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768, 3072), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf65, (384, 196), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf66, (384,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf67, (196, 384), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf68, (196,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf71, (3072, 768), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf72, (3072,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768, 3072), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf77, (384, 196), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf78, (384,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf79, (196, 384), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf80, (196,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf83, (3072, 768), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf84, (3072,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768, 3072), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf89, (384, 196), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf90, (384,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf91, (196, 384), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf92, (196,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf95, (3072, 768), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf96, (3072,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768, 3072), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf99, (768,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf100, (768,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf101, (384, 196), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf102, (384,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf103, (196, 384), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf104, (196,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf107, (3072, 768), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf108, (3072,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768, 3072), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf113, (384, 196), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf114, (384,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf115, (196, 384), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf116, (196,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf119, (3072, 768), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf120, (3072,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768, 3072), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf125, (384, 196), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf126, (384,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf127, (196, 384), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf128, (196,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf131, (3072, 768), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf132, (3072,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768, 3072), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf137, (384, 196), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf138, (384,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf139, (196, 384), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf140, (196,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf143, (3072, 768), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf144, (3072,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768, 3072), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf146, (768,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 3072000, device=device(type='cuda', index=0))
    reader.tensor(buf149, (1000, 768), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf150, (1000,), is_leaf=True)  # arg150_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)