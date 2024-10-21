
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
        view_73 = torch.ops.aten.view.default(convolution_1, [8, 384, 196]);  convolution_1 = None
        permute_62 = torch.ops.aten.permute.default(view_73, [0, 2, 1]);  view_73 = None
        mul_110 = torch.ops.aten.mul.Tensor(arg5_1, 1);  arg5_1 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_110, permute_62);  mul_110 = None
        add_73 = torch.ops.aten.add.Tensor(arg4_1, mul_111);  arg4_1 = mul_111 = None
        permute_63 = torch.ops.aten.permute.default(add_73, [0, 2, 1]);  add_73 = None
        view_74 = torch.ops.aten.view.default(permute_63, [3072, 196]);  permute_63 = None
        permute_64 = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg7_1, view_74, permute_64);  arg7_1 = view_74 = permute_64 = None
        view_75 = torch.ops.aten.view.default(addmm_25, [8, 384, 196]);  addmm_25 = None
        permute_65 = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        mul_112 = torch.ops.aten.mul.Tensor(arg3_1, permute_65);  arg3_1 = permute_65 = None
        add_74 = torch.ops.aten.add.Tensor(permute_62, mul_112);  permute_62 = mul_112 = None
        mul_113 = torch.ops.aten.mul.Tensor(arg10_1, 1);  arg10_1 = None
        mul_114 = torch.ops.aten.mul.Tensor(mul_113, add_74);  mul_113 = None
        add_75 = torch.ops.aten.add.Tensor(arg9_1, mul_114);  arg9_1 = mul_114 = None
        permute_66 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        clone_37 = torch.ops.aten.clone.default(add_75, memory_format = torch.contiguous_format);  add_75 = None
        view_76 = torch.ops.aten.view.default(clone_37, [1568, 384]);  clone_37 = None
        mm_12 = torch.ops.aten.mm.default(view_76, permute_66);  view_76 = permute_66 = None
        view_77 = torch.ops.aten.view.default(mm_12, [8, 196, 1536]);  mm_12 = None
        add_76 = torch.ops.aten.add.Tensor(view_77, arg12_1);  view_77 = arg12_1 = None
        mul_115 = torch.ops.aten.mul.Tensor(add_76, 0.5)
        mul_116 = torch.ops.aten.mul.Tensor(add_76, 0.7071067811865476);  add_76 = None
        erf_12 = torch.ops.aten.erf.default(mul_116);  mul_116 = None
        add_77 = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_115, add_77);  mul_115 = add_77 = None
        view_78 = torch.ops.aten.view.default(mul_117, [1568, 1536]);  mul_117 = None
        permute_67 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg14_1, view_78, permute_67);  arg14_1 = view_78 = permute_67 = None
        view_79 = torch.ops.aten.view.default(addmm_26, [8, 196, 384]);  addmm_26 = None
        mul_118 = torch.ops.aten.mul.Tensor(arg8_1, view_79);  arg8_1 = view_79 = None
        add_78 = torch.ops.aten.add.Tensor(add_74, mul_118);  add_74 = mul_118 = None
        mul_119 = torch.ops.aten.mul.Tensor(arg17_1, 1);  arg17_1 = None
        mul_120 = torch.ops.aten.mul.Tensor(mul_119, add_78);  mul_119 = None
        add_79 = torch.ops.aten.add.Tensor(arg16_1, mul_120);  arg16_1 = mul_120 = None
        permute_68 = torch.ops.aten.permute.default(add_79, [0, 2, 1]);  add_79 = None
        view_80 = torch.ops.aten.view.default(permute_68, [3072, 196]);  permute_68 = None
        permute_69 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg19_1, view_80, permute_69);  arg19_1 = view_80 = permute_69 = None
        view_81 = torch.ops.aten.view.default(addmm_27, [8, 384, 196]);  addmm_27 = None
        permute_70 = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
        mul_121 = torch.ops.aten.mul.Tensor(arg15_1, permute_70);  arg15_1 = permute_70 = None
        add_80 = torch.ops.aten.add.Tensor(add_78, mul_121);  add_78 = mul_121 = None
        mul_122 = torch.ops.aten.mul.Tensor(arg22_1, 1);  arg22_1 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_122, add_80);  mul_122 = None
        add_81 = torch.ops.aten.add.Tensor(arg21_1, mul_123);  arg21_1 = mul_123 = None
        permute_71 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        clone_40 = torch.ops.aten.clone.default(add_81, memory_format = torch.contiguous_format);  add_81 = None
        view_82 = torch.ops.aten.view.default(clone_40, [1568, 384]);  clone_40 = None
        mm_13 = torch.ops.aten.mm.default(view_82, permute_71);  view_82 = permute_71 = None
        view_83 = torch.ops.aten.view.default(mm_13, [8, 196, 1536]);  mm_13 = None
        add_82 = torch.ops.aten.add.Tensor(view_83, arg24_1);  view_83 = arg24_1 = None
        mul_124 = torch.ops.aten.mul.Tensor(add_82, 0.5)
        mul_125 = torch.ops.aten.mul.Tensor(add_82, 0.7071067811865476);  add_82 = None
        erf_13 = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_83 = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_126 = torch.ops.aten.mul.Tensor(mul_124, add_83);  mul_124 = add_83 = None
        view_84 = torch.ops.aten.view.default(mul_126, [1568, 1536]);  mul_126 = None
        permute_72 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg26_1, view_84, permute_72);  arg26_1 = view_84 = permute_72 = None
        view_85 = torch.ops.aten.view.default(addmm_28, [8, 196, 384]);  addmm_28 = None
        mul_127 = torch.ops.aten.mul.Tensor(arg20_1, view_85);  arg20_1 = view_85 = None
        add_84 = torch.ops.aten.add.Tensor(add_80, mul_127);  add_80 = mul_127 = None
        mul_128 = torch.ops.aten.mul.Tensor(arg29_1, 1);  arg29_1 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_128, add_84);  mul_128 = None
        add_85 = torch.ops.aten.add.Tensor(arg28_1, mul_129);  arg28_1 = mul_129 = None
        permute_73 = torch.ops.aten.permute.default(add_85, [0, 2, 1]);  add_85 = None
        view_86 = torch.ops.aten.view.default(permute_73, [3072, 196]);  permute_73 = None
        permute_74 = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg31_1, view_86, permute_74);  arg31_1 = view_86 = permute_74 = None
        view_87 = torch.ops.aten.view.default(addmm_29, [8, 384, 196]);  addmm_29 = None
        permute_75 = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
        mul_130 = torch.ops.aten.mul.Tensor(arg27_1, permute_75);  arg27_1 = permute_75 = None
        add_86 = torch.ops.aten.add.Tensor(add_84, mul_130);  add_84 = mul_130 = None
        mul_131 = torch.ops.aten.mul.Tensor(arg34_1, 1);  arg34_1 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_131, add_86);  mul_131 = None
        add_87 = torch.ops.aten.add.Tensor(arg33_1, mul_132);  arg33_1 = mul_132 = None
        permute_76 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        clone_43 = torch.ops.aten.clone.default(add_87, memory_format = torch.contiguous_format);  add_87 = None
        view_88 = torch.ops.aten.view.default(clone_43, [1568, 384]);  clone_43 = None
        mm_14 = torch.ops.aten.mm.default(view_88, permute_76);  view_88 = permute_76 = None
        view_89 = torch.ops.aten.view.default(mm_14, [8, 196, 1536]);  mm_14 = None
        add_88 = torch.ops.aten.add.Tensor(view_89, arg36_1);  view_89 = arg36_1 = None
        mul_133 = torch.ops.aten.mul.Tensor(add_88, 0.5)
        mul_134 = torch.ops.aten.mul.Tensor(add_88, 0.7071067811865476);  add_88 = None
        erf_14 = torch.ops.aten.erf.default(mul_134);  mul_134 = None
        add_89 = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_135 = torch.ops.aten.mul.Tensor(mul_133, add_89);  mul_133 = add_89 = None
        view_90 = torch.ops.aten.view.default(mul_135, [1568, 1536]);  mul_135 = None
        permute_77 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg38_1, view_90, permute_77);  arg38_1 = view_90 = permute_77 = None
        view_91 = torch.ops.aten.view.default(addmm_30, [8, 196, 384]);  addmm_30 = None
        mul_136 = torch.ops.aten.mul.Tensor(arg32_1, view_91);  arg32_1 = view_91 = None
        add_90 = torch.ops.aten.add.Tensor(add_86, mul_136);  add_86 = mul_136 = None
        mul_137 = torch.ops.aten.mul.Tensor(arg41_1, 1);  arg41_1 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_137, add_90);  mul_137 = None
        add_91 = torch.ops.aten.add.Tensor(arg40_1, mul_138);  arg40_1 = mul_138 = None
        permute_78 = torch.ops.aten.permute.default(add_91, [0, 2, 1]);  add_91 = None
        view_92 = torch.ops.aten.view.default(permute_78, [3072, 196]);  permute_78 = None
        permute_79 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg43_1, view_92, permute_79);  arg43_1 = view_92 = permute_79 = None
        view_93 = torch.ops.aten.view.default(addmm_31, [8, 384, 196]);  addmm_31 = None
        permute_80 = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None
        mul_139 = torch.ops.aten.mul.Tensor(arg39_1, permute_80);  arg39_1 = permute_80 = None
        add_92 = torch.ops.aten.add.Tensor(add_90, mul_139);  add_90 = mul_139 = None
        mul_140 = torch.ops.aten.mul.Tensor(arg46_1, 1);  arg46_1 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_140, add_92);  mul_140 = None
        add_93 = torch.ops.aten.add.Tensor(arg45_1, mul_141);  arg45_1 = mul_141 = None
        permute_81 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        clone_46 = torch.ops.aten.clone.default(add_93, memory_format = torch.contiguous_format);  add_93 = None
        view_94 = torch.ops.aten.view.default(clone_46, [1568, 384]);  clone_46 = None
        mm_15 = torch.ops.aten.mm.default(view_94, permute_81);  view_94 = permute_81 = None
        view_95 = torch.ops.aten.view.default(mm_15, [8, 196, 1536]);  mm_15 = None
        add_94 = torch.ops.aten.add.Tensor(view_95, arg48_1);  view_95 = arg48_1 = None
        mul_142 = torch.ops.aten.mul.Tensor(add_94, 0.5)
        mul_143 = torch.ops.aten.mul.Tensor(add_94, 0.7071067811865476);  add_94 = None
        erf_15 = torch.ops.aten.erf.default(mul_143);  mul_143 = None
        add_95 = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_144 = torch.ops.aten.mul.Tensor(mul_142, add_95);  mul_142 = add_95 = None
        view_96 = torch.ops.aten.view.default(mul_144, [1568, 1536]);  mul_144 = None
        permute_82 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg50_1, view_96, permute_82);  arg50_1 = view_96 = permute_82 = None
        view_97 = torch.ops.aten.view.default(addmm_32, [8, 196, 384]);  addmm_32 = None
        mul_145 = torch.ops.aten.mul.Tensor(arg44_1, view_97);  arg44_1 = view_97 = None
        add_96 = torch.ops.aten.add.Tensor(add_92, mul_145);  add_92 = mul_145 = None
        mul_146 = torch.ops.aten.mul.Tensor(arg53_1, 1);  arg53_1 = None
        mul_147 = torch.ops.aten.mul.Tensor(mul_146, add_96);  mul_146 = None
        add_97 = torch.ops.aten.add.Tensor(arg52_1, mul_147);  arg52_1 = mul_147 = None
        permute_83 = torch.ops.aten.permute.default(add_97, [0, 2, 1]);  add_97 = None
        view_98 = torch.ops.aten.view.default(permute_83, [3072, 196]);  permute_83 = None
        permute_84 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg55_1, view_98, permute_84);  arg55_1 = view_98 = permute_84 = None
        view_99 = torch.ops.aten.view.default(addmm_33, [8, 384, 196]);  addmm_33 = None
        permute_85 = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        mul_148 = torch.ops.aten.mul.Tensor(arg51_1, permute_85);  arg51_1 = permute_85 = None
        add_98 = torch.ops.aten.add.Tensor(add_96, mul_148);  add_96 = mul_148 = None
        mul_149 = torch.ops.aten.mul.Tensor(arg58_1, 1);  arg58_1 = None
        mul_150 = torch.ops.aten.mul.Tensor(mul_149, add_98);  mul_149 = None
        add_99 = torch.ops.aten.add.Tensor(arg57_1, mul_150);  arg57_1 = mul_150 = None
        permute_86 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        clone_49 = torch.ops.aten.clone.default(add_99, memory_format = torch.contiguous_format);  add_99 = None
        view_100 = torch.ops.aten.view.default(clone_49, [1568, 384]);  clone_49 = None
        mm_16 = torch.ops.aten.mm.default(view_100, permute_86);  view_100 = permute_86 = None
        view_101 = torch.ops.aten.view.default(mm_16, [8, 196, 1536]);  mm_16 = None
        add_100 = torch.ops.aten.add.Tensor(view_101, arg60_1);  view_101 = arg60_1 = None
        mul_151 = torch.ops.aten.mul.Tensor(add_100, 0.5)
        mul_152 = torch.ops.aten.mul.Tensor(add_100, 0.7071067811865476);  add_100 = None
        erf_16 = torch.ops.aten.erf.default(mul_152);  mul_152 = None
        add_101 = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_151, add_101);  mul_151 = add_101 = None
        view_102 = torch.ops.aten.view.default(mul_153, [1568, 1536]);  mul_153 = None
        permute_87 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg62_1, view_102, permute_87);  arg62_1 = view_102 = permute_87 = None
        view_103 = torch.ops.aten.view.default(addmm_34, [8, 196, 384]);  addmm_34 = None
        mul_154 = torch.ops.aten.mul.Tensor(arg56_1, view_103);  arg56_1 = view_103 = None
        add_102 = torch.ops.aten.add.Tensor(add_98, mul_154);  add_98 = mul_154 = None
        mul_155 = torch.ops.aten.mul.Tensor(arg65_1, 1);  arg65_1 = None
        mul_156 = torch.ops.aten.mul.Tensor(mul_155, add_102);  mul_155 = None
        add_103 = torch.ops.aten.add.Tensor(arg64_1, mul_156);  arg64_1 = mul_156 = None
        permute_88 = torch.ops.aten.permute.default(add_103, [0, 2, 1]);  add_103 = None
        view_104 = torch.ops.aten.view.default(permute_88, [3072, 196]);  permute_88 = None
        permute_89 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg67_1, view_104, permute_89);  arg67_1 = view_104 = permute_89 = None
        view_105 = torch.ops.aten.view.default(addmm_35, [8, 384, 196]);  addmm_35 = None
        permute_90 = torch.ops.aten.permute.default(view_105, [0, 2, 1]);  view_105 = None
        mul_157 = torch.ops.aten.mul.Tensor(arg63_1, permute_90);  arg63_1 = permute_90 = None
        add_104 = torch.ops.aten.add.Tensor(add_102, mul_157);  add_102 = mul_157 = None
        mul_158 = torch.ops.aten.mul.Tensor(arg70_1, 1);  arg70_1 = None
        mul_159 = torch.ops.aten.mul.Tensor(mul_158, add_104);  mul_158 = None
        add_105 = torch.ops.aten.add.Tensor(arg69_1, mul_159);  arg69_1 = mul_159 = None
        permute_91 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        clone_52 = torch.ops.aten.clone.default(add_105, memory_format = torch.contiguous_format);  add_105 = None
        view_106 = torch.ops.aten.view.default(clone_52, [1568, 384]);  clone_52 = None
        mm_17 = torch.ops.aten.mm.default(view_106, permute_91);  view_106 = permute_91 = None
        view_107 = torch.ops.aten.view.default(mm_17, [8, 196, 1536]);  mm_17 = None
        add_106 = torch.ops.aten.add.Tensor(view_107, arg72_1);  view_107 = arg72_1 = None
        mul_160 = torch.ops.aten.mul.Tensor(add_106, 0.5)
        mul_161 = torch.ops.aten.mul.Tensor(add_106, 0.7071067811865476);  add_106 = None
        erf_17 = torch.ops.aten.erf.default(mul_161);  mul_161 = None
        add_107 = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_162 = torch.ops.aten.mul.Tensor(mul_160, add_107);  mul_160 = add_107 = None
        view_108 = torch.ops.aten.view.default(mul_162, [1568, 1536]);  mul_162 = None
        permute_92 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg74_1, view_108, permute_92);  arg74_1 = view_108 = permute_92 = None
        view_109 = torch.ops.aten.view.default(addmm_36, [8, 196, 384]);  addmm_36 = None
        mul_163 = torch.ops.aten.mul.Tensor(arg68_1, view_109);  arg68_1 = view_109 = None
        add_108 = torch.ops.aten.add.Tensor(add_104, mul_163);  add_104 = mul_163 = None
        mul_164 = torch.ops.aten.mul.Tensor(arg77_1, 1);  arg77_1 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, add_108);  mul_164 = None
        add_109 = torch.ops.aten.add.Tensor(arg76_1, mul_165);  arg76_1 = mul_165 = None
        permute_93 = torch.ops.aten.permute.default(add_109, [0, 2, 1]);  add_109 = None
        view_110 = torch.ops.aten.view.default(permute_93, [3072, 196]);  permute_93 = None
        permute_94 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg79_1, view_110, permute_94);  arg79_1 = view_110 = permute_94 = None
        view_111 = torch.ops.aten.view.default(addmm_37, [8, 384, 196]);  addmm_37 = None
        permute_95 = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
        mul_166 = torch.ops.aten.mul.Tensor(arg75_1, permute_95);  arg75_1 = permute_95 = None
        add_110 = torch.ops.aten.add.Tensor(add_108, mul_166);  add_108 = mul_166 = None
        mul_167 = torch.ops.aten.mul.Tensor(arg82_1, 1);  arg82_1 = None
        mul_168 = torch.ops.aten.mul.Tensor(mul_167, add_110);  mul_167 = None
        add_111 = torch.ops.aten.add.Tensor(arg81_1, mul_168);  arg81_1 = mul_168 = None
        permute_96 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        clone_55 = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format);  add_111 = None
        view_112 = torch.ops.aten.view.default(clone_55, [1568, 384]);  clone_55 = None
        mm_18 = torch.ops.aten.mm.default(view_112, permute_96);  view_112 = permute_96 = None
        view_113 = torch.ops.aten.view.default(mm_18, [8, 196, 1536]);  mm_18 = None
        add_112 = torch.ops.aten.add.Tensor(view_113, arg84_1);  view_113 = arg84_1 = None
        mul_169 = torch.ops.aten.mul.Tensor(add_112, 0.5)
        mul_170 = torch.ops.aten.mul.Tensor(add_112, 0.7071067811865476);  add_112 = None
        erf_18 = torch.ops.aten.erf.default(mul_170);  mul_170 = None
        add_113 = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_171 = torch.ops.aten.mul.Tensor(mul_169, add_113);  mul_169 = add_113 = None
        view_114 = torch.ops.aten.view.default(mul_171, [1568, 1536]);  mul_171 = None
        permute_97 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg86_1, view_114, permute_97);  arg86_1 = view_114 = permute_97 = None
        view_115 = torch.ops.aten.view.default(addmm_38, [8, 196, 384]);  addmm_38 = None
        mul_172 = torch.ops.aten.mul.Tensor(arg80_1, view_115);  arg80_1 = view_115 = None
        add_114 = torch.ops.aten.add.Tensor(add_110, mul_172);  add_110 = mul_172 = None
        mul_173 = torch.ops.aten.mul.Tensor(arg89_1, 1);  arg89_1 = None
        mul_174 = torch.ops.aten.mul.Tensor(mul_173, add_114);  mul_173 = None
        add_115 = torch.ops.aten.add.Tensor(arg88_1, mul_174);  arg88_1 = mul_174 = None
        permute_98 = torch.ops.aten.permute.default(add_115, [0, 2, 1]);  add_115 = None
        view_116 = torch.ops.aten.view.default(permute_98, [3072, 196]);  permute_98 = None
        permute_99 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg91_1, view_116, permute_99);  arg91_1 = view_116 = permute_99 = None
        view_117 = torch.ops.aten.view.default(addmm_39, [8, 384, 196]);  addmm_39 = None
        permute_100 = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
        mul_175 = torch.ops.aten.mul.Tensor(arg87_1, permute_100);  arg87_1 = permute_100 = None
        add_116 = torch.ops.aten.add.Tensor(add_114, mul_175);  add_114 = mul_175 = None
        mul_176 = torch.ops.aten.mul.Tensor(arg94_1, 1);  arg94_1 = None
        mul_177 = torch.ops.aten.mul.Tensor(mul_176, add_116);  mul_176 = None
        add_117 = torch.ops.aten.add.Tensor(arg93_1, mul_177);  arg93_1 = mul_177 = None
        permute_101 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        clone_58 = torch.ops.aten.clone.default(add_117, memory_format = torch.contiguous_format);  add_117 = None
        view_118 = torch.ops.aten.view.default(clone_58, [1568, 384]);  clone_58 = None
        mm_19 = torch.ops.aten.mm.default(view_118, permute_101);  view_118 = permute_101 = None
        view_119 = torch.ops.aten.view.default(mm_19, [8, 196, 1536]);  mm_19 = None
        add_118 = torch.ops.aten.add.Tensor(view_119, arg96_1);  view_119 = arg96_1 = None
        mul_178 = torch.ops.aten.mul.Tensor(add_118, 0.5)
        mul_179 = torch.ops.aten.mul.Tensor(add_118, 0.7071067811865476);  add_118 = None
        erf_19 = torch.ops.aten.erf.default(mul_179);  mul_179 = None
        add_119 = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_180 = torch.ops.aten.mul.Tensor(mul_178, add_119);  mul_178 = add_119 = None
        view_120 = torch.ops.aten.view.default(mul_180, [1568, 1536]);  mul_180 = None
        permute_102 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg98_1, view_120, permute_102);  arg98_1 = view_120 = permute_102 = None
        view_121 = torch.ops.aten.view.default(addmm_40, [8, 196, 384]);  addmm_40 = None
        mul_181 = torch.ops.aten.mul.Tensor(arg92_1, view_121);  arg92_1 = view_121 = None
        add_120 = torch.ops.aten.add.Tensor(add_116, mul_181);  add_116 = mul_181 = None
        mul_182 = torch.ops.aten.mul.Tensor(arg101_1, 1);  arg101_1 = None
        mul_183 = torch.ops.aten.mul.Tensor(mul_182, add_120);  mul_182 = None
        add_121 = torch.ops.aten.add.Tensor(arg100_1, mul_183);  arg100_1 = mul_183 = None
        permute_103 = torch.ops.aten.permute.default(add_121, [0, 2, 1]);  add_121 = None
        view_122 = torch.ops.aten.view.default(permute_103, [3072, 196]);  permute_103 = None
        permute_104 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg103_1, view_122, permute_104);  arg103_1 = view_122 = permute_104 = None
        view_123 = torch.ops.aten.view.default(addmm_41, [8, 384, 196]);  addmm_41 = None
        permute_105 = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        mul_184 = torch.ops.aten.mul.Tensor(arg99_1, permute_105);  arg99_1 = permute_105 = None
        add_122 = torch.ops.aten.add.Tensor(add_120, mul_184);  add_120 = mul_184 = None
        mul_185 = torch.ops.aten.mul.Tensor(arg106_1, 1);  arg106_1 = None
        mul_186 = torch.ops.aten.mul.Tensor(mul_185, add_122);  mul_185 = None
        add_123 = torch.ops.aten.add.Tensor(arg105_1, mul_186);  arg105_1 = mul_186 = None
        permute_106 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        clone_61 = torch.ops.aten.clone.default(add_123, memory_format = torch.contiguous_format);  add_123 = None
        view_124 = torch.ops.aten.view.default(clone_61, [1568, 384]);  clone_61 = None
        mm_20 = torch.ops.aten.mm.default(view_124, permute_106);  view_124 = permute_106 = None
        view_125 = torch.ops.aten.view.default(mm_20, [8, 196, 1536]);  mm_20 = None
        add_124 = torch.ops.aten.add.Tensor(view_125, arg108_1);  view_125 = arg108_1 = None
        mul_187 = torch.ops.aten.mul.Tensor(add_124, 0.5)
        mul_188 = torch.ops.aten.mul.Tensor(add_124, 0.7071067811865476);  add_124 = None
        erf_20 = torch.ops.aten.erf.default(mul_188);  mul_188 = None
        add_125 = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_189 = torch.ops.aten.mul.Tensor(mul_187, add_125);  mul_187 = add_125 = None
        view_126 = torch.ops.aten.view.default(mul_189, [1568, 1536]);  mul_189 = None
        permute_107 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg110_1, view_126, permute_107);  arg110_1 = view_126 = permute_107 = None
        view_127 = torch.ops.aten.view.default(addmm_42, [8, 196, 384]);  addmm_42 = None
        mul_190 = torch.ops.aten.mul.Tensor(arg104_1, view_127);  arg104_1 = view_127 = None
        add_126 = torch.ops.aten.add.Tensor(add_122, mul_190);  add_122 = mul_190 = None
        mul_191 = torch.ops.aten.mul.Tensor(arg113_1, 1);  arg113_1 = None
        mul_192 = torch.ops.aten.mul.Tensor(mul_191, add_126);  mul_191 = None
        add_127 = torch.ops.aten.add.Tensor(arg112_1, mul_192);  arg112_1 = mul_192 = None
        permute_108 = torch.ops.aten.permute.default(add_127, [0, 2, 1]);  add_127 = None
        view_128 = torch.ops.aten.view.default(permute_108, [3072, 196]);  permute_108 = None
        permute_109 = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg115_1, view_128, permute_109);  arg115_1 = view_128 = permute_109 = None
        view_129 = torch.ops.aten.view.default(addmm_43, [8, 384, 196]);  addmm_43 = None
        permute_110 = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None
        mul_193 = torch.ops.aten.mul.Tensor(arg111_1, permute_110);  arg111_1 = permute_110 = None
        add_128 = torch.ops.aten.add.Tensor(add_126, mul_193);  add_126 = mul_193 = None
        mul_194 = torch.ops.aten.mul.Tensor(arg118_1, 1);  arg118_1 = None
        mul_195 = torch.ops.aten.mul.Tensor(mul_194, add_128);  mul_194 = None
        add_129 = torch.ops.aten.add.Tensor(arg117_1, mul_195);  arg117_1 = mul_195 = None
        permute_111 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        clone_64 = torch.ops.aten.clone.default(add_129, memory_format = torch.contiguous_format);  add_129 = None
        view_130 = torch.ops.aten.view.default(clone_64, [1568, 384]);  clone_64 = None
        mm_21 = torch.ops.aten.mm.default(view_130, permute_111);  view_130 = permute_111 = None
        view_131 = torch.ops.aten.view.default(mm_21, [8, 196, 1536]);  mm_21 = None
        add_130 = torch.ops.aten.add.Tensor(view_131, arg120_1);  view_131 = arg120_1 = None
        mul_196 = torch.ops.aten.mul.Tensor(add_130, 0.5)
        mul_197 = torch.ops.aten.mul.Tensor(add_130, 0.7071067811865476);  add_130 = None
        erf_21 = torch.ops.aten.erf.default(mul_197);  mul_197 = None
        add_131 = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_198 = torch.ops.aten.mul.Tensor(mul_196, add_131);  mul_196 = add_131 = None
        view_132 = torch.ops.aten.view.default(mul_198, [1568, 1536]);  mul_198 = None
        permute_112 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg122_1, view_132, permute_112);  arg122_1 = view_132 = permute_112 = None
        view_133 = torch.ops.aten.view.default(addmm_44, [8, 196, 384]);  addmm_44 = None
        mul_199 = torch.ops.aten.mul.Tensor(arg116_1, view_133);  arg116_1 = view_133 = None
        add_132 = torch.ops.aten.add.Tensor(add_128, mul_199);  add_128 = mul_199 = None
        mul_200 = torch.ops.aten.mul.Tensor(arg125_1, 1);  arg125_1 = None
        mul_201 = torch.ops.aten.mul.Tensor(mul_200, add_132);  mul_200 = None
        add_133 = torch.ops.aten.add.Tensor(arg124_1, mul_201);  arg124_1 = mul_201 = None
        permute_113 = torch.ops.aten.permute.default(add_133, [0, 2, 1]);  add_133 = None
        view_134 = torch.ops.aten.view.default(permute_113, [3072, 196]);  permute_113 = None
        permute_114 = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg127_1, view_134, permute_114);  arg127_1 = view_134 = permute_114 = None
        view_135 = torch.ops.aten.view.default(addmm_45, [8, 384, 196]);  addmm_45 = None
        permute_115 = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
        mul_202 = torch.ops.aten.mul.Tensor(arg123_1, permute_115);  arg123_1 = permute_115 = None
        add_134 = torch.ops.aten.add.Tensor(add_132, mul_202);  add_132 = mul_202 = None
        mul_203 = torch.ops.aten.mul.Tensor(arg130_1, 1);  arg130_1 = None
        mul_204 = torch.ops.aten.mul.Tensor(mul_203, add_134);  mul_203 = None
        add_135 = torch.ops.aten.add.Tensor(arg129_1, mul_204);  arg129_1 = mul_204 = None
        permute_116 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        clone_67 = torch.ops.aten.clone.default(add_135, memory_format = torch.contiguous_format);  add_135 = None
        view_136 = torch.ops.aten.view.default(clone_67, [1568, 384]);  clone_67 = None
        mm_22 = torch.ops.aten.mm.default(view_136, permute_116);  view_136 = permute_116 = None
        view_137 = torch.ops.aten.view.default(mm_22, [8, 196, 1536]);  mm_22 = None
        add_136 = torch.ops.aten.add.Tensor(view_137, arg132_1);  view_137 = arg132_1 = None
        mul_205 = torch.ops.aten.mul.Tensor(add_136, 0.5)
        mul_206 = torch.ops.aten.mul.Tensor(add_136, 0.7071067811865476);  add_136 = None
        erf_22 = torch.ops.aten.erf.default(mul_206);  mul_206 = None
        add_137 = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_207 = torch.ops.aten.mul.Tensor(mul_205, add_137);  mul_205 = add_137 = None
        view_138 = torch.ops.aten.view.default(mul_207, [1568, 1536]);  mul_207 = None
        permute_117 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg134_1, view_138, permute_117);  arg134_1 = view_138 = permute_117 = None
        view_139 = torch.ops.aten.view.default(addmm_46, [8, 196, 384]);  addmm_46 = None
        mul_208 = torch.ops.aten.mul.Tensor(arg128_1, view_139);  arg128_1 = view_139 = None
        add_138 = torch.ops.aten.add.Tensor(add_134, mul_208);  add_134 = mul_208 = None
        mul_209 = torch.ops.aten.mul.Tensor(arg137_1, 1);  arg137_1 = None
        mul_210 = torch.ops.aten.mul.Tensor(mul_209, add_138);  mul_209 = None
        add_139 = torch.ops.aten.add.Tensor(arg136_1, mul_210);  arg136_1 = mul_210 = None
        permute_118 = torch.ops.aten.permute.default(add_139, [0, 2, 1]);  add_139 = None
        view_140 = torch.ops.aten.view.default(permute_118, [3072, 196]);  permute_118 = None
        permute_119 = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg139_1, view_140, permute_119);  arg139_1 = view_140 = permute_119 = None
        view_141 = torch.ops.aten.view.default(addmm_47, [8, 384, 196]);  addmm_47 = None
        permute_120 = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        mul_211 = torch.ops.aten.mul.Tensor(arg135_1, permute_120);  arg135_1 = permute_120 = None
        add_140 = torch.ops.aten.add.Tensor(add_138, mul_211);  add_138 = mul_211 = None
        mul_212 = torch.ops.aten.mul.Tensor(arg142_1, 1);  arg142_1 = None
        mul_213 = torch.ops.aten.mul.Tensor(mul_212, add_140);  mul_212 = None
        add_141 = torch.ops.aten.add.Tensor(arg141_1, mul_213);  arg141_1 = mul_213 = None
        permute_121 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        clone_70 = torch.ops.aten.clone.default(add_141, memory_format = torch.contiguous_format);  add_141 = None
        view_142 = torch.ops.aten.view.default(clone_70, [1568, 384]);  clone_70 = None
        mm_23 = torch.ops.aten.mm.default(view_142, permute_121);  view_142 = permute_121 = None
        view_143 = torch.ops.aten.view.default(mm_23, [8, 196, 1536]);  mm_23 = None
        add_142 = torch.ops.aten.add.Tensor(view_143, arg144_1);  view_143 = arg144_1 = None
        mul_214 = torch.ops.aten.mul.Tensor(add_142, 0.5)
        mul_215 = torch.ops.aten.mul.Tensor(add_142, 0.7071067811865476);  add_142 = None
        erf_23 = torch.ops.aten.erf.default(mul_215);  mul_215 = None
        add_143 = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_216 = torch.ops.aten.mul.Tensor(mul_214, add_143);  mul_214 = add_143 = None
        view_144 = torch.ops.aten.view.default(mul_216, [1568, 1536]);  mul_216 = None
        permute_122 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg146_1, view_144, permute_122);  arg146_1 = view_144 = permute_122 = None
        view_145 = torch.ops.aten.view.default(addmm_48, [8, 196, 384]);  addmm_48 = None
        mul_217 = torch.ops.aten.mul.Tensor(arg140_1, view_145);  arg140_1 = view_145 = None
        add_144 = torch.ops.aten.add.Tensor(add_140, mul_217);  add_140 = mul_217 = None
        mul_218 = torch.ops.aten.mul.Tensor(arg148_1, 1);  arg148_1 = None
        mul_219 = torch.ops.aten.mul.Tensor(mul_218, add_144);  mul_218 = add_144 = None
        add_145 = torch.ops.aten.add.Tensor(arg147_1, mul_219);  arg147_1 = mul_219 = None
        mean_1 = torch.ops.aten.mean.dim(add_145, [1]);  add_145 = None
        permute_123 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg150_1, mean_1, permute_123);  arg150_1 = mean_1 = permute_123 = None
        return (addmm_49,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 224, 224), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf1, (384, 3, 16, 16), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf2, (384,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf3, (384,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1, 1, 384), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf5, (1, 1, 384), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf6, (196, 196), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf7, (196,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf8, (384,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1, 1, 384), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1, 1, 384), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf11, (1536, 384), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf12, (1536,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf13, (384, 1536), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf14, (384,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf15, (384,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf16, (1, 1, 384), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf17, (1, 1, 384), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf18, (196, 196), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf19, (196,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf20, (384,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf21, (1, 1, 384), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf22, (1, 1, 384), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf23, (1536, 384), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf24, (1536,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf25, (384, 1536), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf26, (384,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf27, (384,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf28, (1, 1, 384), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf29, (1, 1, 384), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf30, (196, 196), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf31, (196,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf32, (384,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf33, (1, 1, 384), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf34, (1, 1, 384), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf35, (1536, 384), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1536,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf37, (384, 1536), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf38, (384,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf39, (384,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf40, (1, 1, 384), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf41, (1, 1, 384), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf42, (196, 196), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf43, (196,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf44, (384,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf45, (1, 1, 384), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf46, (1, 1, 384), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf47, (1536, 384), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf48, (1536,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf49, (384, 1536), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf50, (384,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf51, (384,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf52, (1, 1, 384), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf53, (1, 1, 384), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf54, (196, 196), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf55, (196,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf56, (384,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf57, (1, 1, 384), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf58, (1, 1, 384), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf59, (1536, 384), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf60, (1536,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf61, (384, 1536), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf62, (384,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf63, (384,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf64, (1, 1, 384), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf65, (1, 1, 384), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf66, (196, 196), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf67, (196,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf68, (384,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf69, (1, 1, 384), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf70, (1, 1, 384), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf71, (1536, 384), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf72, (1536,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf73, (384, 1536), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf74, (384,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf75, (384,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf76, (1, 1, 384), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf77, (1, 1, 384), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf78, (196, 196), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf79, (196,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf80, (384,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf81, (1, 1, 384), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf82, (1, 1, 384), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf83, (1536, 384), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf84, (1536,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf85, (384, 1536), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf86, (384,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf87, (384,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf88, (1, 1, 384), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf89, (1, 1, 384), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf90, (196, 196), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf91, (196,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf92, (384,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf93, (1, 1, 384), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf94, (1, 1, 384), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf95, (1536, 384), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf96, (1536,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf97, (384, 1536), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf98, (384,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf99, (384,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1, 1, 384), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf101, (1, 1, 384), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf102, (196, 196), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf103, (196,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf104, (384,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf105, (1, 1, 384), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf106, (1, 1, 384), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf107, (1536, 384), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf108, (1536,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf109, (384, 1536), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf110, (384,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf111, (384,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf112, (1, 1, 384), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf113, (1, 1, 384), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf114, (196, 196), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf115, (196,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf116, (384,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf117, (1, 1, 384), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf118, (1, 1, 384), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf119, (1536, 384), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf120, (1536,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf121, (384, 1536), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf122, (384,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf123, (384,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf124, (1, 1, 384), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf125, (1, 1, 384), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf126, (196, 196), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf127, (196,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf128, (384,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf129, (1, 1, 384), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf130, (1, 1, 384), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf131, (1536, 384), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1536,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf133, (384, 1536), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf134, (384,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf135, (384,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf136, (1, 1, 384), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf137, (1, 1, 384), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 153664, device=device(type='cuda', index=0))
    reader.tensor(buf138, (196, 196), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 784, device=device(type='cuda', index=0))
    reader.tensor(buf139, (196,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf140, (384,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf141, (1, 1, 384), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf142, (1, 1, 384), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf143, (1536, 384), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf144, (1536,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf145, (384, 1536), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf146, (384,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf147, (1, 1, 384), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf148, (1, 1, 384), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 1536000, device=device(type='cuda', index=0))
    reader.tensor(buf149, (1000, 384), is_leaf=True)  # arg149_1
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