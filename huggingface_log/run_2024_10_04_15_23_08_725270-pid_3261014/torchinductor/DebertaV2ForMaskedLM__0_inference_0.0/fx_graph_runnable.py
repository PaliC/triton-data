
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

torch._inductor.config.triton.cudagraphs = True
torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.5.0a0+git5380214
# torch cuda version: 12.1
# torch git version: 5380214107813f63c7c59f477487d5447085b45a


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Fri_Jan__6_16:45:21_PST_2023 
# Cuda compilation tools, release 12.0, V12.0.140 
# Build cuda_12.0.r12.0/compiler.32267302_0 

# GPU Hardware Info: 
# NVIDIA H100 : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('_tensor_constant0', tensor(64.))
        self.register_buffer('_tensor_constant1', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant2', tensor(64.))
        self.register_buffer('_tensor_constant3', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant4', tensor(64.))
        self.register_buffer('_tensor_constant5', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant6', tensor(64.))
        self.register_buffer('_tensor_constant7', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant8', tensor(64.))
        self.register_buffer('_tensor_constant9', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant10', tensor(64.))
        self.register_buffer('_tensor_constant11', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant12', tensor(64.))
        self.register_buffer('_tensor_constant13', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant14', tensor(64.))
        self.register_buffer('_tensor_constant15', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant16', tensor(64.))
        self.register_buffer('_tensor_constant17', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant18', tensor(64.))
        self.register_buffer('_tensor_constant19', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant20', tensor(64.))
        self.register_buffer('_tensor_constant21', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant22', tensor(64.))
        self.register_buffer('_tensor_constant23', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant24', tensor(64.))
        self.register_buffer('_tensor_constant25', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant26', tensor(64.))
        self.register_buffer('_tensor_constant27', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant28', tensor(64.))
        self.register_buffer('_tensor_constant29', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant30', tensor(64.))
        self.register_buffer('_tensor_constant31', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant32', tensor(64.))
        self.register_buffer('_tensor_constant33', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant34', tensor(64.))
        self.register_buffer('_tensor_constant35', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant36', tensor(64.))
        self.register_buffer('_tensor_constant37', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant38', tensor(64.))
        self.register_buffer('_tensor_constant39', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant40', tensor(64.))
        self.register_buffer('_tensor_constant41', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant42', tensor(64.))
        self.register_buffer('_tensor_constant43', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant44', tensor(64.))
        self.register_buffer('_tensor_constant45', tensor(-3.4028e+38))
        self.register_buffer('_tensor_constant46', tensor(64.))
        self.register_buffer('_tensor_constant47', tensor(-3.4028e+38))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1):
        full = torch.ops.aten.full.default([2, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        embedding = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg0_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg3_1, arg1_1);  arg3_1 = arg1_1 = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-07);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        full_default = torch.ops.aten.full.default([2, 512, 1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        squeeze = torch.ops.aten.squeeze.dim(unsqueeze_2, -2)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(squeeze, -1);  squeeze = None
        mul_3 = torch.ops.aten.mul.Tensor(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None
        view = torch.ops.aten.view.default(add_2, [1024, 1536])
        permute = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [2, 512, 1536]);  addmm = None
        view_2 = torch.ops.aten.view.default(view_1, [2, 512, 24, -1]);  view_1 = None
        permute_1 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3 = torch.ops.aten.view.default(clone, [-1, 512, 64]);  clone = None
        view_4 = torch.ops.aten.view.default(add_2, [1024, 1536])
        permute_2 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg9_1, view_4, permute_2);  arg9_1 = view_4 = permute_2 = None
        view_5 = torch.ops.aten.view.default(addmm_1, [2, 512, 1536]);  addmm_1 = None
        view_6 = torch.ops.aten.view.default(view_5, [2, 512, 24, -1]);  view_5 = None
        permute_3 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone_1 = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_7 = torch.ops.aten.view.default(clone_1, [-1, 512, 64]);  clone_1 = None
        view_8 = torch.ops.aten.view.default(add_2, [1024, 1536])
        permute_4 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg11_1, view_8, permute_4);  arg11_1 = view_8 = permute_4 = None
        view_9 = torch.ops.aten.view.default(addmm_2, [2, 512, 1536]);  addmm_2 = None
        view_10 = torch.ops.aten.view.default(view_9, [2, 512, 24, -1]);  view_9 = None
        permute_5 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_2 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_11 = torch.ops.aten.view.default(clone_2, [-1, 512, 64]);  clone_2 = None
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        mul_4 = torch.ops.aten.mul.Tensor(lift_fresh_copy, 1);  lift_fresh_copy = mul_4 = None
        full_default_1 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_6 = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        div = torch.ops.aten.div.Tensor(permute_6, full_default_1);  permute_6 = full_default_1 = None
        bmm = torch.ops.aten.bmm.default(view_3, div);  view_3 = div = None
        view_12 = torch.ops.aten.view.default(bmm, [-1, 24, 512, 512]);  bmm = None
        convert_element_type = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type = None
        full_default_2 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant1 = self._tensor_constant1;  _tensor_constant1 = None
        full_default_3 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where = torch.ops.aten.where.self(full_default_2, full_default_3, view_12);  full_default_3 = view_12 = None
        amax = torch.ops.aten.amax.default(where, [-1], True)
        sub_1 = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
        exp = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        full_default_4 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(full_default_2, full_default_4, div_1);  full_default_2 = full_default_4 = div_1 = None
        view_14 = torch.ops.aten.view.default(where_1, [-1, 512, 512]);  where_1 = None
        bmm_1 = torch.ops.aten.bmm.default(view_14, view_11);  view_14 = view_11 = None
        view_15 = torch.ops.aten.view.default(bmm_1, [-1, 24, 512, 64]);  bmm_1 = None
        permute_7 = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        clone_3 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_16 = torch.ops.aten.view.default(clone_3, [2, 512, -1]);  clone_3 = None
        view_17 = torch.ops.aten.view.default(view_16, [1024, 1536]);  view_16 = None
        permute_8 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg13_1, view_17, permute_8);  arg13_1 = view_17 = permute_8 = None
        view_18 = torch.ops.aten.view.default(addmm_3, [2, 512, 1536]);  addmm_3 = None
        add_3 = torch.ops.aten.add.Tensor(view_18, add_2);  view_18 = add_2 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_2, 1e-07);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
        add_5 = torch.ops.aten.add.Tensor(mul_6, arg15_1);  mul_6 = arg15_1 = None
        view_19 = torch.ops.aten.view.default(add_5, [1024, 1536])
        permute_9 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg17_1, view_19, permute_9);  arg17_1 = view_19 = permute_9 = None
        view_20 = torch.ops.aten.view.default(addmm_4, [2, 512, 6144]);  addmm_4 = None
        mul_7 = torch.ops.aten.mul.Tensor(view_20, 0.5)
        mul_8 = torch.ops.aten.mul.Tensor(view_20, 0.7071067811865476);  view_20 = None
        erf = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_6 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_7, add_6);  mul_7 = add_6 = None
        view_21 = torch.ops.aten.view.default(mul_9, [1024, 6144]);  mul_9 = None
        permute_10 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg19_1, view_21, permute_10);  arg19_1 = view_21 = permute_10 = None
        view_22 = torch.ops.aten.view.default(addmm_5, [2, 512, 1536]);  addmm_5 = None
        add_7 = torch.ops.aten.add.Tensor(view_22, add_5);  view_22 = add_5 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_4, 1e-07);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_7, getitem_5);  add_7 = getitem_5 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
        add_9 = torch.ops.aten.add.Tensor(mul_11, arg21_1);  mul_11 = arg21_1 = None
        view_23 = torch.ops.aten.view.default(add_9, [1024, 1536])
        permute_11 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg23_1, view_23, permute_11);  arg23_1 = view_23 = permute_11 = None
        view_24 = torch.ops.aten.view.default(addmm_6, [2, 512, 1536]);  addmm_6 = None
        view_25 = torch.ops.aten.view.default(view_24, [2, 512, 24, -1]);  view_24 = None
        permute_12 = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        clone_4 = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        view_26 = torch.ops.aten.view.default(clone_4, [-1, 512, 64]);  clone_4 = None
        view_27 = torch.ops.aten.view.default(add_9, [1024, 1536])
        permute_13 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg25_1, view_27, permute_13);  arg25_1 = view_27 = permute_13 = None
        view_28 = torch.ops.aten.view.default(addmm_7, [2, 512, 1536]);  addmm_7 = None
        view_29 = torch.ops.aten.view.default(view_28, [2, 512, 24, -1]);  view_28 = None
        permute_14 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        clone_5 = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_30 = torch.ops.aten.view.default(clone_5, [-1, 512, 64]);  clone_5 = None
        view_31 = torch.ops.aten.view.default(add_9, [1024, 1536])
        permute_15 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg27_1, view_31, permute_15);  arg27_1 = view_31 = permute_15 = None
        view_32 = torch.ops.aten.view.default(addmm_8, [2, 512, 1536]);  addmm_8 = None
        view_33 = torch.ops.aten.view.default(view_32, [2, 512, 24, -1]);  view_32 = None
        permute_16 = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        clone_6 = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_34 = torch.ops.aten.view.default(clone_6, [-1, 512, 64]);  clone_6 = None
        _tensor_constant2 = self._tensor_constant2
        lift_fresh_copy_2 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        mul_12 = torch.ops.aten.mul.Tensor(lift_fresh_copy_2, 1);  lift_fresh_copy_2 = mul_12 = None
        full_default_5 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_17 = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
        div_2 = torch.ops.aten.div.Tensor(permute_17, full_default_5);  permute_17 = full_default_5 = None
        bmm_2 = torch.ops.aten.bmm.default(view_26, div_2);  view_26 = div_2 = None
        view_35 = torch.ops.aten.view.default(bmm_2, [-1, 24, 512, 512]);  bmm_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_1 = None
        full_default_6 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant3 = self._tensor_constant3;  _tensor_constant3 = None
        full_default_7 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_2 = torch.ops.aten.where.self(full_default_6, full_default_7, view_35);  full_default_7 = view_35 = None
        amax_1 = torch.ops.aten.amax.default(where_2, [-1], True)
        sub_4 = torch.ops.aten.sub.Tensor(where_2, amax_1);  where_2 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        full_default_8 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(full_default_6, full_default_8, div_3);  full_default_6 = full_default_8 = div_3 = None
        view_37 = torch.ops.aten.view.default(where_3, [-1, 512, 512]);  where_3 = None
        bmm_3 = torch.ops.aten.bmm.default(view_37, view_34);  view_37 = view_34 = None
        view_38 = torch.ops.aten.view.default(bmm_3, [-1, 24, 512, 64]);  bmm_3 = None
        permute_18 = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_7 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_39 = torch.ops.aten.view.default(clone_7, [2, 512, -1]);  clone_7 = None
        view_40 = torch.ops.aten.view.default(view_39, [1024, 1536]);  view_39 = None
        permute_19 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg29_1, view_40, permute_19);  arg29_1 = view_40 = permute_19 = None
        view_41 = torch.ops.aten.view.default(addmm_9, [2, 512, 1536]);  addmm_9 = None
        add_10 = torch.ops.aten.add.Tensor(view_41, add_9);  view_41 = add_9 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_6, 1e-07);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_10, getitem_7);  add_10 = getitem_7 = None
        mul_13 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
        add_12 = torch.ops.aten.add.Tensor(mul_14, arg31_1);  mul_14 = arg31_1 = None
        view_42 = torch.ops.aten.view.default(add_12, [1024, 1536])
        permute_20 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg33_1, view_42, permute_20);  arg33_1 = view_42 = permute_20 = None
        view_43 = torch.ops.aten.view.default(addmm_10, [2, 512, 6144]);  addmm_10 = None
        mul_15 = torch.ops.aten.mul.Tensor(view_43, 0.5)
        mul_16 = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
        erf_1 = torch.ops.aten.erf.default(mul_16);  mul_16 = None
        add_13 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_15, add_13);  mul_15 = add_13 = None
        view_44 = torch.ops.aten.view.default(mul_17, [1024, 6144]);  mul_17 = None
        permute_21 = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg35_1, view_44, permute_21);  arg35_1 = view_44 = permute_21 = None
        view_45 = torch.ops.aten.view.default(addmm_11, [2, 512, 1536]);  addmm_11 = None
        add_14 = torch.ops.aten.add.Tensor(view_45, add_12);  view_45 = add_12 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_15 = torch.ops.aten.add.Tensor(getitem_8, 1e-07);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_14, getitem_9);  add_14 = getitem_9 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
        add_16 = torch.ops.aten.add.Tensor(mul_19, arg37_1);  mul_19 = arg37_1 = None
        view_46 = torch.ops.aten.view.default(add_16, [1024, 1536])
        permute_22 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg39_1, view_46, permute_22);  arg39_1 = view_46 = permute_22 = None
        view_47 = torch.ops.aten.view.default(addmm_12, [2, 512, 1536]);  addmm_12 = None
        view_48 = torch.ops.aten.view.default(view_47, [2, 512, 24, -1]);  view_47 = None
        permute_23 = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        clone_8 = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_49 = torch.ops.aten.view.default(clone_8, [-1, 512, 64]);  clone_8 = None
        view_50 = torch.ops.aten.view.default(add_16, [1024, 1536])
        permute_24 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg41_1, view_50, permute_24);  arg41_1 = view_50 = permute_24 = None
        view_51 = torch.ops.aten.view.default(addmm_13, [2, 512, 1536]);  addmm_13 = None
        view_52 = torch.ops.aten.view.default(view_51, [2, 512, 24, -1]);  view_51 = None
        permute_25 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        clone_9 = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        view_53 = torch.ops.aten.view.default(clone_9, [-1, 512, 64]);  clone_9 = None
        view_54 = torch.ops.aten.view.default(add_16, [1024, 1536])
        permute_26 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg43_1, view_54, permute_26);  arg43_1 = view_54 = permute_26 = None
        view_55 = torch.ops.aten.view.default(addmm_14, [2, 512, 1536]);  addmm_14 = None
        view_56 = torch.ops.aten.view.default(view_55, [2, 512, 24, -1]);  view_55 = None
        permute_27 = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        clone_10 = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_57 = torch.ops.aten.view.default(clone_10, [-1, 512, 64]);  clone_10 = None
        _tensor_constant4 = self._tensor_constant4
        lift_fresh_copy_4 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
        mul_20 = torch.ops.aten.mul.Tensor(lift_fresh_copy_4, 1);  lift_fresh_copy_4 = mul_20 = None
        full_default_9 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_28 = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        div_4 = torch.ops.aten.div.Tensor(permute_28, full_default_9);  permute_28 = full_default_9 = None
        bmm_4 = torch.ops.aten.bmm.default(view_49, div_4);  view_49 = div_4 = None
        view_58 = torch.ops.aten.view.default(bmm_4, [-1, 24, 512, 512]);  bmm_4 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_2 = None
        full_default_10 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant5 = self._tensor_constant5;  _tensor_constant5 = None
        full_default_11 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_4 = torch.ops.aten.where.self(full_default_10, full_default_11, view_58);  full_default_11 = view_58 = None
        amax_2 = torch.ops.aten.amax.default(where_4, [-1], True)
        sub_7 = torch.ops.aten.sub.Tensor(where_4, amax_2);  where_4 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        full_default_12 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5 = torch.ops.aten.where.self(full_default_10, full_default_12, div_5);  full_default_10 = full_default_12 = div_5 = None
        view_60 = torch.ops.aten.view.default(where_5, [-1, 512, 512]);  where_5 = None
        bmm_5 = torch.ops.aten.bmm.default(view_60, view_57);  view_60 = view_57 = None
        view_61 = torch.ops.aten.view.default(bmm_5, [-1, 24, 512, 64]);  bmm_5 = None
        permute_29 = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
        clone_11 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_62 = torch.ops.aten.view.default(clone_11, [2, 512, -1]);  clone_11 = None
        view_63 = torch.ops.aten.view.default(view_62, [1024, 1536]);  view_62 = None
        permute_30 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg45_1, view_63, permute_30);  arg45_1 = view_63 = permute_30 = None
        view_64 = torch.ops.aten.view.default(addmm_15, [2, 512, 1536]);  addmm_15 = None
        add_17 = torch.ops.aten.add.Tensor(view_64, add_16);  view_64 = add_16 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_10, 1e-07);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_17, getitem_11);  add_17 = getitem_11 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, arg46_1);  mul_21 = arg46_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_22, arg47_1);  mul_22 = arg47_1 = None
        view_65 = torch.ops.aten.view.default(add_19, [1024, 1536])
        permute_31 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg49_1, view_65, permute_31);  arg49_1 = view_65 = permute_31 = None
        view_66 = torch.ops.aten.view.default(addmm_16, [2, 512, 6144]);  addmm_16 = None
        mul_23 = torch.ops.aten.mul.Tensor(view_66, 0.5)
        mul_24 = torch.ops.aten.mul.Tensor(view_66, 0.7071067811865476);  view_66 = None
        erf_2 = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_20 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_23, add_20);  mul_23 = add_20 = None
        view_67 = torch.ops.aten.view.default(mul_25, [1024, 6144]);  mul_25 = None
        permute_32 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg51_1, view_67, permute_32);  arg51_1 = view_67 = permute_32 = None
        view_68 = torch.ops.aten.view.default(addmm_17, [2, 512, 1536]);  addmm_17 = None
        add_21 = torch.ops.aten.add.Tensor(view_68, add_19);  view_68 = add_19 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_12, 1e-07);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_21, getitem_13);  add_21 = getitem_13 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, arg52_1);  mul_26 = arg52_1 = None
        add_23 = torch.ops.aten.add.Tensor(mul_27, arg53_1);  mul_27 = arg53_1 = None
        view_69 = torch.ops.aten.view.default(add_23, [1024, 1536])
        permute_33 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg55_1, view_69, permute_33);  arg55_1 = view_69 = permute_33 = None
        view_70 = torch.ops.aten.view.default(addmm_18, [2, 512, 1536]);  addmm_18 = None
        view_71 = torch.ops.aten.view.default(view_70, [2, 512, 24, -1]);  view_70 = None
        permute_34 = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        clone_12 = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_72 = torch.ops.aten.view.default(clone_12, [-1, 512, 64]);  clone_12 = None
        view_73 = torch.ops.aten.view.default(add_23, [1024, 1536])
        permute_35 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg57_1, view_73, permute_35);  arg57_1 = view_73 = permute_35 = None
        view_74 = torch.ops.aten.view.default(addmm_19, [2, 512, 1536]);  addmm_19 = None
        view_75 = torch.ops.aten.view.default(view_74, [2, 512, 24, -1]);  view_74 = None
        permute_36 = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        clone_13 = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_76 = torch.ops.aten.view.default(clone_13, [-1, 512, 64]);  clone_13 = None
        view_77 = torch.ops.aten.view.default(add_23, [1024, 1536])
        permute_37 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg59_1, view_77, permute_37);  arg59_1 = view_77 = permute_37 = None
        view_78 = torch.ops.aten.view.default(addmm_20, [2, 512, 1536]);  addmm_20 = None
        view_79 = torch.ops.aten.view.default(view_78, [2, 512, 24, -1]);  view_78 = None
        permute_38 = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
        clone_14 = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_80 = torch.ops.aten.view.default(clone_14, [-1, 512, 64]);  clone_14 = None
        _tensor_constant6 = self._tensor_constant6
        lift_fresh_copy_6 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
        mul_28 = torch.ops.aten.mul.Tensor(lift_fresh_copy_6, 1);  lift_fresh_copy_6 = mul_28 = None
        full_default_13 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_39 = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
        div_6 = torch.ops.aten.div.Tensor(permute_39, full_default_13);  permute_39 = full_default_13 = None
        bmm_6 = torch.ops.aten.bmm.default(view_72, div_6);  view_72 = div_6 = None
        view_81 = torch.ops.aten.view.default(bmm_6, [-1, 24, 512, 512]);  bmm_6 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_3 = None
        full_default_14 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant7 = self._tensor_constant7;  _tensor_constant7 = None
        full_default_15 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_6 = torch.ops.aten.where.self(full_default_14, full_default_15, view_81);  full_default_15 = view_81 = None
        amax_3 = torch.ops.aten.amax.default(where_6, [-1], True)
        sub_10 = torch.ops.aten.sub.Tensor(where_6, amax_3);  where_6 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        full_default_16 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7 = torch.ops.aten.where.self(full_default_14, full_default_16, div_7);  full_default_14 = full_default_16 = div_7 = None
        view_83 = torch.ops.aten.view.default(where_7, [-1, 512, 512]);  where_7 = None
        bmm_7 = torch.ops.aten.bmm.default(view_83, view_80);  view_83 = view_80 = None
        view_84 = torch.ops.aten.view.default(bmm_7, [-1, 24, 512, 64]);  bmm_7 = None
        permute_40 = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        clone_15 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_85 = torch.ops.aten.view.default(clone_15, [2, 512, -1]);  clone_15 = None
        view_86 = torch.ops.aten.view.default(view_85, [1024, 1536]);  view_85 = None
        permute_41 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg61_1, view_86, permute_41);  arg61_1 = view_86 = permute_41 = None
        view_87 = torch.ops.aten.view.default(addmm_21, [2, 512, 1536]);  addmm_21 = None
        add_24 = torch.ops.aten.add.Tensor(view_87, add_23);  view_87 = add_23 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_25 = torch.ops.aten.add.Tensor(getitem_14, 1e-07);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_24, getitem_15);  add_24 = getitem_15 = None
        mul_29 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, arg62_1);  mul_29 = arg62_1 = None
        add_26 = torch.ops.aten.add.Tensor(mul_30, arg63_1);  mul_30 = arg63_1 = None
        view_88 = torch.ops.aten.view.default(add_26, [1024, 1536])
        permute_42 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg65_1, view_88, permute_42);  arg65_1 = view_88 = permute_42 = None
        view_89 = torch.ops.aten.view.default(addmm_22, [2, 512, 6144]);  addmm_22 = None
        mul_31 = torch.ops.aten.mul.Tensor(view_89, 0.5)
        mul_32 = torch.ops.aten.mul.Tensor(view_89, 0.7071067811865476);  view_89 = None
        erf_3 = torch.ops.aten.erf.default(mul_32);  mul_32 = None
        add_27 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_31, add_27);  mul_31 = add_27 = None
        view_90 = torch.ops.aten.view.default(mul_33, [1024, 6144]);  mul_33 = None
        permute_43 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg67_1, view_90, permute_43);  arg67_1 = view_90 = permute_43 = None
        view_91 = torch.ops.aten.view.default(addmm_23, [2, 512, 1536]);  addmm_23 = None
        add_28 = torch.ops.aten.add.Tensor(view_91, add_26);  view_91 = add_26 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_16, 1e-07);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_28, getitem_17);  add_28 = getitem_17 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, arg68_1);  mul_34 = arg68_1 = None
        add_30 = torch.ops.aten.add.Tensor(mul_35, arg69_1);  mul_35 = arg69_1 = None
        view_92 = torch.ops.aten.view.default(add_30, [1024, 1536])
        permute_44 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg71_1, view_92, permute_44);  arg71_1 = view_92 = permute_44 = None
        view_93 = torch.ops.aten.view.default(addmm_24, [2, 512, 1536]);  addmm_24 = None
        view_94 = torch.ops.aten.view.default(view_93, [2, 512, 24, -1]);  view_93 = None
        permute_45 = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        clone_16 = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_95 = torch.ops.aten.view.default(clone_16, [-1, 512, 64]);  clone_16 = None
        view_96 = torch.ops.aten.view.default(add_30, [1024, 1536])
        permute_46 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg73_1, view_96, permute_46);  arg73_1 = view_96 = permute_46 = None
        view_97 = torch.ops.aten.view.default(addmm_25, [2, 512, 1536]);  addmm_25 = None
        view_98 = torch.ops.aten.view.default(view_97, [2, 512, 24, -1]);  view_97 = None
        permute_47 = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        clone_17 = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_99 = torch.ops.aten.view.default(clone_17, [-1, 512, 64]);  clone_17 = None
        view_100 = torch.ops.aten.view.default(add_30, [1024, 1536])
        permute_48 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg75_1, view_100, permute_48);  arg75_1 = view_100 = permute_48 = None
        view_101 = torch.ops.aten.view.default(addmm_26, [2, 512, 1536]);  addmm_26 = None
        view_102 = torch.ops.aten.view.default(view_101, [2, 512, 24, -1]);  view_101 = None
        permute_49 = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_18 = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_103 = torch.ops.aten.view.default(clone_18, [-1, 512, 64]);  clone_18 = None
        _tensor_constant8 = self._tensor_constant8
        lift_fresh_copy_8 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
        mul_36 = torch.ops.aten.mul.Tensor(lift_fresh_copy_8, 1);  lift_fresh_copy_8 = mul_36 = None
        full_default_17 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_50 = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        div_8 = torch.ops.aten.div.Tensor(permute_50, full_default_17);  permute_50 = full_default_17 = None
        bmm_8 = torch.ops.aten.bmm.default(view_95, div_8);  view_95 = div_8 = None
        view_104 = torch.ops.aten.view.default(bmm_8, [-1, 24, 512, 512]);  bmm_8 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_4 = None
        full_default_18 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant9 = self._tensor_constant9;  _tensor_constant9 = None
        full_default_19 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_8 = torch.ops.aten.where.self(full_default_18, full_default_19, view_104);  full_default_19 = view_104 = None
        amax_4 = torch.ops.aten.amax.default(where_8, [-1], True)
        sub_13 = torch.ops.aten.sub.Tensor(where_8, amax_4);  where_8 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        full_default_20 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9 = torch.ops.aten.where.self(full_default_18, full_default_20, div_9);  full_default_18 = full_default_20 = div_9 = None
        view_106 = torch.ops.aten.view.default(where_9, [-1, 512, 512]);  where_9 = None
        bmm_9 = torch.ops.aten.bmm.default(view_106, view_103);  view_106 = view_103 = None
        view_107 = torch.ops.aten.view.default(bmm_9, [-1, 24, 512, 64]);  bmm_9 = None
        permute_51 = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
        clone_19 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_108 = torch.ops.aten.view.default(clone_19, [2, 512, -1]);  clone_19 = None
        view_109 = torch.ops.aten.view.default(view_108, [1024, 1536]);  view_108 = None
        permute_52 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg77_1, view_109, permute_52);  arg77_1 = view_109 = permute_52 = None
        view_110 = torch.ops.aten.view.default(addmm_27, [2, 512, 1536]);  addmm_27 = None
        add_31 = torch.ops.aten.add.Tensor(view_110, add_30);  view_110 = add_30 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_18, 1e-07);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_31, getitem_19);  add_31 = getitem_19 = None
        mul_37 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, arg78_1);  mul_37 = arg78_1 = None
        add_33 = torch.ops.aten.add.Tensor(mul_38, arg79_1);  mul_38 = arg79_1 = None
        view_111 = torch.ops.aten.view.default(add_33, [1024, 1536])
        permute_53 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg81_1, view_111, permute_53);  arg81_1 = view_111 = permute_53 = None
        view_112 = torch.ops.aten.view.default(addmm_28, [2, 512, 6144]);  addmm_28 = None
        mul_39 = torch.ops.aten.mul.Tensor(view_112, 0.5)
        mul_40 = torch.ops.aten.mul.Tensor(view_112, 0.7071067811865476);  view_112 = None
        erf_4 = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_34 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_39, add_34);  mul_39 = add_34 = None
        view_113 = torch.ops.aten.view.default(mul_41, [1024, 6144]);  mul_41 = None
        permute_54 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg83_1, view_113, permute_54);  arg83_1 = view_113 = permute_54 = None
        view_114 = torch.ops.aten.view.default(addmm_29, [2, 512, 1536]);  addmm_29 = None
        add_35 = torch.ops.aten.add.Tensor(view_114, add_33);  view_114 = add_33 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_20, 1e-07);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_35, getitem_21);  add_35 = getitem_21 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg84_1);  mul_42 = arg84_1 = None
        add_37 = torch.ops.aten.add.Tensor(mul_43, arg85_1);  mul_43 = arg85_1 = None
        view_115 = torch.ops.aten.view.default(add_37, [1024, 1536])
        permute_55 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg87_1, view_115, permute_55);  arg87_1 = view_115 = permute_55 = None
        view_116 = torch.ops.aten.view.default(addmm_30, [2, 512, 1536]);  addmm_30 = None
        view_117 = torch.ops.aten.view.default(view_116, [2, 512, 24, -1]);  view_116 = None
        permute_56 = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        clone_20 = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_118 = torch.ops.aten.view.default(clone_20, [-1, 512, 64]);  clone_20 = None
        view_119 = torch.ops.aten.view.default(add_37, [1024, 1536])
        permute_57 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg89_1, view_119, permute_57);  arg89_1 = view_119 = permute_57 = None
        view_120 = torch.ops.aten.view.default(addmm_31, [2, 512, 1536]);  addmm_31 = None
        view_121 = torch.ops.aten.view.default(view_120, [2, 512, 24, -1]);  view_120 = None
        permute_58 = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
        clone_21 = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        view_122 = torch.ops.aten.view.default(clone_21, [-1, 512, 64]);  clone_21 = None
        view_123 = torch.ops.aten.view.default(add_37, [1024, 1536])
        permute_59 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg91_1, view_123, permute_59);  arg91_1 = view_123 = permute_59 = None
        view_124 = torch.ops.aten.view.default(addmm_32, [2, 512, 1536]);  addmm_32 = None
        view_125 = torch.ops.aten.view.default(view_124, [2, 512, 24, -1]);  view_124 = None
        permute_60 = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        clone_22 = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_126 = torch.ops.aten.view.default(clone_22, [-1, 512, 64]);  clone_22 = None
        _tensor_constant10 = self._tensor_constant10
        lift_fresh_copy_10 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
        mul_44 = torch.ops.aten.mul.Tensor(lift_fresh_copy_10, 1);  lift_fresh_copy_10 = mul_44 = None
        full_default_21 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_61 = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        div_10 = torch.ops.aten.div.Tensor(permute_61, full_default_21);  permute_61 = full_default_21 = None
        bmm_10 = torch.ops.aten.bmm.default(view_118, div_10);  view_118 = div_10 = None
        view_127 = torch.ops.aten.view.default(bmm_10, [-1, 24, 512, 512]);  bmm_10 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_5 = None
        full_default_22 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant11 = self._tensor_constant11;  _tensor_constant11 = None
        full_default_23 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_10 = torch.ops.aten.where.self(full_default_22, full_default_23, view_127);  full_default_23 = view_127 = None
        amax_5 = torch.ops.aten.amax.default(where_10, [-1], True)
        sub_16 = torch.ops.aten.sub.Tensor(where_10, amax_5);  where_10 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_11 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        full_default_24 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11 = torch.ops.aten.where.self(full_default_22, full_default_24, div_11);  full_default_22 = full_default_24 = div_11 = None
        view_129 = torch.ops.aten.view.default(where_11, [-1, 512, 512]);  where_11 = None
        bmm_11 = torch.ops.aten.bmm.default(view_129, view_126);  view_129 = view_126 = None
        view_130 = torch.ops.aten.view.default(bmm_11, [-1, 24, 512, 64]);  bmm_11 = None
        permute_62 = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        clone_23 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_131 = torch.ops.aten.view.default(clone_23, [2, 512, -1]);  clone_23 = None
        view_132 = torch.ops.aten.view.default(view_131, [1024, 1536]);  view_131 = None
        permute_63 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg93_1, view_132, permute_63);  arg93_1 = view_132 = permute_63 = None
        view_133 = torch.ops.aten.view.default(addmm_33, [2, 512, 1536]);  addmm_33 = None
        add_38 = torch.ops.aten.add.Tensor(view_133, add_37);  view_133 = add_37 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_39 = torch.ops.aten.add.Tensor(getitem_22, 1e-07);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_38, getitem_23);  add_38 = getitem_23 = None
        mul_45 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, arg94_1);  mul_45 = arg94_1 = None
        add_40 = torch.ops.aten.add.Tensor(mul_46, arg95_1);  mul_46 = arg95_1 = None
        view_134 = torch.ops.aten.view.default(add_40, [1024, 1536])
        permute_64 = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg97_1, view_134, permute_64);  arg97_1 = view_134 = permute_64 = None
        view_135 = torch.ops.aten.view.default(addmm_34, [2, 512, 6144]);  addmm_34 = None
        mul_47 = torch.ops.aten.mul.Tensor(view_135, 0.5)
        mul_48 = torch.ops.aten.mul.Tensor(view_135, 0.7071067811865476);  view_135 = None
        erf_5 = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_41 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_47, add_41);  mul_47 = add_41 = None
        view_136 = torch.ops.aten.view.default(mul_49, [1024, 6144]);  mul_49 = None
        permute_65 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg99_1, view_136, permute_65);  arg99_1 = view_136 = permute_65 = None
        view_137 = torch.ops.aten.view.default(addmm_35, [2, 512, 1536]);  addmm_35 = None
        add_42 = torch.ops.aten.add.Tensor(view_137, add_40);  view_137 = add_40 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_43 = torch.ops.aten.add.Tensor(getitem_24, 1e-07);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_42, getitem_25);  add_42 = getitem_25 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, arg100_1);  mul_50 = arg100_1 = None
        add_44 = torch.ops.aten.add.Tensor(mul_51, arg101_1);  mul_51 = arg101_1 = None
        view_138 = torch.ops.aten.view.default(add_44, [1024, 1536])
        permute_66 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg103_1, view_138, permute_66);  arg103_1 = view_138 = permute_66 = None
        view_139 = torch.ops.aten.view.default(addmm_36, [2, 512, 1536]);  addmm_36 = None
        view_140 = torch.ops.aten.view.default(view_139, [2, 512, 24, -1]);  view_139 = None
        permute_67 = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        clone_24 = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        view_141 = torch.ops.aten.view.default(clone_24, [-1, 512, 64]);  clone_24 = None
        view_142 = torch.ops.aten.view.default(add_44, [1024, 1536])
        permute_68 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg105_1, view_142, permute_68);  arg105_1 = view_142 = permute_68 = None
        view_143 = torch.ops.aten.view.default(addmm_37, [2, 512, 1536]);  addmm_37 = None
        view_144 = torch.ops.aten.view.default(view_143, [2, 512, 24, -1]);  view_143 = None
        permute_69 = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        clone_25 = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        view_145 = torch.ops.aten.view.default(clone_25, [-1, 512, 64]);  clone_25 = None
        view_146 = torch.ops.aten.view.default(add_44, [1024, 1536])
        permute_70 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg107_1, view_146, permute_70);  arg107_1 = view_146 = permute_70 = None
        view_147 = torch.ops.aten.view.default(addmm_38, [2, 512, 1536]);  addmm_38 = None
        view_148 = torch.ops.aten.view.default(view_147, [2, 512, 24, -1]);  view_147 = None
        permute_71 = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_26 = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_149 = torch.ops.aten.view.default(clone_26, [-1, 512, 64]);  clone_26 = None
        _tensor_constant12 = self._tensor_constant12
        lift_fresh_copy_12 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
        mul_52 = torch.ops.aten.mul.Tensor(lift_fresh_copy_12, 1);  lift_fresh_copy_12 = mul_52 = None
        full_default_25 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_72 = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        div_12 = torch.ops.aten.div.Tensor(permute_72, full_default_25);  permute_72 = full_default_25 = None
        bmm_12 = torch.ops.aten.bmm.default(view_141, div_12);  view_141 = div_12 = None
        view_150 = torch.ops.aten.view.default(bmm_12, [-1, 24, 512, 512]);  bmm_12 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_6 = None
        full_default_26 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant13 = self._tensor_constant13;  _tensor_constant13 = None
        full_default_27 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_12 = torch.ops.aten.where.self(full_default_26, full_default_27, view_150);  full_default_27 = view_150 = None
        amax_6 = torch.ops.aten.amax.default(where_12, [-1], True)
        sub_19 = torch.ops.aten.sub.Tensor(where_12, amax_6);  where_12 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_13 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        full_default_28 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13 = torch.ops.aten.where.self(full_default_26, full_default_28, div_13);  full_default_26 = full_default_28 = div_13 = None
        view_152 = torch.ops.aten.view.default(where_13, [-1, 512, 512]);  where_13 = None
        bmm_13 = torch.ops.aten.bmm.default(view_152, view_149);  view_152 = view_149 = None
        view_153 = torch.ops.aten.view.default(bmm_13, [-1, 24, 512, 64]);  bmm_13 = None
        permute_73 = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
        clone_27 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_154 = torch.ops.aten.view.default(clone_27, [2, 512, -1]);  clone_27 = None
        view_155 = torch.ops.aten.view.default(view_154, [1024, 1536]);  view_154 = None
        permute_74 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg109_1, view_155, permute_74);  arg109_1 = view_155 = permute_74 = None
        view_156 = torch.ops.aten.view.default(addmm_39, [2, 512, 1536]);  addmm_39 = None
        add_45 = torch.ops.aten.add.Tensor(view_156, add_44);  view_156 = add_44 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_26, 1e-07);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_45, getitem_27);  add_45 = getitem_27 = None
        mul_53 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_54 = torch.ops.aten.mul.Tensor(mul_53, arg110_1);  mul_53 = arg110_1 = None
        add_47 = torch.ops.aten.add.Tensor(mul_54, arg111_1);  mul_54 = arg111_1 = None
        view_157 = torch.ops.aten.view.default(add_47, [1024, 1536])
        permute_75 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg113_1, view_157, permute_75);  arg113_1 = view_157 = permute_75 = None
        view_158 = torch.ops.aten.view.default(addmm_40, [2, 512, 6144]);  addmm_40 = None
        mul_55 = torch.ops.aten.mul.Tensor(view_158, 0.5)
        mul_56 = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
        erf_6 = torch.ops.aten.erf.default(mul_56);  mul_56 = None
        add_48 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_55, add_48);  mul_55 = add_48 = None
        view_159 = torch.ops.aten.view.default(mul_57, [1024, 6144]);  mul_57 = None
        permute_76 = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg115_1, view_159, permute_76);  arg115_1 = view_159 = permute_76 = None
        view_160 = torch.ops.aten.view.default(addmm_41, [2, 512, 1536]);  addmm_41 = None
        add_49 = torch.ops.aten.add.Tensor(view_160, add_47);  view_160 = add_47 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_28, 1e-07);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_49, getitem_29);  add_49 = getitem_29 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, arg116_1);  mul_58 = arg116_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_59, arg117_1);  mul_59 = arg117_1 = None
        view_161 = torch.ops.aten.view.default(add_51, [1024, 1536])
        permute_77 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg119_1, view_161, permute_77);  arg119_1 = view_161 = permute_77 = None
        view_162 = torch.ops.aten.view.default(addmm_42, [2, 512, 1536]);  addmm_42 = None
        view_163 = torch.ops.aten.view.default(view_162, [2, 512, 24, -1]);  view_162 = None
        permute_78 = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
        clone_28 = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        view_164 = torch.ops.aten.view.default(clone_28, [-1, 512, 64]);  clone_28 = None
        view_165 = torch.ops.aten.view.default(add_51, [1024, 1536])
        permute_79 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg121_1, view_165, permute_79);  arg121_1 = view_165 = permute_79 = None
        view_166 = torch.ops.aten.view.default(addmm_43, [2, 512, 1536]);  addmm_43 = None
        view_167 = torch.ops.aten.view.default(view_166, [2, 512, 24, -1]);  view_166 = None
        permute_80 = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
        clone_29 = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_168 = torch.ops.aten.view.default(clone_29, [-1, 512, 64]);  clone_29 = None
        view_169 = torch.ops.aten.view.default(add_51, [1024, 1536])
        permute_81 = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg123_1, view_169, permute_81);  arg123_1 = view_169 = permute_81 = None
        view_170 = torch.ops.aten.view.default(addmm_44, [2, 512, 1536]);  addmm_44 = None
        view_171 = torch.ops.aten.view.default(view_170, [2, 512, 24, -1]);  view_170 = None
        permute_82 = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
        clone_30 = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_172 = torch.ops.aten.view.default(clone_30, [-1, 512, 64]);  clone_30 = None
        _tensor_constant14 = self._tensor_constant14
        lift_fresh_copy_14 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
        mul_60 = torch.ops.aten.mul.Tensor(lift_fresh_copy_14, 1);  lift_fresh_copy_14 = mul_60 = None
        full_default_29 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_83 = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        div_14 = torch.ops.aten.div.Tensor(permute_83, full_default_29);  permute_83 = full_default_29 = None
        bmm_14 = torch.ops.aten.bmm.default(view_164, div_14);  view_164 = div_14 = None
        view_173 = torch.ops.aten.view.default(bmm_14, [-1, 24, 512, 512]);  bmm_14 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_7 = None
        full_default_30 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant15 = self._tensor_constant15;  _tensor_constant15 = None
        full_default_31 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_14 = torch.ops.aten.where.self(full_default_30, full_default_31, view_173);  full_default_31 = view_173 = None
        amax_7 = torch.ops.aten.amax.default(where_14, [-1], True)
        sub_22 = torch.ops.aten.sub.Tensor(where_14, amax_7);  where_14 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_15 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        full_default_32 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15 = torch.ops.aten.where.self(full_default_30, full_default_32, div_15);  full_default_30 = full_default_32 = div_15 = None
        view_175 = torch.ops.aten.view.default(where_15, [-1, 512, 512]);  where_15 = None
        bmm_15 = torch.ops.aten.bmm.default(view_175, view_172);  view_175 = view_172 = None
        view_176 = torch.ops.aten.view.default(bmm_15, [-1, 24, 512, 64]);  bmm_15 = None
        permute_84 = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
        clone_31 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_177 = torch.ops.aten.view.default(clone_31, [2, 512, -1]);  clone_31 = None
        view_178 = torch.ops.aten.view.default(view_177, [1024, 1536]);  view_177 = None
        permute_85 = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg125_1, view_178, permute_85);  arg125_1 = view_178 = permute_85 = None
        view_179 = torch.ops.aten.view.default(addmm_45, [2, 512, 1536]);  addmm_45 = None
        add_52 = torch.ops.aten.add.Tensor(view_179, add_51);  view_179 = add_51 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_30, 1e-07);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_52, getitem_31);  add_52 = getitem_31 = None
        mul_61 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_61, arg126_1);  mul_61 = arg126_1 = None
        add_54 = torch.ops.aten.add.Tensor(mul_62, arg127_1);  mul_62 = arg127_1 = None
        view_180 = torch.ops.aten.view.default(add_54, [1024, 1536])
        permute_86 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg129_1, view_180, permute_86);  arg129_1 = view_180 = permute_86 = None
        view_181 = torch.ops.aten.view.default(addmm_46, [2, 512, 6144]);  addmm_46 = None
        mul_63 = torch.ops.aten.mul.Tensor(view_181, 0.5)
        mul_64 = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476);  view_181 = None
        erf_7 = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_55 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_63, add_55);  mul_63 = add_55 = None
        view_182 = torch.ops.aten.view.default(mul_65, [1024, 6144]);  mul_65 = None
        permute_87 = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg131_1, view_182, permute_87);  arg131_1 = view_182 = permute_87 = None
        view_183 = torch.ops.aten.view.default(addmm_47, [2, 512, 1536]);  addmm_47 = None
        add_56 = torch.ops.aten.add.Tensor(view_183, add_54);  view_183 = add_54 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_57 = torch.ops.aten.add.Tensor(getitem_32, 1e-07);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_56, getitem_33);  add_56 = getitem_33 = None
        mul_66 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, arg132_1);  mul_66 = arg132_1 = None
        add_58 = torch.ops.aten.add.Tensor(mul_67, arg133_1);  mul_67 = arg133_1 = None
        view_184 = torch.ops.aten.view.default(add_58, [1024, 1536])
        permute_88 = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg135_1, view_184, permute_88);  arg135_1 = view_184 = permute_88 = None
        view_185 = torch.ops.aten.view.default(addmm_48, [2, 512, 1536]);  addmm_48 = None
        view_186 = torch.ops.aten.view.default(view_185, [2, 512, 24, -1]);  view_185 = None
        permute_89 = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        clone_32 = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        view_187 = torch.ops.aten.view.default(clone_32, [-1, 512, 64]);  clone_32 = None
        view_188 = torch.ops.aten.view.default(add_58, [1024, 1536])
        permute_90 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg137_1, view_188, permute_90);  arg137_1 = view_188 = permute_90 = None
        view_189 = torch.ops.aten.view.default(addmm_49, [2, 512, 1536]);  addmm_49 = None
        view_190 = torch.ops.aten.view.default(view_189, [2, 512, 24, -1]);  view_189 = None
        permute_91 = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_33 = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
        view_191 = torch.ops.aten.view.default(clone_33, [-1, 512, 64]);  clone_33 = None
        view_192 = torch.ops.aten.view.default(add_58, [1024, 1536])
        permute_92 = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg139_1, view_192, permute_92);  arg139_1 = view_192 = permute_92 = None
        view_193 = torch.ops.aten.view.default(addmm_50, [2, 512, 1536]);  addmm_50 = None
        view_194 = torch.ops.aten.view.default(view_193, [2, 512, 24, -1]);  view_193 = None
        permute_93 = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        clone_34 = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_195 = torch.ops.aten.view.default(clone_34, [-1, 512, 64]);  clone_34 = None
        _tensor_constant16 = self._tensor_constant16
        lift_fresh_copy_16 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
        mul_68 = torch.ops.aten.mul.Tensor(lift_fresh_copy_16, 1);  lift_fresh_copy_16 = mul_68 = None
        full_default_33 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_94 = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
        div_16 = torch.ops.aten.div.Tensor(permute_94, full_default_33);  permute_94 = full_default_33 = None
        bmm_16 = torch.ops.aten.bmm.default(view_187, div_16);  view_187 = div_16 = None
        view_196 = torch.ops.aten.view.default(bmm_16, [-1, 24, 512, 512]);  bmm_16 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_8 = None
        full_default_34 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant17 = self._tensor_constant17;  _tensor_constant17 = None
        full_default_35 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_16 = torch.ops.aten.where.self(full_default_34, full_default_35, view_196);  full_default_35 = view_196 = None
        amax_8 = torch.ops.aten.amax.default(where_16, [-1], True)
        sub_25 = torch.ops.aten.sub.Tensor(where_16, amax_8);  where_16 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_17 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        full_default_36 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17 = torch.ops.aten.where.self(full_default_34, full_default_36, div_17);  full_default_34 = full_default_36 = div_17 = None
        view_198 = torch.ops.aten.view.default(where_17, [-1, 512, 512]);  where_17 = None
        bmm_17 = torch.ops.aten.bmm.default(view_198, view_195);  view_198 = view_195 = None
        view_199 = torch.ops.aten.view.default(bmm_17, [-1, 24, 512, 64]);  bmm_17 = None
        permute_95 = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
        clone_35 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_200 = torch.ops.aten.view.default(clone_35, [2, 512, -1]);  clone_35 = None
        view_201 = torch.ops.aten.view.default(view_200, [1024, 1536]);  view_200 = None
        permute_96 = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg141_1, view_201, permute_96);  arg141_1 = view_201 = permute_96 = None
        view_202 = torch.ops.aten.view.default(addmm_51, [2, 512, 1536]);  addmm_51 = None
        add_59 = torch.ops.aten.add.Tensor(view_202, add_58);  view_202 = add_58 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_34, 1e-07);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_26 = torch.ops.aten.sub.Tensor(add_59, getitem_35);  add_59 = getitem_35 = None
        mul_69 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_70 = torch.ops.aten.mul.Tensor(mul_69, arg142_1);  mul_69 = arg142_1 = None
        add_61 = torch.ops.aten.add.Tensor(mul_70, arg143_1);  mul_70 = arg143_1 = None
        view_203 = torch.ops.aten.view.default(add_61, [1024, 1536])
        permute_97 = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg145_1, view_203, permute_97);  arg145_1 = view_203 = permute_97 = None
        view_204 = torch.ops.aten.view.default(addmm_52, [2, 512, 6144]);  addmm_52 = None
        mul_71 = torch.ops.aten.mul.Tensor(view_204, 0.5)
        mul_72 = torch.ops.aten.mul.Tensor(view_204, 0.7071067811865476);  view_204 = None
        erf_8 = torch.ops.aten.erf.default(mul_72);  mul_72 = None
        add_62 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_71, add_62);  mul_71 = add_62 = None
        view_205 = torch.ops.aten.view.default(mul_73, [1024, 6144]);  mul_73 = None
        permute_98 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg147_1, view_205, permute_98);  arg147_1 = view_205 = permute_98 = None
        view_206 = torch.ops.aten.view.default(addmm_53, [2, 512, 1536]);  addmm_53 = None
        add_63 = torch.ops.aten.add.Tensor(view_206, add_61);  view_206 = add_61 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_36, 1e-07);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_63, getitem_37);  add_63 = getitem_37 = None
        mul_74 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, arg148_1);  mul_74 = arg148_1 = None
        add_65 = torch.ops.aten.add.Tensor(mul_75, arg149_1);  mul_75 = arg149_1 = None
        view_207 = torch.ops.aten.view.default(add_65, [1024, 1536])
        permute_99 = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg151_1, view_207, permute_99);  arg151_1 = view_207 = permute_99 = None
        view_208 = torch.ops.aten.view.default(addmm_54, [2, 512, 1536]);  addmm_54 = None
        view_209 = torch.ops.aten.view.default(view_208, [2, 512, 24, -1]);  view_208 = None
        permute_100 = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
        clone_36 = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_210 = torch.ops.aten.view.default(clone_36, [-1, 512, 64]);  clone_36 = None
        view_211 = torch.ops.aten.view.default(add_65, [1024, 1536])
        permute_101 = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg153_1, view_211, permute_101);  arg153_1 = view_211 = permute_101 = None
        view_212 = torch.ops.aten.view.default(addmm_55, [2, 512, 1536]);  addmm_55 = None
        view_213 = torch.ops.aten.view.default(view_212, [2, 512, 24, -1]);  view_212 = None
        permute_102 = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
        clone_37 = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_214 = torch.ops.aten.view.default(clone_37, [-1, 512, 64]);  clone_37 = None
        view_215 = torch.ops.aten.view.default(add_65, [1024, 1536])
        permute_103 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg155_1, view_215, permute_103);  arg155_1 = view_215 = permute_103 = None
        view_216 = torch.ops.aten.view.default(addmm_56, [2, 512, 1536]);  addmm_56 = None
        view_217 = torch.ops.aten.view.default(view_216, [2, 512, 24, -1]);  view_216 = None
        permute_104 = torch.ops.aten.permute.default(view_217, [0, 2, 1, 3]);  view_217 = None
        clone_38 = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_218 = torch.ops.aten.view.default(clone_38, [-1, 512, 64]);  clone_38 = None
        _tensor_constant18 = self._tensor_constant18
        lift_fresh_copy_18 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
        mul_76 = torch.ops.aten.mul.Tensor(lift_fresh_copy_18, 1);  lift_fresh_copy_18 = mul_76 = None
        full_default_37 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_105 = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None
        div_18 = torch.ops.aten.div.Tensor(permute_105, full_default_37);  permute_105 = full_default_37 = None
        bmm_18 = torch.ops.aten.bmm.default(view_210, div_18);  view_210 = div_18 = None
        view_219 = torch.ops.aten.view.default(bmm_18, [-1, 24, 512, 512]);  bmm_18 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_9 = None
        full_default_38 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant19 = self._tensor_constant19;  _tensor_constant19 = None
        full_default_39 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_18 = torch.ops.aten.where.self(full_default_38, full_default_39, view_219);  full_default_39 = view_219 = None
        amax_9 = torch.ops.aten.amax.default(where_18, [-1], True)
        sub_28 = torch.ops.aten.sub.Tensor(where_18, amax_9);  where_18 = amax_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_19 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        full_default_40 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19 = torch.ops.aten.where.self(full_default_38, full_default_40, div_19);  full_default_38 = full_default_40 = div_19 = None
        view_221 = torch.ops.aten.view.default(where_19, [-1, 512, 512]);  where_19 = None
        bmm_19 = torch.ops.aten.bmm.default(view_221, view_218);  view_221 = view_218 = None
        view_222 = torch.ops.aten.view.default(bmm_19, [-1, 24, 512, 64]);  bmm_19 = None
        permute_106 = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        clone_39 = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_223 = torch.ops.aten.view.default(clone_39, [2, 512, -1]);  clone_39 = None
        view_224 = torch.ops.aten.view.default(view_223, [1024, 1536]);  view_223 = None
        permute_107 = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg157_1, view_224, permute_107);  arg157_1 = view_224 = permute_107 = None
        view_225 = torch.ops.aten.view.default(addmm_57, [2, 512, 1536]);  addmm_57 = None
        add_66 = torch.ops.aten.add.Tensor(view_225, add_65);  view_225 = add_65 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_67 = torch.ops.aten.add.Tensor(getitem_38, 1e-07);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_66, getitem_39);  add_66 = getitem_39 = None
        mul_77 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_77, arg158_1);  mul_77 = arg158_1 = None
        add_68 = torch.ops.aten.add.Tensor(mul_78, arg159_1);  mul_78 = arg159_1 = None
        view_226 = torch.ops.aten.view.default(add_68, [1024, 1536])
        permute_108 = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg161_1, view_226, permute_108);  arg161_1 = view_226 = permute_108 = None
        view_227 = torch.ops.aten.view.default(addmm_58, [2, 512, 6144]);  addmm_58 = None
        mul_79 = torch.ops.aten.mul.Tensor(view_227, 0.5)
        mul_80 = torch.ops.aten.mul.Tensor(view_227, 0.7071067811865476);  view_227 = None
        erf_9 = torch.ops.aten.erf.default(mul_80);  mul_80 = None
        add_69 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_79, add_69);  mul_79 = add_69 = None
        view_228 = torch.ops.aten.view.default(mul_81, [1024, 6144]);  mul_81 = None
        permute_109 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg163_1, view_228, permute_109);  arg163_1 = view_228 = permute_109 = None
        view_229 = torch.ops.aten.view.default(addmm_59, [2, 512, 1536]);  addmm_59 = None
        add_70 = torch.ops.aten.add.Tensor(view_229, add_68);  view_229 = add_68 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_71 = torch.ops.aten.add.Tensor(getitem_40, 1e-07);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_70, getitem_41);  add_70 = getitem_41 = None
        mul_82 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, arg164_1);  mul_82 = arg164_1 = None
        add_72 = torch.ops.aten.add.Tensor(mul_83, arg165_1);  mul_83 = arg165_1 = None
        view_230 = torch.ops.aten.view.default(add_72, [1024, 1536])
        permute_110 = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg167_1, view_230, permute_110);  arg167_1 = view_230 = permute_110 = None
        view_231 = torch.ops.aten.view.default(addmm_60, [2, 512, 1536]);  addmm_60 = None
        view_232 = torch.ops.aten.view.default(view_231, [2, 512, 24, -1]);  view_231 = None
        permute_111 = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
        clone_40 = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        view_233 = torch.ops.aten.view.default(clone_40, [-1, 512, 64]);  clone_40 = None
        view_234 = torch.ops.aten.view.default(add_72, [1024, 1536])
        permute_112 = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg169_1, view_234, permute_112);  arg169_1 = view_234 = permute_112 = None
        view_235 = torch.ops.aten.view.default(addmm_61, [2, 512, 1536]);  addmm_61 = None
        view_236 = torch.ops.aten.view.default(view_235, [2, 512, 24, -1]);  view_235 = None
        permute_113 = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        clone_41 = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        view_237 = torch.ops.aten.view.default(clone_41, [-1, 512, 64]);  clone_41 = None
        view_238 = torch.ops.aten.view.default(add_72, [1024, 1536])
        permute_114 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg171_1, view_238, permute_114);  arg171_1 = view_238 = permute_114 = None
        view_239 = torch.ops.aten.view.default(addmm_62, [2, 512, 1536]);  addmm_62 = None
        view_240 = torch.ops.aten.view.default(view_239, [2, 512, 24, -1]);  view_239 = None
        permute_115 = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
        clone_42 = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_241 = torch.ops.aten.view.default(clone_42, [-1, 512, 64]);  clone_42 = None
        _tensor_constant20 = self._tensor_constant20
        lift_fresh_copy_20 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant20);  _tensor_constant20 = None
        mul_84 = torch.ops.aten.mul.Tensor(lift_fresh_copy_20, 1);  lift_fresh_copy_20 = mul_84 = None
        full_default_41 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_116 = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
        div_20 = torch.ops.aten.div.Tensor(permute_116, full_default_41);  permute_116 = full_default_41 = None
        bmm_20 = torch.ops.aten.bmm.default(view_233, div_20);  view_233 = div_20 = None
        view_242 = torch.ops.aten.view.default(bmm_20, [-1, 24, 512, 512]);  bmm_20 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_10 = None
        full_default_42 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant21 = self._tensor_constant21;  _tensor_constant21 = None
        full_default_43 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_20 = torch.ops.aten.where.self(full_default_42, full_default_43, view_242);  full_default_43 = view_242 = None
        amax_10 = torch.ops.aten.amax.default(where_20, [-1], True)
        sub_31 = torch.ops.aten.sub.Tensor(where_20, amax_10);  where_20 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_21 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        full_default_44 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21 = torch.ops.aten.where.self(full_default_42, full_default_44, div_21);  full_default_42 = full_default_44 = div_21 = None
        view_244 = torch.ops.aten.view.default(where_21, [-1, 512, 512]);  where_21 = None
        bmm_21 = torch.ops.aten.bmm.default(view_244, view_241);  view_244 = view_241 = None
        view_245 = torch.ops.aten.view.default(bmm_21, [-1, 24, 512, 64]);  bmm_21 = None
        permute_117 = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
        clone_43 = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_246 = torch.ops.aten.view.default(clone_43, [2, 512, -1]);  clone_43 = None
        view_247 = torch.ops.aten.view.default(view_246, [1024, 1536]);  view_246 = None
        permute_118 = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg173_1, view_247, permute_118);  arg173_1 = view_247 = permute_118 = None
        view_248 = torch.ops.aten.view.default(addmm_63, [2, 512, 1536]);  addmm_63 = None
        add_73 = torch.ops.aten.add.Tensor(view_248, add_72);  view_248 = add_72 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_42, 1e-07);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_32 = torch.ops.aten.sub.Tensor(add_73, getitem_43);  add_73 = getitem_43 = None
        mul_85 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_85, arg174_1);  mul_85 = arg174_1 = None
        add_75 = torch.ops.aten.add.Tensor(mul_86, arg175_1);  mul_86 = arg175_1 = None
        view_249 = torch.ops.aten.view.default(add_75, [1024, 1536])
        permute_119 = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg177_1, view_249, permute_119);  arg177_1 = view_249 = permute_119 = None
        view_250 = torch.ops.aten.view.default(addmm_64, [2, 512, 6144]);  addmm_64 = None
        mul_87 = torch.ops.aten.mul.Tensor(view_250, 0.5)
        mul_88 = torch.ops.aten.mul.Tensor(view_250, 0.7071067811865476);  view_250 = None
        erf_10 = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_76 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_87, add_76);  mul_87 = add_76 = None
        view_251 = torch.ops.aten.view.default(mul_89, [1024, 6144]);  mul_89 = None
        permute_120 = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg179_1, view_251, permute_120);  arg179_1 = view_251 = permute_120 = None
        view_252 = torch.ops.aten.view.default(addmm_65, [2, 512, 1536]);  addmm_65 = None
        add_77 = torch.ops.aten.add.Tensor(view_252, add_75);  view_252 = add_75 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_78 = torch.ops.aten.add.Tensor(getitem_44, 1e-07);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_77, getitem_45);  add_77 = getitem_45 = None
        mul_90 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_90, arg180_1);  mul_90 = arg180_1 = None
        add_79 = torch.ops.aten.add.Tensor(mul_91, arg181_1);  mul_91 = arg181_1 = None
        view_253 = torch.ops.aten.view.default(add_79, [1024, 1536])
        permute_121 = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg183_1, view_253, permute_121);  arg183_1 = view_253 = permute_121 = None
        view_254 = torch.ops.aten.view.default(addmm_66, [2, 512, 1536]);  addmm_66 = None
        view_255 = torch.ops.aten.view.default(view_254, [2, 512, 24, -1]);  view_254 = None
        permute_122 = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
        clone_44 = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_256 = torch.ops.aten.view.default(clone_44, [-1, 512, 64]);  clone_44 = None
        view_257 = torch.ops.aten.view.default(add_79, [1024, 1536])
        permute_123 = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg185_1, view_257, permute_123);  arg185_1 = view_257 = permute_123 = None
        view_258 = torch.ops.aten.view.default(addmm_67, [2, 512, 1536]);  addmm_67 = None
        view_259 = torch.ops.aten.view.default(view_258, [2, 512, 24, -1]);  view_258 = None
        permute_124 = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
        clone_45 = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_260 = torch.ops.aten.view.default(clone_45, [-1, 512, 64]);  clone_45 = None
        view_261 = torch.ops.aten.view.default(add_79, [1024, 1536])
        permute_125 = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg187_1, view_261, permute_125);  arg187_1 = view_261 = permute_125 = None
        view_262 = torch.ops.aten.view.default(addmm_68, [2, 512, 1536]);  addmm_68 = None
        view_263 = torch.ops.aten.view.default(view_262, [2, 512, 24, -1]);  view_262 = None
        permute_126 = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
        clone_46 = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_264 = torch.ops.aten.view.default(clone_46, [-1, 512, 64]);  clone_46 = None
        _tensor_constant22 = self._tensor_constant22
        lift_fresh_copy_22 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant22);  _tensor_constant22 = None
        mul_92 = torch.ops.aten.mul.Tensor(lift_fresh_copy_22, 1);  lift_fresh_copy_22 = mul_92 = None
        full_default_45 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_127 = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None
        div_22 = torch.ops.aten.div.Tensor(permute_127, full_default_45);  permute_127 = full_default_45 = None
        bmm_22 = torch.ops.aten.bmm.default(view_256, div_22);  view_256 = div_22 = None
        view_265 = torch.ops.aten.view.default(bmm_22, [-1, 24, 512, 512]);  bmm_22 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_11 = None
        full_default_46 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant23 = self._tensor_constant23;  _tensor_constant23 = None
        full_default_47 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_22 = torch.ops.aten.where.self(full_default_46, full_default_47, view_265);  full_default_47 = view_265 = None
        amax_11 = torch.ops.aten.amax.default(where_22, [-1], True)
        sub_34 = torch.ops.aten.sub.Tensor(where_22, amax_11);  where_22 = amax_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_23 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        full_default_48 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23 = torch.ops.aten.where.self(full_default_46, full_default_48, div_23);  full_default_46 = full_default_48 = div_23 = None
        view_267 = torch.ops.aten.view.default(where_23, [-1, 512, 512]);  where_23 = None
        bmm_23 = torch.ops.aten.bmm.default(view_267, view_264);  view_267 = view_264 = None
        view_268 = torch.ops.aten.view.default(bmm_23, [-1, 24, 512, 64]);  bmm_23 = None
        permute_128 = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
        clone_47 = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_269 = torch.ops.aten.view.default(clone_47, [2, 512, -1]);  clone_47 = None
        view_270 = torch.ops.aten.view.default(view_269, [1024, 1536]);  view_269 = None
        permute_129 = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg189_1, view_270, permute_129);  arg189_1 = view_270 = permute_129 = None
        view_271 = torch.ops.aten.view.default(addmm_69, [2, 512, 1536]);  addmm_69 = None
        add_80 = torch.ops.aten.add.Tensor(view_271, add_79);  view_271 = add_79 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_81 = torch.ops.aten.add.Tensor(getitem_46, 1e-07);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_35 = torch.ops.aten.sub.Tensor(add_80, getitem_47);  add_80 = getitem_47 = None
        mul_93 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_94 = torch.ops.aten.mul.Tensor(mul_93, arg190_1);  mul_93 = arg190_1 = None
        add_82 = torch.ops.aten.add.Tensor(mul_94, arg191_1);  mul_94 = arg191_1 = None
        view_272 = torch.ops.aten.view.default(add_82, [1024, 1536])
        permute_130 = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg193_1, view_272, permute_130);  arg193_1 = view_272 = permute_130 = None
        view_273 = torch.ops.aten.view.default(addmm_70, [2, 512, 6144]);  addmm_70 = None
        mul_95 = torch.ops.aten.mul.Tensor(view_273, 0.5)
        mul_96 = torch.ops.aten.mul.Tensor(view_273, 0.7071067811865476);  view_273 = None
        erf_11 = torch.ops.aten.erf.default(mul_96);  mul_96 = None
        add_83 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_95, add_83);  mul_95 = add_83 = None
        view_274 = torch.ops.aten.view.default(mul_97, [1024, 6144]);  mul_97 = None
        permute_131 = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg195_1, view_274, permute_131);  arg195_1 = view_274 = permute_131 = None
        view_275 = torch.ops.aten.view.default(addmm_71, [2, 512, 1536]);  addmm_71 = None
        add_84 = torch.ops.aten.add.Tensor(view_275, add_82);  view_275 = add_82 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_48, 1e-07);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_84, getitem_49);  add_84 = getitem_49 = None
        mul_98 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_99 = torch.ops.aten.mul.Tensor(mul_98, arg196_1);  mul_98 = arg196_1 = None
        add_86 = torch.ops.aten.add.Tensor(mul_99, arg197_1);  mul_99 = arg197_1 = None
        view_276 = torch.ops.aten.view.default(add_86, [1024, 1536])
        permute_132 = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg199_1, view_276, permute_132);  arg199_1 = view_276 = permute_132 = None
        view_277 = torch.ops.aten.view.default(addmm_72, [2, 512, 1536]);  addmm_72 = None
        view_278 = torch.ops.aten.view.default(view_277, [2, 512, 24, -1]);  view_277 = None
        permute_133 = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
        clone_48 = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
        view_279 = torch.ops.aten.view.default(clone_48, [-1, 512, 64]);  clone_48 = None
        view_280 = torch.ops.aten.view.default(add_86, [1024, 1536])
        permute_134 = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg201_1, view_280, permute_134);  arg201_1 = view_280 = permute_134 = None
        view_281 = torch.ops.aten.view.default(addmm_73, [2, 512, 1536]);  addmm_73 = None
        view_282 = torch.ops.aten.view.default(view_281, [2, 512, 24, -1]);  view_281 = None
        permute_135 = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
        clone_49 = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_283 = torch.ops.aten.view.default(clone_49, [-1, 512, 64]);  clone_49 = None
        view_284 = torch.ops.aten.view.default(add_86, [1024, 1536])
        permute_136 = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg203_1, view_284, permute_136);  arg203_1 = view_284 = permute_136 = None
        view_285 = torch.ops.aten.view.default(addmm_74, [2, 512, 1536]);  addmm_74 = None
        view_286 = torch.ops.aten.view.default(view_285, [2, 512, 24, -1]);  view_285 = None
        permute_137 = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
        clone_50 = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_287 = torch.ops.aten.view.default(clone_50, [-1, 512, 64]);  clone_50 = None
        _tensor_constant24 = self._tensor_constant24
        lift_fresh_copy_24 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant24);  _tensor_constant24 = None
        mul_100 = torch.ops.aten.mul.Tensor(lift_fresh_copy_24, 1);  lift_fresh_copy_24 = mul_100 = None
        full_default_49 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_138 = torch.ops.aten.permute.default(view_283, [0, 2, 1]);  view_283 = None
        div_24 = torch.ops.aten.div.Tensor(permute_138, full_default_49);  permute_138 = full_default_49 = None
        bmm_24 = torch.ops.aten.bmm.default(view_279, div_24);  view_279 = div_24 = None
        view_288 = torch.ops.aten.view.default(bmm_24, [-1, 24, 512, 512]);  bmm_24 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_12 = None
        full_default_50 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant25 = self._tensor_constant25;  _tensor_constant25 = None
        full_default_51 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_24 = torch.ops.aten.where.self(full_default_50, full_default_51, view_288);  full_default_51 = view_288 = None
        amax_12 = torch.ops.aten.amax.default(where_24, [-1], True)
        sub_37 = torch.ops.aten.sub.Tensor(where_24, amax_12);  where_24 = amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_25 = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        full_default_52 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25 = torch.ops.aten.where.self(full_default_50, full_default_52, div_25);  full_default_50 = full_default_52 = div_25 = None
        view_290 = torch.ops.aten.view.default(where_25, [-1, 512, 512]);  where_25 = None
        bmm_25 = torch.ops.aten.bmm.default(view_290, view_287);  view_290 = view_287 = None
        view_291 = torch.ops.aten.view.default(bmm_25, [-1, 24, 512, 64]);  bmm_25 = None
        permute_139 = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        clone_51 = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_292 = torch.ops.aten.view.default(clone_51, [2, 512, -1]);  clone_51 = None
        view_293 = torch.ops.aten.view.default(view_292, [1024, 1536]);  view_292 = None
        permute_140 = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg205_1, view_293, permute_140);  arg205_1 = view_293 = permute_140 = None
        view_294 = torch.ops.aten.view.default(addmm_75, [2, 512, 1536]);  addmm_75 = None
        add_87 = torch.ops.aten.add.Tensor(view_294, add_86);  view_294 = add_86 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_88 = torch.ops.aten.add.Tensor(getitem_50, 1e-07);  getitem_50 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_38 = torch.ops.aten.sub.Tensor(add_87, getitem_51);  add_87 = getitem_51 = None
        mul_101 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
        mul_102 = torch.ops.aten.mul.Tensor(mul_101, arg206_1);  mul_101 = arg206_1 = None
        add_89 = torch.ops.aten.add.Tensor(mul_102, arg207_1);  mul_102 = arg207_1 = None
        view_295 = torch.ops.aten.view.default(add_89, [1024, 1536])
        permute_141 = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg209_1, view_295, permute_141);  arg209_1 = view_295 = permute_141 = None
        view_296 = torch.ops.aten.view.default(addmm_76, [2, 512, 6144]);  addmm_76 = None
        mul_103 = torch.ops.aten.mul.Tensor(view_296, 0.5)
        mul_104 = torch.ops.aten.mul.Tensor(view_296, 0.7071067811865476);  view_296 = None
        erf_12 = torch.ops.aten.erf.default(mul_104);  mul_104 = None
        add_90 = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_103, add_90);  mul_103 = add_90 = None
        view_297 = torch.ops.aten.view.default(mul_105, [1024, 6144]);  mul_105 = None
        permute_142 = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg211_1, view_297, permute_142);  arg211_1 = view_297 = permute_142 = None
        view_298 = torch.ops.aten.view.default(addmm_77, [2, 512, 1536]);  addmm_77 = None
        add_91 = torch.ops.aten.add.Tensor(view_298, add_89);  view_298 = add_89 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_52 = var_mean_26[0]
        getitem_53 = var_mean_26[1];  var_mean_26 = None
        add_92 = torch.ops.aten.add.Tensor(getitem_52, 1e-07);  getitem_52 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_91, getitem_53);  add_91 = getitem_53 = None
        mul_106 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
        mul_107 = torch.ops.aten.mul.Tensor(mul_106, arg212_1);  mul_106 = arg212_1 = None
        add_93 = torch.ops.aten.add.Tensor(mul_107, arg213_1);  mul_107 = arg213_1 = None
        view_299 = torch.ops.aten.view.default(add_93, [1024, 1536])
        permute_143 = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg215_1, view_299, permute_143);  arg215_1 = view_299 = permute_143 = None
        view_300 = torch.ops.aten.view.default(addmm_78, [2, 512, 1536]);  addmm_78 = None
        view_301 = torch.ops.aten.view.default(view_300, [2, 512, 24, -1]);  view_300 = None
        permute_144 = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None
        clone_52 = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_302 = torch.ops.aten.view.default(clone_52, [-1, 512, 64]);  clone_52 = None
        view_303 = torch.ops.aten.view.default(add_93, [1024, 1536])
        permute_145 = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg217_1, view_303, permute_145);  arg217_1 = view_303 = permute_145 = None
        view_304 = torch.ops.aten.view.default(addmm_79, [2, 512, 1536]);  addmm_79 = None
        view_305 = torch.ops.aten.view.default(view_304, [2, 512, 24, -1]);  view_304 = None
        permute_146 = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
        clone_53 = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
        view_306 = torch.ops.aten.view.default(clone_53, [-1, 512, 64]);  clone_53 = None
        view_307 = torch.ops.aten.view.default(add_93, [1024, 1536])
        permute_147 = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg219_1, view_307, permute_147);  arg219_1 = view_307 = permute_147 = None
        view_308 = torch.ops.aten.view.default(addmm_80, [2, 512, 1536]);  addmm_80 = None
        view_309 = torch.ops.aten.view.default(view_308, [2, 512, 24, -1]);  view_308 = None
        permute_148 = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None
        clone_54 = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
        view_310 = torch.ops.aten.view.default(clone_54, [-1, 512, 64]);  clone_54 = None
        _tensor_constant26 = self._tensor_constant26
        lift_fresh_copy_26 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant26);  _tensor_constant26 = None
        mul_108 = torch.ops.aten.mul.Tensor(lift_fresh_copy_26, 1);  lift_fresh_copy_26 = mul_108 = None
        full_default_53 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_149 = torch.ops.aten.permute.default(view_306, [0, 2, 1]);  view_306 = None
        div_26 = torch.ops.aten.div.Tensor(permute_149, full_default_53);  permute_149 = full_default_53 = None
        bmm_26 = torch.ops.aten.bmm.default(view_302, div_26);  view_302 = div_26 = None
        view_311 = torch.ops.aten.view.default(bmm_26, [-1, 24, 512, 512]);  bmm_26 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_13 = None
        full_default_54 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant27 = self._tensor_constant27;  _tensor_constant27 = None
        full_default_55 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_26 = torch.ops.aten.where.self(full_default_54, full_default_55, view_311);  full_default_55 = view_311 = None
        amax_13 = torch.ops.aten.amax.default(where_26, [-1], True)
        sub_40 = torch.ops.aten.sub.Tensor(where_26, amax_13);  where_26 = amax_13 = None
        exp_13 = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_27 = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        full_default_56 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27 = torch.ops.aten.where.self(full_default_54, full_default_56, div_27);  full_default_54 = full_default_56 = div_27 = None
        view_313 = torch.ops.aten.view.default(where_27, [-1, 512, 512]);  where_27 = None
        bmm_27 = torch.ops.aten.bmm.default(view_313, view_310);  view_313 = view_310 = None
        view_314 = torch.ops.aten.view.default(bmm_27, [-1, 24, 512, 64]);  bmm_27 = None
        permute_150 = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
        clone_55 = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_315 = torch.ops.aten.view.default(clone_55, [2, 512, -1]);  clone_55 = None
        view_316 = torch.ops.aten.view.default(view_315, [1024, 1536]);  view_315 = None
        permute_151 = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg221_1, view_316, permute_151);  arg221_1 = view_316 = permute_151 = None
        view_317 = torch.ops.aten.view.default(addmm_81, [2, 512, 1536]);  addmm_81 = None
        add_94 = torch.ops.aten.add.Tensor(view_317, add_93);  view_317 = add_93 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_27[0]
        getitem_55 = var_mean_27[1];  var_mean_27 = None
        add_95 = torch.ops.aten.add.Tensor(getitem_54, 1e-07);  getitem_54 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        sub_41 = torch.ops.aten.sub.Tensor(add_94, getitem_55);  add_94 = getitem_55 = None
        mul_109 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
        mul_110 = torch.ops.aten.mul.Tensor(mul_109, arg222_1);  mul_109 = arg222_1 = None
        add_96 = torch.ops.aten.add.Tensor(mul_110, arg223_1);  mul_110 = arg223_1 = None
        view_318 = torch.ops.aten.view.default(add_96, [1024, 1536])
        permute_152 = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg225_1, view_318, permute_152);  arg225_1 = view_318 = permute_152 = None
        view_319 = torch.ops.aten.view.default(addmm_82, [2, 512, 6144]);  addmm_82 = None
        mul_111 = torch.ops.aten.mul.Tensor(view_319, 0.5)
        mul_112 = torch.ops.aten.mul.Tensor(view_319, 0.7071067811865476);  view_319 = None
        erf_13 = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_97 = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_111, add_97);  mul_111 = add_97 = None
        view_320 = torch.ops.aten.view.default(mul_113, [1024, 6144]);  mul_113 = None
        permute_153 = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg227_1, view_320, permute_153);  arg227_1 = view_320 = permute_153 = None
        view_321 = torch.ops.aten.view.default(addmm_83, [2, 512, 1536]);  addmm_83 = None
        add_98 = torch.ops.aten.add.Tensor(view_321, add_96);  view_321 = add_96 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_28[0]
        getitem_57 = var_mean_28[1];  var_mean_28 = None
        add_99 = torch.ops.aten.add.Tensor(getitem_56, 1e-07);  getitem_56 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_42 = torch.ops.aten.sub.Tensor(add_98, getitem_57);  add_98 = getitem_57 = None
        mul_114 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
        mul_115 = torch.ops.aten.mul.Tensor(mul_114, arg228_1);  mul_114 = arg228_1 = None
        add_100 = torch.ops.aten.add.Tensor(mul_115, arg229_1);  mul_115 = arg229_1 = None
        view_322 = torch.ops.aten.view.default(add_100, [1024, 1536])
        permute_154 = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg231_1, view_322, permute_154);  arg231_1 = view_322 = permute_154 = None
        view_323 = torch.ops.aten.view.default(addmm_84, [2, 512, 1536]);  addmm_84 = None
        view_324 = torch.ops.aten.view.default(view_323, [2, 512, 24, -1]);  view_323 = None
        permute_155 = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
        clone_56 = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_325 = torch.ops.aten.view.default(clone_56, [-1, 512, 64]);  clone_56 = None
        view_326 = torch.ops.aten.view.default(add_100, [1024, 1536])
        permute_156 = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg233_1, view_326, permute_156);  arg233_1 = view_326 = permute_156 = None
        view_327 = torch.ops.aten.view.default(addmm_85, [2, 512, 1536]);  addmm_85 = None
        view_328 = torch.ops.aten.view.default(view_327, [2, 512, 24, -1]);  view_327 = None
        permute_157 = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
        clone_57 = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
        view_329 = torch.ops.aten.view.default(clone_57, [-1, 512, 64]);  clone_57 = None
        view_330 = torch.ops.aten.view.default(add_100, [1024, 1536])
        permute_158 = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg235_1, view_330, permute_158);  arg235_1 = view_330 = permute_158 = None
        view_331 = torch.ops.aten.view.default(addmm_86, [2, 512, 1536]);  addmm_86 = None
        view_332 = torch.ops.aten.view.default(view_331, [2, 512, 24, -1]);  view_331 = None
        permute_159 = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        clone_58 = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_333 = torch.ops.aten.view.default(clone_58, [-1, 512, 64]);  clone_58 = None
        _tensor_constant28 = self._tensor_constant28
        lift_fresh_copy_28 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant28);  _tensor_constant28 = None
        mul_116 = torch.ops.aten.mul.Tensor(lift_fresh_copy_28, 1);  lift_fresh_copy_28 = mul_116 = None
        full_default_57 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_160 = torch.ops.aten.permute.default(view_329, [0, 2, 1]);  view_329 = None
        div_28 = torch.ops.aten.div.Tensor(permute_160, full_default_57);  permute_160 = full_default_57 = None
        bmm_28 = torch.ops.aten.bmm.default(view_325, div_28);  view_325 = div_28 = None
        view_334 = torch.ops.aten.view.default(bmm_28, [-1, 24, 512, 512]);  bmm_28 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_14 = None
        full_default_58 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant29 = self._tensor_constant29;  _tensor_constant29 = None
        full_default_59 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_28 = torch.ops.aten.where.self(full_default_58, full_default_59, view_334);  full_default_59 = view_334 = None
        amax_14 = torch.ops.aten.amax.default(where_28, [-1], True)
        sub_43 = torch.ops.aten.sub.Tensor(where_28, amax_14);  where_28 = amax_14 = None
        exp_14 = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_29 = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        full_default_60 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_29 = torch.ops.aten.where.self(full_default_58, full_default_60, div_29);  full_default_58 = full_default_60 = div_29 = None
        view_336 = torch.ops.aten.view.default(where_29, [-1, 512, 512]);  where_29 = None
        bmm_29 = torch.ops.aten.bmm.default(view_336, view_333);  view_336 = view_333 = None
        view_337 = torch.ops.aten.view.default(bmm_29, [-1, 24, 512, 64]);  bmm_29 = None
        permute_161 = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
        clone_59 = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_338 = torch.ops.aten.view.default(clone_59, [2, 512, -1]);  clone_59 = None
        view_339 = torch.ops.aten.view.default(view_338, [1024, 1536]);  view_338 = None
        permute_162 = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg237_1, view_339, permute_162);  arg237_1 = view_339 = permute_162 = None
        view_340 = torch.ops.aten.view.default(addmm_87, [2, 512, 1536]);  addmm_87 = None
        add_101 = torch.ops.aten.add.Tensor(view_340, add_100);  view_340 = add_100 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_58 = var_mean_29[0]
        getitem_59 = var_mean_29[1];  var_mean_29 = None
        add_102 = torch.ops.aten.add.Tensor(getitem_58, 1e-07);  getitem_58 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_44 = torch.ops.aten.sub.Tensor(add_101, getitem_59);  add_101 = getitem_59 = None
        mul_117 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
        mul_118 = torch.ops.aten.mul.Tensor(mul_117, arg238_1);  mul_117 = arg238_1 = None
        add_103 = torch.ops.aten.add.Tensor(mul_118, arg239_1);  mul_118 = arg239_1 = None
        view_341 = torch.ops.aten.view.default(add_103, [1024, 1536])
        permute_163 = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg241_1, view_341, permute_163);  arg241_1 = view_341 = permute_163 = None
        view_342 = torch.ops.aten.view.default(addmm_88, [2, 512, 6144]);  addmm_88 = None
        mul_119 = torch.ops.aten.mul.Tensor(view_342, 0.5)
        mul_120 = torch.ops.aten.mul.Tensor(view_342, 0.7071067811865476);  view_342 = None
        erf_14 = torch.ops.aten.erf.default(mul_120);  mul_120 = None
        add_104 = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_119, add_104);  mul_119 = add_104 = None
        view_343 = torch.ops.aten.view.default(mul_121, [1024, 6144]);  mul_121 = None
        permute_164 = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg243_1, view_343, permute_164);  arg243_1 = view_343 = permute_164 = None
        view_344 = torch.ops.aten.view.default(addmm_89, [2, 512, 1536]);  addmm_89 = None
        add_105 = torch.ops.aten.add.Tensor(view_344, add_103);  view_344 = add_103 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_60 = var_mean_30[0]
        getitem_61 = var_mean_30[1];  var_mean_30 = None
        add_106 = torch.ops.aten.add.Tensor(getitem_60, 1e-07);  getitem_60 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_45 = torch.ops.aten.sub.Tensor(add_105, getitem_61);  add_105 = getitem_61 = None
        mul_122 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = rsqrt_30 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_122, arg244_1);  mul_122 = arg244_1 = None
        add_107 = torch.ops.aten.add.Tensor(mul_123, arg245_1);  mul_123 = arg245_1 = None
        view_345 = torch.ops.aten.view.default(add_107, [1024, 1536])
        permute_165 = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg247_1, view_345, permute_165);  arg247_1 = view_345 = permute_165 = None
        view_346 = torch.ops.aten.view.default(addmm_90, [2, 512, 1536]);  addmm_90 = None
        view_347 = torch.ops.aten.view.default(view_346, [2, 512, 24, -1]);  view_346 = None
        permute_166 = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
        clone_60 = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
        view_348 = torch.ops.aten.view.default(clone_60, [-1, 512, 64]);  clone_60 = None
        view_349 = torch.ops.aten.view.default(add_107, [1024, 1536])
        permute_167 = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg249_1, view_349, permute_167);  arg249_1 = view_349 = permute_167 = None
        view_350 = torch.ops.aten.view.default(addmm_91, [2, 512, 1536]);  addmm_91 = None
        view_351 = torch.ops.aten.view.default(view_350, [2, 512, 24, -1]);  view_350 = None
        permute_168 = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
        clone_61 = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_352 = torch.ops.aten.view.default(clone_61, [-1, 512, 64]);  clone_61 = None
        view_353 = torch.ops.aten.view.default(add_107, [1024, 1536])
        permute_169 = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg251_1, view_353, permute_169);  arg251_1 = view_353 = permute_169 = None
        view_354 = torch.ops.aten.view.default(addmm_92, [2, 512, 1536]);  addmm_92 = None
        view_355 = torch.ops.aten.view.default(view_354, [2, 512, 24, -1]);  view_354 = None
        permute_170 = torch.ops.aten.permute.default(view_355, [0, 2, 1, 3]);  view_355 = None
        clone_62 = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_356 = torch.ops.aten.view.default(clone_62, [-1, 512, 64]);  clone_62 = None
        _tensor_constant30 = self._tensor_constant30
        lift_fresh_copy_30 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant30);  _tensor_constant30 = None
        mul_124 = torch.ops.aten.mul.Tensor(lift_fresh_copy_30, 1);  lift_fresh_copy_30 = mul_124 = None
        full_default_61 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_171 = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None
        div_30 = torch.ops.aten.div.Tensor(permute_171, full_default_61);  permute_171 = full_default_61 = None
        bmm_30 = torch.ops.aten.bmm.default(view_348, div_30);  view_348 = div_30 = None
        view_357 = torch.ops.aten.view.default(bmm_30, [-1, 24, 512, 512]);  bmm_30 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_15 = None
        full_default_62 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant31 = self._tensor_constant31;  _tensor_constant31 = None
        full_default_63 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_30 = torch.ops.aten.where.self(full_default_62, full_default_63, view_357);  full_default_63 = view_357 = None
        amax_15 = torch.ops.aten.amax.default(where_30, [-1], True)
        sub_46 = torch.ops.aten.sub.Tensor(where_30, amax_15);  where_30 = amax_15 = None
        exp_15 = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_31 = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        full_default_64 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_31 = torch.ops.aten.where.self(full_default_62, full_default_64, div_31);  full_default_62 = full_default_64 = div_31 = None
        view_359 = torch.ops.aten.view.default(where_31, [-1, 512, 512]);  where_31 = None
        bmm_31 = torch.ops.aten.bmm.default(view_359, view_356);  view_359 = view_356 = None
        view_360 = torch.ops.aten.view.default(bmm_31, [-1, 24, 512, 64]);  bmm_31 = None
        permute_172 = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
        clone_63 = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_361 = torch.ops.aten.view.default(clone_63, [2, 512, -1]);  clone_63 = None
        view_362 = torch.ops.aten.view.default(view_361, [1024, 1536]);  view_361 = None
        permute_173 = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg253_1, view_362, permute_173);  arg253_1 = view_362 = permute_173 = None
        view_363 = torch.ops.aten.view.default(addmm_93, [2, 512, 1536]);  addmm_93 = None
        add_108 = torch.ops.aten.add.Tensor(view_363, add_107);  view_363 = add_107 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_108, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_31[0]
        getitem_63 = var_mean_31[1];  var_mean_31 = None
        add_109 = torch.ops.aten.add.Tensor(getitem_62, 1e-07);  getitem_62 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        sub_47 = torch.ops.aten.sub.Tensor(add_108, getitem_63);  add_108 = getitem_63 = None
        mul_125 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
        mul_126 = torch.ops.aten.mul.Tensor(mul_125, arg254_1);  mul_125 = arg254_1 = None
        add_110 = torch.ops.aten.add.Tensor(mul_126, arg255_1);  mul_126 = arg255_1 = None
        view_364 = torch.ops.aten.view.default(add_110, [1024, 1536])
        permute_174 = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg257_1, view_364, permute_174);  arg257_1 = view_364 = permute_174 = None
        view_365 = torch.ops.aten.view.default(addmm_94, [2, 512, 6144]);  addmm_94 = None
        mul_127 = torch.ops.aten.mul.Tensor(view_365, 0.5)
        mul_128 = torch.ops.aten.mul.Tensor(view_365, 0.7071067811865476);  view_365 = None
        erf_15 = torch.ops.aten.erf.default(mul_128);  mul_128 = None
        add_111 = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_127, add_111);  mul_127 = add_111 = None
        view_366 = torch.ops.aten.view.default(mul_129, [1024, 6144]);  mul_129 = None
        permute_175 = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg259_1, view_366, permute_175);  arg259_1 = view_366 = permute_175 = None
        view_367 = torch.ops.aten.view.default(addmm_95, [2, 512, 1536]);  addmm_95 = None
        add_112 = torch.ops.aten.add.Tensor(view_367, add_110);  view_367 = add_110 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_32[0]
        getitem_65 = var_mean_32[1];  var_mean_32 = None
        add_113 = torch.ops.aten.add.Tensor(getitem_64, 1e-07);  getitem_64 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        sub_48 = torch.ops.aten.sub.Tensor(add_112, getitem_65);  add_112 = getitem_65 = None
        mul_130 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = rsqrt_32 = None
        mul_131 = torch.ops.aten.mul.Tensor(mul_130, arg260_1);  mul_130 = arg260_1 = None
        add_114 = torch.ops.aten.add.Tensor(mul_131, arg261_1);  mul_131 = arg261_1 = None
        view_368 = torch.ops.aten.view.default(add_114, [1024, 1536])
        permute_176 = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg263_1, view_368, permute_176);  arg263_1 = view_368 = permute_176 = None
        view_369 = torch.ops.aten.view.default(addmm_96, [2, 512, 1536]);  addmm_96 = None
        view_370 = torch.ops.aten.view.default(view_369, [2, 512, 24, -1]);  view_369 = None
        permute_177 = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
        clone_64 = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_371 = torch.ops.aten.view.default(clone_64, [-1, 512, 64]);  clone_64 = None
        view_372 = torch.ops.aten.view.default(add_114, [1024, 1536])
        permute_178 = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg265_1, view_372, permute_178);  arg265_1 = view_372 = permute_178 = None
        view_373 = torch.ops.aten.view.default(addmm_97, [2, 512, 1536]);  addmm_97 = None
        view_374 = torch.ops.aten.view.default(view_373, [2, 512, 24, -1]);  view_373 = None
        permute_179 = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
        clone_65 = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        view_375 = torch.ops.aten.view.default(clone_65, [-1, 512, 64]);  clone_65 = None
        view_376 = torch.ops.aten.view.default(add_114, [1024, 1536])
        permute_180 = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg267_1, view_376, permute_180);  arg267_1 = view_376 = permute_180 = None
        view_377 = torch.ops.aten.view.default(addmm_98, [2, 512, 1536]);  addmm_98 = None
        view_378 = torch.ops.aten.view.default(view_377, [2, 512, 24, -1]);  view_377 = None
        permute_181 = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
        clone_66 = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_379 = torch.ops.aten.view.default(clone_66, [-1, 512, 64]);  clone_66 = None
        _tensor_constant32 = self._tensor_constant32
        lift_fresh_copy_32 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant32);  _tensor_constant32 = None
        mul_132 = torch.ops.aten.mul.Tensor(lift_fresh_copy_32, 1);  lift_fresh_copy_32 = mul_132 = None
        full_default_65 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_182 = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
        div_32 = torch.ops.aten.div.Tensor(permute_182, full_default_65);  permute_182 = full_default_65 = None
        bmm_32 = torch.ops.aten.bmm.default(view_371, div_32);  view_371 = div_32 = None
        view_380 = torch.ops.aten.view.default(bmm_32, [-1, 24, 512, 512]);  bmm_32 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_16 = None
        full_default_66 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant33 = self._tensor_constant33;  _tensor_constant33 = None
        full_default_67 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_32 = torch.ops.aten.where.self(full_default_66, full_default_67, view_380);  full_default_67 = view_380 = None
        amax_16 = torch.ops.aten.amax.default(where_32, [-1], True)
        sub_49 = torch.ops.aten.sub.Tensor(where_32, amax_16);  where_32 = amax_16 = None
        exp_16 = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_33 = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        full_default_68 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_33 = torch.ops.aten.where.self(full_default_66, full_default_68, div_33);  full_default_66 = full_default_68 = div_33 = None
        view_382 = torch.ops.aten.view.default(where_33, [-1, 512, 512]);  where_33 = None
        bmm_33 = torch.ops.aten.bmm.default(view_382, view_379);  view_382 = view_379 = None
        view_383 = torch.ops.aten.view.default(bmm_33, [-1, 24, 512, 64]);  bmm_33 = None
        permute_183 = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
        clone_67 = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_384 = torch.ops.aten.view.default(clone_67, [2, 512, -1]);  clone_67 = None
        view_385 = torch.ops.aten.view.default(view_384, [1024, 1536]);  view_384 = None
        permute_184 = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg269_1, view_385, permute_184);  arg269_1 = view_385 = permute_184 = None
        view_386 = torch.ops.aten.view.default(addmm_99, [2, 512, 1536]);  addmm_99 = None
        add_115 = torch.ops.aten.add.Tensor(view_386, add_114);  view_386 = add_114 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_115, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_33[0]
        getitem_67 = var_mean_33[1];  var_mean_33 = None
        add_116 = torch.ops.aten.add.Tensor(getitem_66, 1e-07);  getitem_66 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_50 = torch.ops.aten.sub.Tensor(add_115, getitem_67);  add_115 = getitem_67 = None
        mul_133 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_133, arg270_1);  mul_133 = arg270_1 = None
        add_117 = torch.ops.aten.add.Tensor(mul_134, arg271_1);  mul_134 = arg271_1 = None
        view_387 = torch.ops.aten.view.default(add_117, [1024, 1536])
        permute_185 = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg273_1, view_387, permute_185);  arg273_1 = view_387 = permute_185 = None
        view_388 = torch.ops.aten.view.default(addmm_100, [2, 512, 6144]);  addmm_100 = None
        mul_135 = torch.ops.aten.mul.Tensor(view_388, 0.5)
        mul_136 = torch.ops.aten.mul.Tensor(view_388, 0.7071067811865476);  view_388 = None
        erf_16 = torch.ops.aten.erf.default(mul_136);  mul_136 = None
        add_118 = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_135, add_118);  mul_135 = add_118 = None
        view_389 = torch.ops.aten.view.default(mul_137, [1024, 6144]);  mul_137 = None
        permute_186 = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg275_1, view_389, permute_186);  arg275_1 = view_389 = permute_186 = None
        view_390 = torch.ops.aten.view.default(addmm_101, [2, 512, 1536]);  addmm_101 = None
        add_119 = torch.ops.aten.add.Tensor(view_390, add_117);  view_390 = add_117 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
        getitem_68 = var_mean_34[0]
        getitem_69 = var_mean_34[1];  var_mean_34 = None
        add_120 = torch.ops.aten.add.Tensor(getitem_68, 1e-07);  getitem_68 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        sub_51 = torch.ops.aten.sub.Tensor(add_119, getitem_69);  add_119 = getitem_69 = None
        mul_138 = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = rsqrt_34 = None
        mul_139 = torch.ops.aten.mul.Tensor(mul_138, arg276_1);  mul_138 = arg276_1 = None
        add_121 = torch.ops.aten.add.Tensor(mul_139, arg277_1);  mul_139 = arg277_1 = None
        view_391 = torch.ops.aten.view.default(add_121, [1024, 1536])
        permute_187 = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg279_1, view_391, permute_187);  arg279_1 = view_391 = permute_187 = None
        view_392 = torch.ops.aten.view.default(addmm_102, [2, 512, 1536]);  addmm_102 = None
        view_393 = torch.ops.aten.view.default(view_392, [2, 512, 24, -1]);  view_392 = None
        permute_188 = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
        clone_68 = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_394 = torch.ops.aten.view.default(clone_68, [-1, 512, 64]);  clone_68 = None
        view_395 = torch.ops.aten.view.default(add_121, [1024, 1536])
        permute_189 = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg281_1, view_395, permute_189);  arg281_1 = view_395 = permute_189 = None
        view_396 = torch.ops.aten.view.default(addmm_103, [2, 512, 1536]);  addmm_103 = None
        view_397 = torch.ops.aten.view.default(view_396, [2, 512, 24, -1]);  view_396 = None
        permute_190 = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
        clone_69 = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_398 = torch.ops.aten.view.default(clone_69, [-1, 512, 64]);  clone_69 = None
        view_399 = torch.ops.aten.view.default(add_121, [1024, 1536])
        permute_191 = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg283_1, view_399, permute_191);  arg283_1 = view_399 = permute_191 = None
        view_400 = torch.ops.aten.view.default(addmm_104, [2, 512, 1536]);  addmm_104 = None
        view_401 = torch.ops.aten.view.default(view_400, [2, 512, 24, -1]);  view_400 = None
        permute_192 = torch.ops.aten.permute.default(view_401, [0, 2, 1, 3]);  view_401 = None
        clone_70 = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_402 = torch.ops.aten.view.default(clone_70, [-1, 512, 64]);  clone_70 = None
        _tensor_constant34 = self._tensor_constant34
        lift_fresh_copy_34 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant34);  _tensor_constant34 = None
        mul_140 = torch.ops.aten.mul.Tensor(lift_fresh_copy_34, 1);  lift_fresh_copy_34 = mul_140 = None
        full_default_69 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_193 = torch.ops.aten.permute.default(view_398, [0, 2, 1]);  view_398 = None
        div_34 = torch.ops.aten.div.Tensor(permute_193, full_default_69);  permute_193 = full_default_69 = None
        bmm_34 = torch.ops.aten.bmm.default(view_394, div_34);  view_394 = div_34 = None
        view_403 = torch.ops.aten.view.default(bmm_34, [-1, 24, 512, 512]);  bmm_34 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_17 = None
        full_default_70 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant35 = self._tensor_constant35;  _tensor_constant35 = None
        full_default_71 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_34 = torch.ops.aten.where.self(full_default_70, full_default_71, view_403);  full_default_71 = view_403 = None
        amax_17 = torch.ops.aten.amax.default(where_34, [-1], True)
        sub_52 = torch.ops.aten.sub.Tensor(where_34, amax_17);  where_34 = amax_17 = None
        exp_17 = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_35 = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        full_default_72 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_35 = torch.ops.aten.where.self(full_default_70, full_default_72, div_35);  full_default_70 = full_default_72 = div_35 = None
        view_405 = torch.ops.aten.view.default(where_35, [-1, 512, 512]);  where_35 = None
        bmm_35 = torch.ops.aten.bmm.default(view_405, view_402);  view_405 = view_402 = None
        view_406 = torch.ops.aten.view.default(bmm_35, [-1, 24, 512, 64]);  bmm_35 = None
        permute_194 = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
        clone_71 = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_407 = torch.ops.aten.view.default(clone_71, [2, 512, -1]);  clone_71 = None
        view_408 = torch.ops.aten.view.default(view_407, [1024, 1536]);  view_407 = None
        permute_195 = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg285_1, view_408, permute_195);  arg285_1 = view_408 = permute_195 = None
        view_409 = torch.ops.aten.view.default(addmm_105, [2, 512, 1536]);  addmm_105 = None
        add_122 = torch.ops.aten.add.Tensor(view_409, add_121);  view_409 = add_121 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_35[0]
        getitem_71 = var_mean_35[1];  var_mean_35 = None
        add_123 = torch.ops.aten.add.Tensor(getitem_70, 1e-07);  getitem_70 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_53 = torch.ops.aten.sub.Tensor(add_122, getitem_71);  add_122 = getitem_71 = None
        mul_141 = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
        mul_142 = torch.ops.aten.mul.Tensor(mul_141, arg286_1);  mul_141 = arg286_1 = None
        add_124 = torch.ops.aten.add.Tensor(mul_142, arg287_1);  mul_142 = arg287_1 = None
        view_410 = torch.ops.aten.view.default(add_124, [1024, 1536])
        permute_196 = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg289_1, view_410, permute_196);  arg289_1 = view_410 = permute_196 = None
        view_411 = torch.ops.aten.view.default(addmm_106, [2, 512, 6144]);  addmm_106 = None
        mul_143 = torch.ops.aten.mul.Tensor(view_411, 0.5)
        mul_144 = torch.ops.aten.mul.Tensor(view_411, 0.7071067811865476);  view_411 = None
        erf_17 = torch.ops.aten.erf.default(mul_144);  mul_144 = None
        add_125 = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_143, add_125);  mul_143 = add_125 = None
        view_412 = torch.ops.aten.view.default(mul_145, [1024, 6144]);  mul_145 = None
        permute_197 = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg291_1, view_412, permute_197);  arg291_1 = view_412 = permute_197 = None
        view_413 = torch.ops.aten.view.default(addmm_107, [2, 512, 1536]);  addmm_107 = None
        add_126 = torch.ops.aten.add.Tensor(view_413, add_124);  view_413 = add_124 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_126, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_36[0]
        getitem_73 = var_mean_36[1];  var_mean_36 = None
        add_127 = torch.ops.aten.add.Tensor(getitem_72, 1e-07);  getitem_72 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        sub_54 = torch.ops.aten.sub.Tensor(add_126, getitem_73);  add_126 = getitem_73 = None
        mul_146 = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = rsqrt_36 = None
        mul_147 = torch.ops.aten.mul.Tensor(mul_146, arg292_1);  mul_146 = arg292_1 = None
        add_128 = torch.ops.aten.add.Tensor(mul_147, arg293_1);  mul_147 = arg293_1 = None
        view_414 = torch.ops.aten.view.default(add_128, [1024, 1536])
        permute_198 = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg295_1, view_414, permute_198);  arg295_1 = view_414 = permute_198 = None
        view_415 = torch.ops.aten.view.default(addmm_108, [2, 512, 1536]);  addmm_108 = None
        view_416 = torch.ops.aten.view.default(view_415, [2, 512, 24, -1]);  view_415 = None
        permute_199 = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
        clone_72 = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_417 = torch.ops.aten.view.default(clone_72, [-1, 512, 64]);  clone_72 = None
        view_418 = torch.ops.aten.view.default(add_128, [1024, 1536])
        permute_200 = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg297_1, view_418, permute_200);  arg297_1 = view_418 = permute_200 = None
        view_419 = torch.ops.aten.view.default(addmm_109, [2, 512, 1536]);  addmm_109 = None
        view_420 = torch.ops.aten.view.default(view_419, [2, 512, 24, -1]);  view_419 = None
        permute_201 = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        clone_73 = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        view_421 = torch.ops.aten.view.default(clone_73, [-1, 512, 64]);  clone_73 = None
        view_422 = torch.ops.aten.view.default(add_128, [1024, 1536])
        permute_202 = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg299_1, view_422, permute_202);  arg299_1 = view_422 = permute_202 = None
        view_423 = torch.ops.aten.view.default(addmm_110, [2, 512, 1536]);  addmm_110 = None
        view_424 = torch.ops.aten.view.default(view_423, [2, 512, 24, -1]);  view_423 = None
        permute_203 = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
        clone_74 = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_425 = torch.ops.aten.view.default(clone_74, [-1, 512, 64]);  clone_74 = None
        _tensor_constant36 = self._tensor_constant36
        lift_fresh_copy_36 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant36);  _tensor_constant36 = None
        mul_148 = torch.ops.aten.mul.Tensor(lift_fresh_copy_36, 1);  lift_fresh_copy_36 = mul_148 = None
        full_default_73 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_204 = torch.ops.aten.permute.default(view_421, [0, 2, 1]);  view_421 = None
        div_36 = torch.ops.aten.div.Tensor(permute_204, full_default_73);  permute_204 = full_default_73 = None
        bmm_36 = torch.ops.aten.bmm.default(view_417, div_36);  view_417 = div_36 = None
        view_426 = torch.ops.aten.view.default(bmm_36, [-1, 24, 512, 512]);  bmm_36 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_18 = None
        full_default_74 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant37 = self._tensor_constant37;  _tensor_constant37 = None
        full_default_75 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_36 = torch.ops.aten.where.self(full_default_74, full_default_75, view_426);  full_default_75 = view_426 = None
        amax_18 = torch.ops.aten.amax.default(where_36, [-1], True)
        sub_55 = torch.ops.aten.sub.Tensor(where_36, amax_18);  where_36 = amax_18 = None
        exp_18 = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_37 = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        full_default_76 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_37 = torch.ops.aten.where.self(full_default_74, full_default_76, div_37);  full_default_74 = full_default_76 = div_37 = None
        view_428 = torch.ops.aten.view.default(where_37, [-1, 512, 512]);  where_37 = None
        bmm_37 = torch.ops.aten.bmm.default(view_428, view_425);  view_428 = view_425 = None
        view_429 = torch.ops.aten.view.default(bmm_37, [-1, 24, 512, 64]);  bmm_37 = None
        permute_205 = torch.ops.aten.permute.default(view_429, [0, 2, 1, 3]);  view_429 = None
        clone_75 = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_430 = torch.ops.aten.view.default(clone_75, [2, 512, -1]);  clone_75 = None
        view_431 = torch.ops.aten.view.default(view_430, [1024, 1536]);  view_430 = None
        permute_206 = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg301_1, view_431, permute_206);  arg301_1 = view_431 = permute_206 = None
        view_432 = torch.ops.aten.view.default(addmm_111, [2, 512, 1536]);  addmm_111 = None
        add_129 = torch.ops.aten.add.Tensor(view_432, add_128);  view_432 = add_128 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_74 = var_mean_37[0]
        getitem_75 = var_mean_37[1];  var_mean_37 = None
        add_130 = torch.ops.aten.add.Tensor(getitem_74, 1e-07);  getitem_74 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_56 = torch.ops.aten.sub.Tensor(add_129, getitem_75);  add_129 = getitem_75 = None
        mul_149 = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
        mul_150 = torch.ops.aten.mul.Tensor(mul_149, arg302_1);  mul_149 = arg302_1 = None
        add_131 = torch.ops.aten.add.Tensor(mul_150, arg303_1);  mul_150 = arg303_1 = None
        view_433 = torch.ops.aten.view.default(add_131, [1024, 1536])
        permute_207 = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg305_1, view_433, permute_207);  arg305_1 = view_433 = permute_207 = None
        view_434 = torch.ops.aten.view.default(addmm_112, [2, 512, 6144]);  addmm_112 = None
        mul_151 = torch.ops.aten.mul.Tensor(view_434, 0.5)
        mul_152 = torch.ops.aten.mul.Tensor(view_434, 0.7071067811865476);  view_434 = None
        erf_18 = torch.ops.aten.erf.default(mul_152);  mul_152 = None
        add_132 = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_151, add_132);  mul_151 = add_132 = None
        view_435 = torch.ops.aten.view.default(mul_153, [1024, 6144]);  mul_153 = None
        permute_208 = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg307_1, view_435, permute_208);  arg307_1 = view_435 = permute_208 = None
        view_436 = torch.ops.aten.view.default(addmm_113, [2, 512, 1536]);  addmm_113 = None
        add_133 = torch.ops.aten.add.Tensor(view_436, add_131);  view_436 = add_131 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_76 = var_mean_38[0]
        getitem_77 = var_mean_38[1];  var_mean_38 = None
        add_134 = torch.ops.aten.add.Tensor(getitem_76, 1e-07);  getitem_76 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_57 = torch.ops.aten.sub.Tensor(add_133, getitem_77);  add_133 = getitem_77 = None
        mul_154 = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = rsqrt_38 = None
        mul_155 = torch.ops.aten.mul.Tensor(mul_154, arg308_1);  mul_154 = arg308_1 = None
        add_135 = torch.ops.aten.add.Tensor(mul_155, arg309_1);  mul_155 = arg309_1 = None
        view_437 = torch.ops.aten.view.default(add_135, [1024, 1536])
        permute_209 = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg311_1, view_437, permute_209);  arg311_1 = view_437 = permute_209 = None
        view_438 = torch.ops.aten.view.default(addmm_114, [2, 512, 1536]);  addmm_114 = None
        view_439 = torch.ops.aten.view.default(view_438, [2, 512, 24, -1]);  view_438 = None
        permute_210 = torch.ops.aten.permute.default(view_439, [0, 2, 1, 3]);  view_439 = None
        clone_76 = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_440 = torch.ops.aten.view.default(clone_76, [-1, 512, 64]);  clone_76 = None
        view_441 = torch.ops.aten.view.default(add_135, [1024, 1536])
        permute_211 = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg313_1, view_441, permute_211);  arg313_1 = view_441 = permute_211 = None
        view_442 = torch.ops.aten.view.default(addmm_115, [2, 512, 1536]);  addmm_115 = None
        view_443 = torch.ops.aten.view.default(view_442, [2, 512, 24, -1]);  view_442 = None
        permute_212 = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
        clone_77 = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_444 = torch.ops.aten.view.default(clone_77, [-1, 512, 64]);  clone_77 = None
        view_445 = torch.ops.aten.view.default(add_135, [1024, 1536])
        permute_213 = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg315_1, view_445, permute_213);  arg315_1 = view_445 = permute_213 = None
        view_446 = torch.ops.aten.view.default(addmm_116, [2, 512, 1536]);  addmm_116 = None
        view_447 = torch.ops.aten.view.default(view_446, [2, 512, 24, -1]);  view_446 = None
        permute_214 = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
        clone_78 = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_448 = torch.ops.aten.view.default(clone_78, [-1, 512, 64]);  clone_78 = None
        _tensor_constant38 = self._tensor_constant38
        lift_fresh_copy_38 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant38);  _tensor_constant38 = None
        mul_156 = torch.ops.aten.mul.Tensor(lift_fresh_copy_38, 1);  lift_fresh_copy_38 = mul_156 = None
        full_default_77 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_215 = torch.ops.aten.permute.default(view_444, [0, 2, 1]);  view_444 = None
        div_38 = torch.ops.aten.div.Tensor(permute_215, full_default_77);  permute_215 = full_default_77 = None
        bmm_38 = torch.ops.aten.bmm.default(view_440, div_38);  view_440 = div_38 = None
        view_449 = torch.ops.aten.view.default(bmm_38, [-1, 24, 512, 512]);  bmm_38 = None
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_19 = None
        full_default_78 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant39 = self._tensor_constant39;  _tensor_constant39 = None
        full_default_79 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_38 = torch.ops.aten.where.self(full_default_78, full_default_79, view_449);  full_default_79 = view_449 = None
        amax_19 = torch.ops.aten.amax.default(where_38, [-1], True)
        sub_58 = torch.ops.aten.sub.Tensor(where_38, amax_19);  where_38 = amax_19 = None
        exp_19 = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_39 = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        full_default_80 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_39 = torch.ops.aten.where.self(full_default_78, full_default_80, div_39);  full_default_78 = full_default_80 = div_39 = None
        view_451 = torch.ops.aten.view.default(where_39, [-1, 512, 512]);  where_39 = None
        bmm_39 = torch.ops.aten.bmm.default(view_451, view_448);  view_451 = view_448 = None
        view_452 = torch.ops.aten.view.default(bmm_39, [-1, 24, 512, 64]);  bmm_39 = None
        permute_216 = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None
        clone_79 = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        view_453 = torch.ops.aten.view.default(clone_79, [2, 512, -1]);  clone_79 = None
        view_454 = torch.ops.aten.view.default(view_453, [1024, 1536]);  view_453 = None
        permute_217 = torch.ops.aten.permute.default(arg316_1, [1, 0]);  arg316_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg317_1, view_454, permute_217);  arg317_1 = view_454 = permute_217 = None
        view_455 = torch.ops.aten.view.default(addmm_117, [2, 512, 1536]);  addmm_117 = None
        add_136 = torch.ops.aten.add.Tensor(view_455, add_135);  view_455 = add_135 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_39[0]
        getitem_79 = var_mean_39[1];  var_mean_39 = None
        add_137 = torch.ops.aten.add.Tensor(getitem_78, 1e-07);  getitem_78 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        sub_59 = torch.ops.aten.sub.Tensor(add_136, getitem_79);  add_136 = getitem_79 = None
        mul_157 = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_157, arg318_1);  mul_157 = arg318_1 = None
        add_138 = torch.ops.aten.add.Tensor(mul_158, arg319_1);  mul_158 = arg319_1 = None
        view_456 = torch.ops.aten.view.default(add_138, [1024, 1536])
        permute_218 = torch.ops.aten.permute.default(arg320_1, [1, 0]);  arg320_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg321_1, view_456, permute_218);  arg321_1 = view_456 = permute_218 = None
        view_457 = torch.ops.aten.view.default(addmm_118, [2, 512, 6144]);  addmm_118 = None
        mul_159 = torch.ops.aten.mul.Tensor(view_457, 0.5)
        mul_160 = torch.ops.aten.mul.Tensor(view_457, 0.7071067811865476);  view_457 = None
        erf_19 = torch.ops.aten.erf.default(mul_160);  mul_160 = None
        add_139 = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_159, add_139);  mul_159 = add_139 = None
        view_458 = torch.ops.aten.view.default(mul_161, [1024, 6144]);  mul_161 = None
        permute_219 = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg323_1, view_458, permute_219);  arg323_1 = view_458 = permute_219 = None
        view_459 = torch.ops.aten.view.default(addmm_119, [2, 512, 1536]);  addmm_119 = None
        add_140 = torch.ops.aten.add.Tensor(view_459, add_138);  view_459 = add_138 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_40[0]
        getitem_81 = var_mean_40[1];  var_mean_40 = None
        add_141 = torch.ops.aten.add.Tensor(getitem_80, 1e-07);  getitem_80 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_60 = torch.ops.aten.sub.Tensor(add_140, getitem_81);  add_140 = getitem_81 = None
        mul_162 = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = rsqrt_40 = None
        mul_163 = torch.ops.aten.mul.Tensor(mul_162, arg324_1);  mul_162 = arg324_1 = None
        add_142 = torch.ops.aten.add.Tensor(mul_163, arg325_1);  mul_163 = arg325_1 = None
        view_460 = torch.ops.aten.view.default(add_142, [1024, 1536])
        permute_220 = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg327_1, view_460, permute_220);  arg327_1 = view_460 = permute_220 = None
        view_461 = torch.ops.aten.view.default(addmm_120, [2, 512, 1536]);  addmm_120 = None
        view_462 = torch.ops.aten.view.default(view_461, [2, 512, 24, -1]);  view_461 = None
        permute_221 = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
        clone_80 = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_463 = torch.ops.aten.view.default(clone_80, [-1, 512, 64]);  clone_80 = None
        view_464 = torch.ops.aten.view.default(add_142, [1024, 1536])
        permute_222 = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg329_1, view_464, permute_222);  arg329_1 = view_464 = permute_222 = None
        view_465 = torch.ops.aten.view.default(addmm_121, [2, 512, 1536]);  addmm_121 = None
        view_466 = torch.ops.aten.view.default(view_465, [2, 512, 24, -1]);  view_465 = None
        permute_223 = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
        clone_81 = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        view_467 = torch.ops.aten.view.default(clone_81, [-1, 512, 64]);  clone_81 = None
        view_468 = torch.ops.aten.view.default(add_142, [1024, 1536])
        permute_224 = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg331_1, view_468, permute_224);  arg331_1 = view_468 = permute_224 = None
        view_469 = torch.ops.aten.view.default(addmm_122, [2, 512, 1536]);  addmm_122 = None
        view_470 = torch.ops.aten.view.default(view_469, [2, 512, 24, -1]);  view_469 = None
        permute_225 = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
        clone_82 = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_471 = torch.ops.aten.view.default(clone_82, [-1, 512, 64]);  clone_82 = None
        _tensor_constant40 = self._tensor_constant40
        lift_fresh_copy_40 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant40);  _tensor_constant40 = None
        mul_164 = torch.ops.aten.mul.Tensor(lift_fresh_copy_40, 1);  lift_fresh_copy_40 = mul_164 = None
        full_default_81 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_226 = torch.ops.aten.permute.default(view_467, [0, 2, 1]);  view_467 = None
        div_40 = torch.ops.aten.div.Tensor(permute_226, full_default_81);  permute_226 = full_default_81 = None
        bmm_40 = torch.ops.aten.bmm.default(view_463, div_40);  view_463 = div_40 = None
        view_472 = torch.ops.aten.view.default(bmm_40, [-1, 24, 512, 512]);  bmm_40 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_20 = None
        full_default_82 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant41 = self._tensor_constant41;  _tensor_constant41 = None
        full_default_83 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_40 = torch.ops.aten.where.self(full_default_82, full_default_83, view_472);  full_default_83 = view_472 = None
        amax_20 = torch.ops.aten.amax.default(where_40, [-1], True)
        sub_61 = torch.ops.aten.sub.Tensor(where_40, amax_20);  where_40 = amax_20 = None
        exp_20 = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_41 = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        full_default_84 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_41 = torch.ops.aten.where.self(full_default_82, full_default_84, div_41);  full_default_82 = full_default_84 = div_41 = None
        view_474 = torch.ops.aten.view.default(where_41, [-1, 512, 512]);  where_41 = None
        bmm_41 = torch.ops.aten.bmm.default(view_474, view_471);  view_474 = view_471 = None
        view_475 = torch.ops.aten.view.default(bmm_41, [-1, 24, 512, 64]);  bmm_41 = None
        permute_227 = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
        clone_83 = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_476 = torch.ops.aten.view.default(clone_83, [2, 512, -1]);  clone_83 = None
        view_477 = torch.ops.aten.view.default(view_476, [1024, 1536]);  view_476 = None
        permute_228 = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg333_1, view_477, permute_228);  arg333_1 = view_477 = permute_228 = None
        view_478 = torch.ops.aten.view.default(addmm_123, [2, 512, 1536]);  addmm_123 = None
        add_143 = torch.ops.aten.add.Tensor(view_478, add_142);  view_478 = add_142 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
        getitem_82 = var_mean_41[0]
        getitem_83 = var_mean_41[1];  var_mean_41 = None
        add_144 = torch.ops.aten.add.Tensor(getitem_82, 1e-07);  getitem_82 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_62 = torch.ops.aten.sub.Tensor(add_143, getitem_83);  add_143 = getitem_83 = None
        mul_165 = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
        mul_166 = torch.ops.aten.mul.Tensor(mul_165, arg334_1);  mul_165 = arg334_1 = None
        add_145 = torch.ops.aten.add.Tensor(mul_166, arg335_1);  mul_166 = arg335_1 = None
        view_479 = torch.ops.aten.view.default(add_145, [1024, 1536])
        permute_229 = torch.ops.aten.permute.default(arg336_1, [1, 0]);  arg336_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg337_1, view_479, permute_229);  arg337_1 = view_479 = permute_229 = None
        view_480 = torch.ops.aten.view.default(addmm_124, [2, 512, 6144]);  addmm_124 = None
        mul_167 = torch.ops.aten.mul.Tensor(view_480, 0.5)
        mul_168 = torch.ops.aten.mul.Tensor(view_480, 0.7071067811865476);  view_480 = None
        erf_20 = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_146 = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_167, add_146);  mul_167 = add_146 = None
        view_481 = torch.ops.aten.view.default(mul_169, [1024, 6144]);  mul_169 = None
        permute_230 = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg339_1, view_481, permute_230);  arg339_1 = view_481 = permute_230 = None
        view_482 = torch.ops.aten.view.default(addmm_125, [2, 512, 1536]);  addmm_125 = None
        add_147 = torch.ops.aten.add.Tensor(view_482, add_145);  view_482 = add_145 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_84 = var_mean_42[0]
        getitem_85 = var_mean_42[1];  var_mean_42 = None
        add_148 = torch.ops.aten.add.Tensor(getitem_84, 1e-07);  getitem_84 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_63 = torch.ops.aten.sub.Tensor(add_147, getitem_85);  add_147 = getitem_85 = None
        mul_170 = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = rsqrt_42 = None
        mul_171 = torch.ops.aten.mul.Tensor(mul_170, arg340_1);  mul_170 = arg340_1 = None
        add_149 = torch.ops.aten.add.Tensor(mul_171, arg341_1);  mul_171 = arg341_1 = None
        view_483 = torch.ops.aten.view.default(add_149, [1024, 1536])
        permute_231 = torch.ops.aten.permute.default(arg342_1, [1, 0]);  arg342_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg343_1, view_483, permute_231);  arg343_1 = view_483 = permute_231 = None
        view_484 = torch.ops.aten.view.default(addmm_126, [2, 512, 1536]);  addmm_126 = None
        view_485 = torch.ops.aten.view.default(view_484, [2, 512, 24, -1]);  view_484 = None
        permute_232 = torch.ops.aten.permute.default(view_485, [0, 2, 1, 3]);  view_485 = None
        clone_84 = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_486 = torch.ops.aten.view.default(clone_84, [-1, 512, 64]);  clone_84 = None
        view_487 = torch.ops.aten.view.default(add_149, [1024, 1536])
        permute_233 = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg345_1, view_487, permute_233);  arg345_1 = view_487 = permute_233 = None
        view_488 = torch.ops.aten.view.default(addmm_127, [2, 512, 1536]);  addmm_127 = None
        view_489 = torch.ops.aten.view.default(view_488, [2, 512, 24, -1]);  view_488 = None
        permute_234 = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None
        clone_85 = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
        view_490 = torch.ops.aten.view.default(clone_85, [-1, 512, 64]);  clone_85 = None
        view_491 = torch.ops.aten.view.default(add_149, [1024, 1536])
        permute_235 = torch.ops.aten.permute.default(arg346_1, [1, 0]);  arg346_1 = None
        addmm_128 = torch.ops.aten.addmm.default(arg347_1, view_491, permute_235);  arg347_1 = view_491 = permute_235 = None
        view_492 = torch.ops.aten.view.default(addmm_128, [2, 512, 1536]);  addmm_128 = None
        view_493 = torch.ops.aten.view.default(view_492, [2, 512, 24, -1]);  view_492 = None
        permute_236 = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
        clone_86 = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_494 = torch.ops.aten.view.default(clone_86, [-1, 512, 64]);  clone_86 = None
        _tensor_constant42 = self._tensor_constant42
        lift_fresh_copy_42 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant42);  _tensor_constant42 = None
        mul_172 = torch.ops.aten.mul.Tensor(lift_fresh_copy_42, 1);  lift_fresh_copy_42 = mul_172 = None
        full_default_85 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_237 = torch.ops.aten.permute.default(view_490, [0, 2, 1]);  view_490 = None
        div_42 = torch.ops.aten.div.Tensor(permute_237, full_default_85);  permute_237 = full_default_85 = None
        bmm_42 = torch.ops.aten.bmm.default(view_486, div_42);  view_486 = div_42 = None
        view_495 = torch.ops.aten.view.default(bmm_42, [-1, 24, 512, 512]);  bmm_42 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_21 = None
        full_default_86 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant43 = self._tensor_constant43;  _tensor_constant43 = None
        full_default_87 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_42 = torch.ops.aten.where.self(full_default_86, full_default_87, view_495);  full_default_87 = view_495 = None
        amax_21 = torch.ops.aten.amax.default(where_42, [-1], True)
        sub_64 = torch.ops.aten.sub.Tensor(where_42, amax_21);  where_42 = amax_21 = None
        exp_21 = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_43 = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        full_default_88 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_43 = torch.ops.aten.where.self(full_default_86, full_default_88, div_43);  full_default_86 = full_default_88 = div_43 = None
        view_497 = torch.ops.aten.view.default(where_43, [-1, 512, 512]);  where_43 = None
        bmm_43 = torch.ops.aten.bmm.default(view_497, view_494);  view_497 = view_494 = None
        view_498 = torch.ops.aten.view.default(bmm_43, [-1, 24, 512, 64]);  bmm_43 = None
        permute_238 = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
        clone_87 = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_499 = torch.ops.aten.view.default(clone_87, [2, 512, -1]);  clone_87 = None
        view_500 = torch.ops.aten.view.default(view_499, [1024, 1536]);  view_499 = None
        permute_239 = torch.ops.aten.permute.default(arg348_1, [1, 0]);  arg348_1 = None
        addmm_129 = torch.ops.aten.addmm.default(arg349_1, view_500, permute_239);  arg349_1 = view_500 = permute_239 = None
        view_501 = torch.ops.aten.view.default(addmm_129, [2, 512, 1536]);  addmm_129 = None
        add_150 = torch.ops.aten.add.Tensor(view_501, add_149);  view_501 = add_149 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_43[0]
        getitem_87 = var_mean_43[1];  var_mean_43 = None
        add_151 = torch.ops.aten.add.Tensor(getitem_86, 1e-07);  getitem_86 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        sub_65 = torch.ops.aten.sub.Tensor(add_150, getitem_87);  add_150 = getitem_87 = None
        mul_173 = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
        mul_174 = torch.ops.aten.mul.Tensor(mul_173, arg350_1);  mul_173 = arg350_1 = None
        add_152 = torch.ops.aten.add.Tensor(mul_174, arg351_1);  mul_174 = arg351_1 = None
        view_502 = torch.ops.aten.view.default(add_152, [1024, 1536])
        permute_240 = torch.ops.aten.permute.default(arg352_1, [1, 0]);  arg352_1 = None
        addmm_130 = torch.ops.aten.addmm.default(arg353_1, view_502, permute_240);  arg353_1 = view_502 = permute_240 = None
        view_503 = torch.ops.aten.view.default(addmm_130, [2, 512, 6144]);  addmm_130 = None
        mul_175 = torch.ops.aten.mul.Tensor(view_503, 0.5)
        mul_176 = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476);  view_503 = None
        erf_21 = torch.ops.aten.erf.default(mul_176);  mul_176 = None
        add_153 = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_177 = torch.ops.aten.mul.Tensor(mul_175, add_153);  mul_175 = add_153 = None
        view_504 = torch.ops.aten.view.default(mul_177, [1024, 6144]);  mul_177 = None
        permute_241 = torch.ops.aten.permute.default(arg354_1, [1, 0]);  arg354_1 = None
        addmm_131 = torch.ops.aten.addmm.default(arg355_1, view_504, permute_241);  arg355_1 = view_504 = permute_241 = None
        view_505 = torch.ops.aten.view.default(addmm_131, [2, 512, 1536]);  addmm_131 = None
        add_154 = torch.ops.aten.add.Tensor(view_505, add_152);  view_505 = add_152 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_44[0]
        getitem_89 = var_mean_44[1];  var_mean_44 = None
        add_155 = torch.ops.aten.add.Tensor(getitem_88, 1e-07);  getitem_88 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        sub_66 = torch.ops.aten.sub.Tensor(add_154, getitem_89);  add_154 = getitem_89 = None
        mul_178 = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = rsqrt_44 = None
        mul_179 = torch.ops.aten.mul.Tensor(mul_178, arg356_1);  mul_178 = arg356_1 = None
        add_156 = torch.ops.aten.add.Tensor(mul_179, arg357_1);  mul_179 = arg357_1 = None
        view_506 = torch.ops.aten.view.default(add_156, [1024, 1536])
        permute_242 = torch.ops.aten.permute.default(arg358_1, [1, 0]);  arg358_1 = None
        addmm_132 = torch.ops.aten.addmm.default(arg359_1, view_506, permute_242);  arg359_1 = view_506 = permute_242 = None
        view_507 = torch.ops.aten.view.default(addmm_132, [2, 512, 1536]);  addmm_132 = None
        view_508 = torch.ops.aten.view.default(view_507, [2, 512, 24, -1]);  view_507 = None
        permute_243 = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
        clone_88 = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_509 = torch.ops.aten.view.default(clone_88, [-1, 512, 64]);  clone_88 = None
        view_510 = torch.ops.aten.view.default(add_156, [1024, 1536])
        permute_244 = torch.ops.aten.permute.default(arg360_1, [1, 0]);  arg360_1 = None
        addmm_133 = torch.ops.aten.addmm.default(arg361_1, view_510, permute_244);  arg361_1 = view_510 = permute_244 = None
        view_511 = torch.ops.aten.view.default(addmm_133, [2, 512, 1536]);  addmm_133 = None
        view_512 = torch.ops.aten.view.default(view_511, [2, 512, 24, -1]);  view_511 = None
        permute_245 = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
        clone_89 = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
        view_513 = torch.ops.aten.view.default(clone_89, [-1, 512, 64]);  clone_89 = None
        view_514 = torch.ops.aten.view.default(add_156, [1024, 1536])
        permute_246 = torch.ops.aten.permute.default(arg362_1, [1, 0]);  arg362_1 = None
        addmm_134 = torch.ops.aten.addmm.default(arg363_1, view_514, permute_246);  arg363_1 = view_514 = permute_246 = None
        view_515 = torch.ops.aten.view.default(addmm_134, [2, 512, 1536]);  addmm_134 = None
        view_516 = torch.ops.aten.view.default(view_515, [2, 512, 24, -1]);  view_515 = None
        permute_247 = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
        clone_90 = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_517 = torch.ops.aten.view.default(clone_90, [-1, 512, 64]);  clone_90 = None
        _tensor_constant44 = self._tensor_constant44
        lift_fresh_copy_44 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant44);  _tensor_constant44 = None
        mul_180 = torch.ops.aten.mul.Tensor(lift_fresh_copy_44, 1);  lift_fresh_copy_44 = mul_180 = None
        full_default_89 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_248 = torch.ops.aten.permute.default(view_513, [0, 2, 1]);  view_513 = None
        div_44 = torch.ops.aten.div.Tensor(permute_248, full_default_89);  permute_248 = full_default_89 = None
        bmm_44 = torch.ops.aten.bmm.default(view_509, div_44);  view_509 = div_44 = None
        view_518 = torch.ops.aten.view.default(bmm_44, [-1, 24, 512, 512]);  bmm_44 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  convert_element_type_22 = None
        full_default_90 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant45 = self._tensor_constant45;  _tensor_constant45 = None
        full_default_91 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_44 = torch.ops.aten.where.self(full_default_90, full_default_91, view_518);  full_default_91 = view_518 = None
        amax_22 = torch.ops.aten.amax.default(where_44, [-1], True)
        sub_67 = torch.ops.aten.sub.Tensor(where_44, amax_22);  where_44 = amax_22 = None
        exp_22 = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_45 = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        full_default_92 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_45 = torch.ops.aten.where.self(full_default_90, full_default_92, div_45);  full_default_90 = full_default_92 = div_45 = None
        view_520 = torch.ops.aten.view.default(where_45, [-1, 512, 512]);  where_45 = None
        bmm_45 = torch.ops.aten.bmm.default(view_520, view_517);  view_520 = view_517 = None
        view_521 = torch.ops.aten.view.default(bmm_45, [-1, 24, 512, 64]);  bmm_45 = None
        permute_249 = torch.ops.aten.permute.default(view_521, [0, 2, 1, 3]);  view_521 = None
        clone_91 = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_522 = torch.ops.aten.view.default(clone_91, [2, 512, -1]);  clone_91 = None
        view_523 = torch.ops.aten.view.default(view_522, [1024, 1536]);  view_522 = None
        permute_250 = torch.ops.aten.permute.default(arg364_1, [1, 0]);  arg364_1 = None
        addmm_135 = torch.ops.aten.addmm.default(arg365_1, view_523, permute_250);  arg365_1 = view_523 = permute_250 = None
        view_524 = torch.ops.aten.view.default(addmm_135, [2, 512, 1536]);  addmm_135 = None
        add_157 = torch.ops.aten.add.Tensor(view_524, add_156);  view_524 = add_156 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_90 = var_mean_45[0]
        getitem_91 = var_mean_45[1];  var_mean_45 = None
        add_158 = torch.ops.aten.add.Tensor(getitem_90, 1e-07);  getitem_90 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        sub_68 = torch.ops.aten.sub.Tensor(add_157, getitem_91);  add_157 = getitem_91 = None
        mul_181 = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
        mul_182 = torch.ops.aten.mul.Tensor(mul_181, arg366_1);  mul_181 = arg366_1 = None
        add_159 = torch.ops.aten.add.Tensor(mul_182, arg367_1);  mul_182 = arg367_1 = None
        view_525 = torch.ops.aten.view.default(add_159, [1024, 1536])
        permute_251 = torch.ops.aten.permute.default(arg368_1, [1, 0]);  arg368_1 = None
        addmm_136 = torch.ops.aten.addmm.default(arg369_1, view_525, permute_251);  arg369_1 = view_525 = permute_251 = None
        view_526 = torch.ops.aten.view.default(addmm_136, [2, 512, 6144]);  addmm_136 = None
        mul_183 = torch.ops.aten.mul.Tensor(view_526, 0.5)
        mul_184 = torch.ops.aten.mul.Tensor(view_526, 0.7071067811865476);  view_526 = None
        erf_22 = torch.ops.aten.erf.default(mul_184);  mul_184 = None
        add_160 = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_183, add_160);  mul_183 = add_160 = None
        view_527 = torch.ops.aten.view.default(mul_185, [1024, 6144]);  mul_185 = None
        permute_252 = torch.ops.aten.permute.default(arg370_1, [1, 0]);  arg370_1 = None
        addmm_137 = torch.ops.aten.addmm.default(arg371_1, view_527, permute_252);  arg371_1 = view_527 = permute_252 = None
        view_528 = torch.ops.aten.view.default(addmm_137, [2, 512, 1536]);  addmm_137 = None
        add_161 = torch.ops.aten.add.Tensor(view_528, add_159);  view_528 = add_159 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_92 = var_mean_46[0]
        getitem_93 = var_mean_46[1];  var_mean_46 = None
        add_162 = torch.ops.aten.add.Tensor(getitem_92, 1e-07);  getitem_92 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_69 = torch.ops.aten.sub.Tensor(add_161, getitem_93);  add_161 = getitem_93 = None
        mul_186 = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = rsqrt_46 = None
        mul_187 = torch.ops.aten.mul.Tensor(mul_186, arg372_1);  mul_186 = arg372_1 = None
        add_163 = torch.ops.aten.add.Tensor(mul_187, arg373_1);  mul_187 = arg373_1 = None
        view_529 = torch.ops.aten.view.default(add_163, [1024, 1536])
        permute_253 = torch.ops.aten.permute.default(arg374_1, [1, 0]);  arg374_1 = None
        addmm_138 = torch.ops.aten.addmm.default(arg375_1, view_529, permute_253);  arg375_1 = view_529 = permute_253 = None
        view_530 = torch.ops.aten.view.default(addmm_138, [2, 512, 1536]);  addmm_138 = None
        view_531 = torch.ops.aten.view.default(view_530, [2, 512, 24, -1]);  view_530 = None
        permute_254 = torch.ops.aten.permute.default(view_531, [0, 2, 1, 3]);  view_531 = None
        clone_92 = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_532 = torch.ops.aten.view.default(clone_92, [-1, 512, 64]);  clone_92 = None
        view_533 = torch.ops.aten.view.default(add_163, [1024, 1536])
        permute_255 = torch.ops.aten.permute.default(arg376_1, [1, 0]);  arg376_1 = None
        addmm_139 = torch.ops.aten.addmm.default(arg377_1, view_533, permute_255);  arg377_1 = view_533 = permute_255 = None
        view_534 = torch.ops.aten.view.default(addmm_139, [2, 512, 1536]);  addmm_139 = None
        view_535 = torch.ops.aten.view.default(view_534, [2, 512, 24, -1]);  view_534 = None
        permute_256 = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
        clone_93 = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_536 = torch.ops.aten.view.default(clone_93, [-1, 512, 64]);  clone_93 = None
        view_537 = torch.ops.aten.view.default(add_163, [1024, 1536])
        permute_257 = torch.ops.aten.permute.default(arg378_1, [1, 0]);  arg378_1 = None
        addmm_140 = torch.ops.aten.addmm.default(arg379_1, view_537, permute_257);  arg379_1 = view_537 = permute_257 = None
        view_538 = torch.ops.aten.view.default(addmm_140, [2, 512, 1536]);  addmm_140 = None
        view_539 = torch.ops.aten.view.default(view_538, [2, 512, 24, -1]);  view_538 = None
        permute_258 = torch.ops.aten.permute.default(view_539, [0, 2, 1, 3]);  view_539 = None
        clone_94 = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
        view_540 = torch.ops.aten.view.default(clone_94, [-1, 512, 64]);  clone_94 = None
        _tensor_constant46 = self._tensor_constant46
        lift_fresh_copy_46 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant46);  _tensor_constant46 = None
        mul_188 = torch.ops.aten.mul.Tensor(lift_fresh_copy_46, 1);  lift_fresh_copy_46 = mul_188 = None
        full_default_93 = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        permute_259 = torch.ops.aten.permute.default(view_536, [0, 2, 1]);  view_536 = None
        div_46 = torch.ops.aten.div.Tensor(permute_259, full_default_93);  permute_259 = full_default_93 = None
        bmm_46 = torch.ops.aten.bmm.default(view_532, div_46);  view_532 = div_46 = None
        view_541 = torch.ops.aten.view.default(bmm_46, [-1, 24, 512, 512]);  bmm_46 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  mul_3 = convert_element_type_23 = None
        full_default_94 = torch.ops.aten.full.default([2, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant47 = self._tensor_constant47;  _tensor_constant47 = None
        full_default_95 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_46 = torch.ops.aten.where.self(full_default_94, full_default_95, view_541);  full_default_95 = view_541 = None
        amax_23 = torch.ops.aten.amax.default(where_46, [-1], True)
        sub_70 = torch.ops.aten.sub.Tensor(where_46, amax_23);  where_46 = amax_23 = None
        exp_23 = torch.ops.aten.exp.default(sub_70);  sub_70 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_47 = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        full_default_96 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_47 = torch.ops.aten.where.self(full_default_94, full_default_96, div_47);  full_default_94 = full_default_96 = div_47 = None
        view_543 = torch.ops.aten.view.default(where_47, [-1, 512, 512]);  where_47 = None
        bmm_47 = torch.ops.aten.bmm.default(view_543, view_540);  view_543 = view_540 = None
        view_544 = torch.ops.aten.view.default(bmm_47, [-1, 24, 512, 64]);  bmm_47 = None
        permute_260 = torch.ops.aten.permute.default(view_544, [0, 2, 1, 3]);  view_544 = None
        clone_95 = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_545 = torch.ops.aten.view.default(clone_95, [2, 512, -1]);  clone_95 = None
        view_546 = torch.ops.aten.view.default(view_545, [1024, 1536]);  view_545 = None
        permute_261 = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg381_1, view_546, permute_261);  arg381_1 = view_546 = permute_261 = None
        view_547 = torch.ops.aten.view.default(addmm_141, [2, 512, 1536]);  addmm_141 = None
        add_164 = torch.ops.aten.add.Tensor(view_547, add_163);  view_547 = add_163 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
        getitem_94 = var_mean_47[0]
        getitem_95 = var_mean_47[1];  var_mean_47 = None
        add_165 = torch.ops.aten.add.Tensor(getitem_94, 1e-07);  getitem_94 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_71 = torch.ops.aten.sub.Tensor(add_164, getitem_95);  add_164 = getitem_95 = None
        mul_189 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
        mul_190 = torch.ops.aten.mul.Tensor(mul_189, arg382_1);  mul_189 = arg382_1 = None
        add_166 = torch.ops.aten.add.Tensor(mul_190, arg383_1);  mul_190 = arg383_1 = None
        view_548 = torch.ops.aten.view.default(add_166, [1024, 1536])
        permute_262 = torch.ops.aten.permute.default(arg384_1, [1, 0]);  arg384_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg385_1, view_548, permute_262);  arg385_1 = view_548 = permute_262 = None
        view_549 = torch.ops.aten.view.default(addmm_142, [2, 512, 6144]);  addmm_142 = None
        mul_191 = torch.ops.aten.mul.Tensor(view_549, 0.5)
        mul_192 = torch.ops.aten.mul.Tensor(view_549, 0.7071067811865476);  view_549 = None
        erf_23 = torch.ops.aten.erf.default(mul_192);  mul_192 = None
        add_167 = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_191, add_167);  mul_191 = add_167 = None
        view_550 = torch.ops.aten.view.default(mul_193, [1024, 6144]);  mul_193 = None
        permute_263 = torch.ops.aten.permute.default(arg386_1, [1, 0]);  arg386_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg387_1, view_550, permute_263);  arg387_1 = view_550 = permute_263 = None
        view_551 = torch.ops.aten.view.default(addmm_143, [2, 512, 1536]);  addmm_143 = None
        add_168 = torch.ops.aten.add.Tensor(view_551, add_166);  view_551 = add_166 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(add_168, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_48[0]
        getitem_97 = var_mean_48[1];  var_mean_48 = None
        add_169 = torch.ops.aten.add.Tensor(getitem_96, 1e-07);  getitem_96 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
        sub_72 = torch.ops.aten.sub.Tensor(add_168, getitem_97);  add_168 = getitem_97 = None
        mul_194 = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = rsqrt_48 = None
        mul_195 = torch.ops.aten.mul.Tensor(mul_194, arg388_1);  mul_194 = arg388_1 = None
        add_170 = torch.ops.aten.add.Tensor(mul_195, arg389_1);  mul_195 = arg389_1 = None
        view_552 = torch.ops.aten.view.default(add_170, [1024, 1536]);  add_170 = None
        permute_264 = torch.ops.aten.permute.default(arg390_1, [1, 0]);  arg390_1 = None
        addmm_144 = torch.ops.aten.addmm.default(arg391_1, view_552, permute_264);  arg391_1 = view_552 = permute_264 = None
        view_553 = torch.ops.aten.view.default(addmm_144, [2, 512, 1536]);  addmm_144 = None
        mul_196 = torch.ops.aten.mul.Tensor(view_553, 0.5)
        mul_197 = torch.ops.aten.mul.Tensor(view_553, 0.7071067811865476);  view_553 = None
        erf_24 = torch.ops.aten.erf.default(mul_197);  mul_197 = None
        add_171 = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_198 = torch.ops.aten.mul.Tensor(mul_196, add_171);  mul_196 = add_171 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(mul_198, [2], correction = 0, keepdim = True)
        getitem_98 = var_mean_49[0]
        getitem_99 = var_mean_49[1];  var_mean_49 = None
        add_172 = torch.ops.aten.add.Tensor(getitem_98, 1e-07);  getitem_98 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        sub_73 = torch.ops.aten.sub.Tensor(mul_198, getitem_99);  mul_198 = getitem_99 = None
        mul_199 = torch.ops.aten.mul.Tensor(sub_73, rsqrt_49);  sub_73 = rsqrt_49 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, arg392_1);  mul_199 = arg392_1 = None
        add_173 = torch.ops.aten.add.Tensor(mul_200, arg393_1);  mul_200 = arg393_1 = None
        view_554 = torch.ops.aten.view.default(add_173, [1024, 1536]);  add_173 = None
        permute_265 = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        addmm_145 = torch.ops.aten.addmm.default(arg394_1, view_554, permute_265);  arg394_1 = view_554 = permute_265 = None
        view_555 = torch.ops.aten.view.default(addmm_145, [2, 512, 128100]);  addmm_145 = None
        view_556 = torch.ops.aten.view.default(view_555, [-1, 128100])
        view_557 = torch.ops.aten.view.default(arg395_1, [-1]);  arg395_1 = None
        amax_24 = torch.ops.aten.amax.default(view_556, [1], True)
        sub_74 = torch.ops.aten.sub.Tensor(view_556, amax_24);  view_556 = amax_24 = None
        exp_24 = torch.ops.aten.exp.default(sub_74)
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_75 = torch.ops.aten.sub.Tensor(sub_74, log);  sub_74 = log = None
        ne = torch.ops.aten.ne.Scalar(view_557, -100)
        full_default_97 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_48 = torch.ops.aten.where.self(ne, view_557, full_default_97);  ne = full_default_97 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where_48, 1);  where_48 = None
        gather = torch.ops.aten.gather.default(sub_75, 1, unsqueeze_4);  sub_75 = unsqueeze_4 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
        ne_1 = torch.ops.aten.ne.Scalar(view_557, -100)
        full_default_98 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_49 = torch.ops.aten.where.self(ne_1, neg, full_default_98);  ne_1 = neg = full_default_98 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_557, -100);  view_557 = None
        sum_26 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27 = torch.ops.aten.sum.default(where_49);  where_49 = None
        div_48 = torch.ops.aten.div.Tensor(sum_27, convert_element_type_24);  sum_27 = convert_element_type_24 = None
        return (div_48, view_555)
        
def load_args(reader):
    buf0 = reader.storage(None, 8192, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (2, 512), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (1, 512), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 787046400, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128100, 1536), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf3, (512, 1536), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1536,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf5, (1536,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf6, (1536, 1536), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf7, (1536,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf8, (1536, 1536), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1536,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1536, 1536), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf11, (1536,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf12, (1536, 1536), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf13, (1536,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf14, (1536,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf15, (1536,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf16, (6144, 1536), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf17, (6144,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf18, (1536, 6144), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf19, (1536,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1536,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf21, (1536,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf22, (1536, 1536), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf23, (1536,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf24, (1536, 1536), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf25, (1536,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf26, (1536, 1536), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf27, (1536,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf28, (1536, 1536), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf29, (1536,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf30, (1536,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf31, (1536,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf32, (6144, 1536), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf33, (6144,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf34, (1536, 6144), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf35, (1536,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1536,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf37, (1536,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf38, (1536, 1536), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf39, (1536,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf40, (1536, 1536), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf41, (1536,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf42, (1536, 1536), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf43, (1536,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf44, (1536, 1536), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf45, (1536,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf46, (1536,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf47, (1536,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf48, (6144, 1536), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf49, (6144,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf50, (1536, 6144), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf51, (1536,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf52, (1536,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf53, (1536,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf54, (1536, 1536), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf55, (1536,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf56, (1536, 1536), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf57, (1536,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf58, (1536, 1536), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf59, (1536,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf60, (1536, 1536), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf61, (1536,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf62, (1536,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf63, (1536,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf64, (6144, 1536), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf65, (6144,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf66, (1536, 6144), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf67, (1536,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf68, (1536,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf69, (1536,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf70, (1536, 1536), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf71, (1536,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf72, (1536, 1536), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf73, (1536,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf74, (1536, 1536), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf75, (1536,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf76, (1536, 1536), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf77, (1536,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf78, (1536,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf79, (1536,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf80, (6144, 1536), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf81, (6144,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf82, (1536, 6144), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf83, (1536,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf84, (1536,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf85, (1536,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf86, (1536, 1536), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf87, (1536,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf88, (1536, 1536), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf89, (1536,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf90, (1536, 1536), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf91, (1536,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf92, (1536, 1536), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf93, (1536,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf94, (1536,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf95, (1536,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf96, (6144, 1536), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf97, (6144,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf98, (1536, 6144), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf99, (1536,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1536,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf101, (1536,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf102, (1536, 1536), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf103, (1536,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf104, (1536, 1536), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf105, (1536,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf106, (1536, 1536), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf107, (1536,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf108, (1536, 1536), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf109, (1536,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf110, (1536,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf111, (1536,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf112, (6144, 1536), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf113, (6144,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf114, (1536, 6144), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf115, (1536,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1536,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf117, (1536,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf118, (1536, 1536), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf119, (1536,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf120, (1536, 1536), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf121, (1536,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf122, (1536, 1536), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf123, (1536,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf124, (1536, 1536), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf125, (1536,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf126, (1536,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf127, (1536,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf128, (6144, 1536), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf129, (6144,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf130, (1536, 6144), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf131, (1536,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1536,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf133, (1536,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf134, (1536, 1536), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf135, (1536,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf136, (1536, 1536), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf137, (1536,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf138, (1536, 1536), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf139, (1536,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf140, (1536, 1536), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf141, (1536,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf142, (1536,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf143, (1536,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf144, (6144, 1536), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf145, (6144,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf146, (1536, 6144), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf147, (1536,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf148, (1536,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf149, (1536,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf150, (1536, 1536), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf151, (1536,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf152, (1536, 1536), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf153, (1536,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf154, (1536, 1536), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf155, (1536,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf156, (1536, 1536), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf157, (1536,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf158, (1536,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf159, (1536,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf160, (6144, 1536), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf161, (6144,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf162, (1536, 6144), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf163, (1536,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf164, (1536,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf165, (1536,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf166, (1536, 1536), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf167, (1536,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf168, (1536, 1536), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf169, (1536,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf170, (1536, 1536), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf171, (1536,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf172, (1536, 1536), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf173, (1536,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf174, (1536,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf175, (1536,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf176, (6144, 1536), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf177, (6144,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf178, (1536, 6144), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf179, (1536,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf180, (1536,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf181, (1536,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf182, (1536, 1536), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf183, (1536,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf184, (1536, 1536), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf185, (1536,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf186, (1536, 1536), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf187, (1536,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf188, (1536, 1536), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf189, (1536,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf190, (1536,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf191, (1536,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf192, (6144, 1536), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf193, (6144,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf194, (1536, 6144), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf195, (1536,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf196, (1536,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf197, (1536,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf198, (1536, 1536), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf199, (1536,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf200, (1536, 1536), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf201, (1536,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf202, (1536, 1536), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf203, (1536,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf204, (1536, 1536), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf205, (1536,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf206, (1536,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf207, (1536,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf208, (6144, 1536), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf209, (6144,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf210, (1536, 6144), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf211, (1536,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf212, (1536,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf213, (1536,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf214, (1536, 1536), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf215, (1536,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf216, (1536, 1536), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf217, (1536,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf218, (1536, 1536), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf219, (1536,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf220, (1536, 1536), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf221, (1536,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf222, (1536,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf223, (1536,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf224, (6144, 1536), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf225, (6144,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf226, (1536, 6144), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf227, (1536,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf228, (1536,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf229, (1536,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf230, (1536, 1536), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf231, (1536,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf232, (1536, 1536), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf233, (1536,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf234, (1536, 1536), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf235, (1536,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf236, (1536, 1536), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf237, (1536,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf238, (1536,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf239, (1536,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf240, (6144, 1536), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf241, (6144,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf242, (1536, 6144), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf243, (1536,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf244, (1536,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf245, (1536,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf246, (1536, 1536), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf247, (1536,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf248, (1536, 1536), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf249, (1536,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf250, (1536, 1536), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf251, (1536,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf252, (1536, 1536), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf253, (1536,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf254, (1536,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf255, (1536,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf256, (6144, 1536), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf257, (6144,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf258, (1536, 6144), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf259, (1536,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf260, (1536,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf261, (1536,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf262, (1536, 1536), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf263, (1536,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf264, (1536, 1536), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf265, (1536,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf266, (1536, 1536), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf267, (1536,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf268, (1536, 1536), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf269, (1536,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf270, (1536,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf271, (1536,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf272, (6144, 1536), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf273, (6144,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf274, (1536, 6144), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf275, (1536,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf276, (1536,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf277, (1536,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf278, (1536, 1536), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf279, (1536,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf280, (1536, 1536), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf281, (1536,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf282, (1536, 1536), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf283, (1536,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf284, (1536, 1536), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf285, (1536,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf286, (1536,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf287, (1536,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf288, (6144, 1536), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf289, (6144,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf290, (1536, 6144), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf291, (1536,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf292, (1536,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf293, (1536,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf294, (1536, 1536), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf295, (1536,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf296, (1536, 1536), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf297, (1536,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf298, (1536, 1536), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf299, (1536,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf300, (1536, 1536), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf301, (1536,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf302, (1536,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf303, (1536,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf304, (6144, 1536), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf305, (6144,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf306, (1536, 6144), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf307, (1536,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf308, (1536,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf309, (1536,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf310, (1536, 1536), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf311, (1536,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf312, (1536, 1536), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf313, (1536,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf314, (1536, 1536), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf315, (1536,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf316, (1536, 1536), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf317, (1536,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf318, (1536,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf319, (1536,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf320, (6144, 1536), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf321, (6144,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf322, (1536, 6144), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf323, (1536,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf324, (1536,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf325, (1536,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf326, (1536, 1536), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf327, (1536,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf328, (1536, 1536), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf329, (1536,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf330, (1536, 1536), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf331, (1536,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf332, (1536, 1536), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf333, (1536,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf334, (1536,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf335, (1536,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf336, (6144, 1536), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf337, (6144,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf338, (1536, 6144), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf339, (1536,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf340, (1536,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf341, (1536,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf342, (1536, 1536), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf343, (1536,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf344, (1536, 1536), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf345, (1536,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf346, (1536, 1536), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf347, (1536,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf348, (1536, 1536), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf349, (1536,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf350, (1536,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf351, (1536,), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf352, (6144, 1536), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf353, (6144,), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf354, (1536, 6144), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf355, (1536,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf356, (1536,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf357, (1536,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf358, (1536, 1536), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf359, (1536,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf360, (1536, 1536), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf361, (1536,), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf362, (1536, 1536), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf363, (1536,), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf364, (1536, 1536), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf365, (1536,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf366, (1536,), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf367, (1536,), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf368, (6144, 1536), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf369, (6144,), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf370, (1536, 6144), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf371, (1536,), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf372, (1536,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf373, (1536,), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf374, (1536, 1536), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf375, (1536,), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf376, (1536, 1536), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf377, (1536,), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf378, (1536, 1536), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf379, (1536,), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf380, (1536, 1536), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf381, (1536,), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf382, (1536,), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf383, (1536,), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf384, (6144, 1536), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf385, (6144,), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 37748736, device=device(type='cuda', index=0))
    reader.tensor(buf386, (1536, 6144), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf387, (1536,), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf388, (1536,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf389, (1536,), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf390, (1536, 1536), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf391, (1536,), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf392, (1536,), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf393, (1536,), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 512400, device=device(type='cuda', index=0))
    reader.tensor(buf394, (128100,), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 8192, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf395, (2, 512), dtype=torch.int64, is_leaf=True)  # arg395_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)