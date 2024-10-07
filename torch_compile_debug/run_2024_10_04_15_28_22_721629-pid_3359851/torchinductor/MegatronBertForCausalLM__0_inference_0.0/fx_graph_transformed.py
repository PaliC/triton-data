class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 512]", arg1_1: "i64[4, 512]", arg2_1: "f32[29056, 1024]", arg3_1: "f32[512, 1024]", arg4_1: "f32[2, 1024]", arg5_1: "f32[1024]", arg6_1: "f32[1024]", arg7_1: "f32[1024, 1024]", arg8_1: "f32[1024]", arg9_1: "f32[1024, 1024]", arg10_1: "f32[1024]", arg11_1: "f32[1024, 1024]", arg12_1: "f32[1024]", arg13_1: "f32[1024, 1024]", arg14_1: "f32[1024]", arg15_1: "f32[1024]", arg16_1: "f32[1024]", arg17_1: "f32[4096, 1024]", arg18_1: "f32[4096]", arg19_1: "f32[1024, 4096]", arg20_1: "f32[1024]", arg21_1: "f32[1024]", arg22_1: "f32[1024]", arg23_1: "f32[1024, 1024]", arg24_1: "f32[1024]", arg25_1: "f32[1024, 1024]", arg26_1: "f32[1024]", arg27_1: "f32[1024, 1024]", arg28_1: "f32[1024]", arg29_1: "f32[1024, 1024]", arg30_1: "f32[1024]", arg31_1: "f32[1024]", arg32_1: "f32[1024]", arg33_1: "f32[4096, 1024]", arg34_1: "f32[4096]", arg35_1: "f32[1024, 4096]", arg36_1: "f32[1024]", arg37_1: "f32[1024]", arg38_1: "f32[1024]", arg39_1: "f32[1024, 1024]", arg40_1: "f32[1024]", arg41_1: "f32[1024, 1024]", arg42_1: "f32[1024]", arg43_1: "f32[1024, 1024]", arg44_1: "f32[1024]", arg45_1: "f32[1024, 1024]", arg46_1: "f32[1024]", arg47_1: "f32[1024]", arg48_1: "f32[1024]", arg49_1: "f32[4096, 1024]", arg50_1: "f32[4096]", arg51_1: "f32[1024, 4096]", arg52_1: "f32[1024]", arg53_1: "f32[1024]", arg54_1: "f32[1024]", arg55_1: "f32[1024, 1024]", arg56_1: "f32[1024]", arg57_1: "f32[1024, 1024]", arg58_1: "f32[1024]", arg59_1: "f32[1024, 1024]", arg60_1: "f32[1024]", arg61_1: "f32[1024, 1024]", arg62_1: "f32[1024]", arg63_1: "f32[1024]", arg64_1: "f32[1024]", arg65_1: "f32[4096, 1024]", arg66_1: "f32[4096]", arg67_1: "f32[1024, 4096]", arg68_1: "f32[1024]", arg69_1: "f32[1024]", arg70_1: "f32[1024]", arg71_1: "f32[1024, 1024]", arg72_1: "f32[1024]", arg73_1: "f32[1024, 1024]", arg74_1: "f32[1024]", arg75_1: "f32[1024, 1024]", arg76_1: "f32[1024]", arg77_1: "f32[1024, 1024]", arg78_1: "f32[1024]", arg79_1: "f32[1024]", arg80_1: "f32[1024]", arg81_1: "f32[4096, 1024]", arg82_1: "f32[4096]", arg83_1: "f32[1024, 4096]", arg84_1: "f32[1024]", arg85_1: "f32[1024]", arg86_1: "f32[1024]", arg87_1: "f32[1024, 1024]", arg88_1: "f32[1024]", arg89_1: "f32[1024, 1024]", arg90_1: "f32[1024]", arg91_1: "f32[1024, 1024]", arg92_1: "f32[1024]", arg93_1: "f32[1024, 1024]", arg94_1: "f32[1024]", arg95_1: "f32[1024]", arg96_1: "f32[1024]", arg97_1: "f32[4096, 1024]", arg98_1: "f32[4096]", arg99_1: "f32[1024, 4096]", arg100_1: "f32[1024]", arg101_1: "f32[1024]", arg102_1: "f32[1024]", arg103_1: "f32[1024, 1024]", arg104_1: "f32[1024]", arg105_1: "f32[1024, 1024]", arg106_1: "f32[1024]", arg107_1: "f32[1024, 1024]", arg108_1: "f32[1024]", arg109_1: "f32[1024, 1024]", arg110_1: "f32[1024]", arg111_1: "f32[1024]", arg112_1: "f32[1024]", arg113_1: "f32[4096, 1024]", arg114_1: "f32[4096]", arg115_1: "f32[1024, 4096]", arg116_1: "f32[1024]", arg117_1: "f32[1024]", arg118_1: "f32[1024]", arg119_1: "f32[1024, 1024]", arg120_1: "f32[1024]", arg121_1: "f32[1024, 1024]", arg122_1: "f32[1024]", arg123_1: "f32[1024, 1024]", arg124_1: "f32[1024]", arg125_1: "f32[1024, 1024]", arg126_1: "f32[1024]", arg127_1: "f32[1024]", arg128_1: "f32[1024]", arg129_1: "f32[4096, 1024]", arg130_1: "f32[4096]", arg131_1: "f32[1024, 4096]", arg132_1: "f32[1024]", arg133_1: "f32[1024]", arg134_1: "f32[1024]", arg135_1: "f32[1024, 1024]", arg136_1: "f32[1024]", arg137_1: "f32[1024, 1024]", arg138_1: "f32[1024]", arg139_1: "f32[1024, 1024]", arg140_1: "f32[1024]", arg141_1: "f32[1024, 1024]", arg142_1: "f32[1024]", arg143_1: "f32[1024]", arg144_1: "f32[1024]", arg145_1: "f32[4096, 1024]", arg146_1: "f32[4096]", arg147_1: "f32[1024, 4096]", arg148_1: "f32[1024]", arg149_1: "f32[1024]", arg150_1: "f32[1024]", arg151_1: "f32[1024, 1024]", arg152_1: "f32[1024]", arg153_1: "f32[1024, 1024]", arg154_1: "f32[1024]", arg155_1: "f32[1024, 1024]", arg156_1: "f32[1024]", arg157_1: "f32[1024, 1024]", arg158_1: "f32[1024]", arg159_1: "f32[1024]", arg160_1: "f32[1024]", arg161_1: "f32[4096, 1024]", arg162_1: "f32[4096]", arg163_1: "f32[1024, 4096]", arg164_1: "f32[1024]", arg165_1: "f32[1024]", arg166_1: "f32[1024]", arg167_1: "f32[1024, 1024]", arg168_1: "f32[1024]", arg169_1: "f32[1024, 1024]", arg170_1: "f32[1024]", arg171_1: "f32[1024, 1024]", arg172_1: "f32[1024]", arg173_1: "f32[1024, 1024]", arg174_1: "f32[1024]", arg175_1: "f32[1024]", arg176_1: "f32[1024]", arg177_1: "f32[4096, 1024]", arg178_1: "f32[4096]", arg179_1: "f32[1024, 4096]", arg180_1: "f32[1024]", arg181_1: "f32[1024]", arg182_1: "f32[1024]", arg183_1: "f32[1024, 1024]", arg184_1: "f32[1024]", arg185_1: "f32[1024, 1024]", arg186_1: "f32[1024]", arg187_1: "f32[1024, 1024]", arg188_1: "f32[1024]", arg189_1: "f32[1024, 1024]", arg190_1: "f32[1024]", arg191_1: "f32[1024]", arg192_1: "f32[1024]", arg193_1: "f32[4096, 1024]", arg194_1: "f32[4096]", arg195_1: "f32[1024, 4096]", arg196_1: "f32[1024]", arg197_1: "f32[1024]", arg198_1: "f32[1024]", arg199_1: "f32[1024, 1024]", arg200_1: "f32[1024]", arg201_1: "f32[1024, 1024]", arg202_1: "f32[1024]", arg203_1: "f32[1024, 1024]", arg204_1: "f32[1024]", arg205_1: "f32[1024, 1024]", arg206_1: "f32[1024]", arg207_1: "f32[1024]", arg208_1: "f32[1024]", arg209_1: "f32[4096, 1024]", arg210_1: "f32[4096]", arg211_1: "f32[1024, 4096]", arg212_1: "f32[1024]", arg213_1: "f32[1024]", arg214_1: "f32[1024]", arg215_1: "f32[1024, 1024]", arg216_1: "f32[1024]", arg217_1: "f32[1024, 1024]", arg218_1: "f32[1024]", arg219_1: "f32[1024, 1024]", arg220_1: "f32[1024]", arg221_1: "f32[1024, 1024]", arg222_1: "f32[1024]", arg223_1: "f32[1024]", arg224_1: "f32[1024]", arg225_1: "f32[4096, 1024]", arg226_1: "f32[4096]", arg227_1: "f32[1024, 4096]", arg228_1: "f32[1024]", arg229_1: "f32[1024]", arg230_1: "f32[1024]", arg231_1: "f32[1024, 1024]", arg232_1: "f32[1024]", arg233_1: "f32[1024, 1024]", arg234_1: "f32[1024]", arg235_1: "f32[1024, 1024]", arg236_1: "f32[1024]", arg237_1: "f32[1024, 1024]", arg238_1: "f32[1024]", arg239_1: "f32[1024]", arg240_1: "f32[1024]", arg241_1: "f32[4096, 1024]", arg242_1: "f32[4096]", arg243_1: "f32[1024, 4096]", arg244_1: "f32[1024]", arg245_1: "f32[1024]", arg246_1: "f32[1024]", arg247_1: "f32[1024, 1024]", arg248_1: "f32[1024]", arg249_1: "f32[1024, 1024]", arg250_1: "f32[1024]", arg251_1: "f32[1024, 1024]", arg252_1: "f32[1024]", arg253_1: "f32[1024, 1024]", arg254_1: "f32[1024]", arg255_1: "f32[1024]", arg256_1: "f32[1024]", arg257_1: "f32[4096, 1024]", arg258_1: "f32[4096]", arg259_1: "f32[1024, 4096]", arg260_1: "f32[1024]", arg261_1: "f32[1024]", arg262_1: "f32[1024]", arg263_1: "f32[1024, 1024]", arg264_1: "f32[1024]", arg265_1: "f32[1024, 1024]", arg266_1: "f32[1024]", arg267_1: "f32[1024, 1024]", arg268_1: "f32[1024]", arg269_1: "f32[1024, 1024]", arg270_1: "f32[1024]", arg271_1: "f32[1024]", arg272_1: "f32[1024]", arg273_1: "f32[4096, 1024]", arg274_1: "f32[4096]", arg275_1: "f32[1024, 4096]", arg276_1: "f32[1024]", arg277_1: "f32[1024]", arg278_1: "f32[1024]", arg279_1: "f32[1024, 1024]", arg280_1: "f32[1024]", arg281_1: "f32[1024, 1024]", arg282_1: "f32[1024]", arg283_1: "f32[1024, 1024]", arg284_1: "f32[1024]", arg285_1: "f32[1024, 1024]", arg286_1: "f32[1024]", arg287_1: "f32[1024]", arg288_1: "f32[1024]", arg289_1: "f32[4096, 1024]", arg290_1: "f32[4096]", arg291_1: "f32[1024, 4096]", arg292_1: "f32[1024]", arg293_1: "f32[1024]", arg294_1: "f32[1024]", arg295_1: "f32[1024, 1024]", arg296_1: "f32[1024]", arg297_1: "f32[1024, 1024]", arg298_1: "f32[1024]", arg299_1: "f32[1024, 1024]", arg300_1: "f32[1024]", arg301_1: "f32[1024, 1024]", arg302_1: "f32[1024]", arg303_1: "f32[1024]", arg304_1: "f32[1024]", arg305_1: "f32[4096, 1024]", arg306_1: "f32[4096]", arg307_1: "f32[1024, 4096]", arg308_1: "f32[1024]", arg309_1: "f32[1024]", arg310_1: "f32[1024]", arg311_1: "f32[1024, 1024]", arg312_1: "f32[1024]", arg313_1: "f32[1024, 1024]", arg314_1: "f32[1024]", arg315_1: "f32[1024, 1024]", arg316_1: "f32[1024]", arg317_1: "f32[1024, 1024]", arg318_1: "f32[1024]", arg319_1: "f32[1024]", arg320_1: "f32[1024]", arg321_1: "f32[4096, 1024]", arg322_1: "f32[4096]", arg323_1: "f32[1024, 4096]", arg324_1: "f32[1024]", arg325_1: "f32[1024]", arg326_1: "f32[1024]", arg327_1: "f32[1024, 1024]", arg328_1: "f32[1024]", arg329_1: "f32[1024, 1024]", arg330_1: "f32[1024]", arg331_1: "f32[1024, 1024]", arg332_1: "f32[1024]", arg333_1: "f32[1024, 1024]", arg334_1: "f32[1024]", arg335_1: "f32[1024]", arg336_1: "f32[1024]", arg337_1: "f32[4096, 1024]", arg338_1: "f32[4096]", arg339_1: "f32[1024, 4096]", arg340_1: "f32[1024]", arg341_1: "f32[1024]", arg342_1: "f32[1024]", arg343_1: "f32[1024, 1024]", arg344_1: "f32[1024]", arg345_1: "f32[1024, 1024]", arg346_1: "f32[1024]", arg347_1: "f32[1024, 1024]", arg348_1: "f32[1024]", arg349_1: "f32[1024, 1024]", arg350_1: "f32[1024]", arg351_1: "f32[1024]", arg352_1: "f32[1024]", arg353_1: "f32[4096, 1024]", arg354_1: "f32[4096]", arg355_1: "f32[1024, 4096]", arg356_1: "f32[1024]", arg357_1: "f32[1024]", arg358_1: "f32[1024]", arg359_1: "f32[1024, 1024]", arg360_1: "f32[1024]", arg361_1: "f32[1024, 1024]", arg362_1: "f32[1024]", arg363_1: "f32[1024, 1024]", arg364_1: "f32[1024]", arg365_1: "f32[1024, 1024]", arg366_1: "f32[1024]", arg367_1: "f32[1024]", arg368_1: "f32[1024]", arg369_1: "f32[4096, 1024]", arg370_1: "f32[4096]", arg371_1: "f32[1024, 4096]", arg372_1: "f32[1024]", arg373_1: "f32[1024]", arg374_1: "f32[1024]", arg375_1: "f32[1024, 1024]", arg376_1: "f32[1024]", arg377_1: "f32[1024, 1024]", arg378_1: "f32[1024]", arg379_1: "f32[1024, 1024]", arg380_1: "f32[1024]", arg381_1: "f32[1024, 1024]", arg382_1: "f32[1024]", arg383_1: "f32[1024]", arg384_1: "f32[1024]", arg385_1: "f32[4096, 1024]", arg386_1: "f32[4096]", arg387_1: "f32[1024, 4096]", arg388_1: "f32[1024]", arg389_1: "f32[1024]", arg390_1: "f32[1024]", arg391_1: "i64[1, 512]", arg392_1: "f32[1024, 1024]", arg393_1: "f32[1024]", arg394_1: "f32[1024]", arg395_1: "f32[1024]", arg396_1: "f32[29056]"):
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:939 in forward, code: attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        full: "f32[4, 512]" = torch.ops.aten.full.default([4, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_utils.py:1175 in get_extended_attention_mask, code: extended_attention_mask = attention_mask[:, None, None, :]
        unsqueeze: "f32[4, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1: "f32[4, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_utils.py:1187 in get_extended_attention_mask, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        sub: "f32[4, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = sub = None
        full_default_1: "f32[4, 1, 1, 512]" = torch.ops.aten.full.default([4, 1, 1, 512], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:174 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[4, 512, 1024]" = torch.ops.aten.embedding.default(arg2_1, arg1_1, 0);  arg1_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:941 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        full_default: "i64[4, 512]" = torch.ops.aten.full.default([4, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:175 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[4, 512, 1024]" = torch.ops.aten.embedding.default(arg4_1, full_default);  arg4_1 = full_default = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:177 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:179 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 512, 1024]" = torch.ops.aten.embedding.default(arg3_1, arg391_1);  arg3_1 = arg391_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:180 in forward, code: embeddings += position_embeddings
        add_1: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 512, 1]" = var_mean[0]
        getitem_1: "f32[4, 512, 1]" = var_mean[1];  var_mean = None
        sub_1: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  getitem_1 = None
        add_2: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_2: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        add_3: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_2, arg6_1);  mul_2 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_3, [2048, 1024])
        permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg8_1, view, permute);  arg8_1 = view = permute = None
        view_1: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm, [4, 512, 1024]);  addmm = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_8: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1, [4, 512, 16, 64]);  view_1 = None
        
        # No stacktrace found for following nodes
        permute_default_69: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_2: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_3, [2048, 1024])
        permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg10_1, view_2, permute_1);  arg10_1 = view_2 = permute_1 = None
        view_3: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_1, [4, 512, 1024]);  addmm_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_4: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_3, [4, 512, 16, 64]);  view_3 = None
        
        # No stacktrace found for following nodes
        permute_default_70: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_5: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_3, [2048, 1024]);  add_3 = None
        permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_2: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg12_1, view_5, permute_3);  arg12_1 = view_5 = permute_3 = None
        view_6: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_2, [4, 512, 1024]);  addmm_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_7: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_6, [4, 512, 16, 64]);  view_6 = None
        
        # No stacktrace found for following nodes
        permute_default_71: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        _scaled_dot_product_efficient_attention_default_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_69, permute_default_70, permute_default_71, None, False, scale = 0.125);  permute_default_69 = permute_default_70 = permute_default_71 = None
        getitem_123: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_23[0];  _scaled_dot_product_efficient_attention_default_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_7: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_123, [0, 2, 1, 3]);  getitem_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_15: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_7, [4, 512, 1024]);  permute_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_16: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_15, [2048, 1024]);  view_15 = None
        permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        
        # No stacktrace found for following nodes
        mm_default_72: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_16, permute_8);  view_16 = permute_8 = None
        add_tensor_72: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_72, arg14_1);  mm_default_72 = arg14_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_17: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_72, [4, 512, 1024]);  add_tensor_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_5: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_1, view_17);  add_1 = view_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2: "f32[4, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[4, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_3: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  getitem_3 = None
        add_6: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_3: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_4: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_3, arg15_1);  mul_3 = arg15_1 = None
        add_7: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_4, arg16_1);  mul_4 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_7, [2048, 1024]);  add_7 = None
        permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        
        # No stacktrace found for following nodes
        mm_default_71: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_18, permute_9);  view_18 = permute_9 = None
        add_tensor_71: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_71, arg18_1);  mm_default_71 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_19: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_71, [4, 512, 4096]);  add_tensor_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_5: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_6: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_8: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_20: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_7, [2048, 4096]);  mul_7 = None
        permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        
        # No stacktrace found for following nodes
        mm_default_70: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_20, permute_10);  view_20 = permute_10 = None
        add_tensor_70: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_70, arg20_1);  mm_default_70 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_21: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_70, [4, 512, 1024]);  add_tensor_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_9: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_5, view_21);  add_5 = view_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4: "f32[4, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[4, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_4: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  getitem_5 = None
        add_10: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_8: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_9: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_8, arg21_1);  mul_8 = arg21_1 = None
        add_11: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_9, arg22_1);  mul_9 = arg22_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_22: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_11, [2048, 1024])
        permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_6: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg24_1, view_22, permute_11);  arg24_1 = view_22 = permute_11 = None
        view_23: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_6, [4, 512, 1024]);  addmm_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_30: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_23, [4, 512, 16, 64]);  view_23 = None
        
        # No stacktrace found for following nodes
        permute_default_66: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_24: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_11, [2048, 1024])
        permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_7: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg26_1, view_24, permute_12);  arg26_1 = view_24 = permute_12 = None
        view_25: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_7, [4, 512, 1024]);  addmm_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_26: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_25, [4, 512, 16, 64]);  view_25 = None
        
        # No stacktrace found for following nodes
        permute_default_67: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_27: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_11, [2048, 1024]);  add_11 = None
        permute_14: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_8: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg28_1, view_27, permute_14);  arg28_1 = view_27 = permute_14 = None
        view_28: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_8, [4, 512, 1024]);  addmm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_29: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_28, [4, 512, 16, 64]);  view_28 = None
        
        # No stacktrace found for following nodes
        permute_default_68: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        _scaled_dot_product_efficient_attention_default_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_66, permute_default_67, permute_default_68, None, False, scale = 0.125);  permute_default_66 = permute_default_67 = permute_default_68 = None
        getitem_122: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_22[0];  _scaled_dot_product_efficient_attention_default_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_18: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_122, [0, 2, 1, 3]);  getitem_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_37: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_18, [4, 512, 1024]);  permute_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_38: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_37, [2048, 1024]);  view_37 = None
        permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        
        # No stacktrace found for following nodes
        mm_default_69: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_38, permute_19);  view_38 = permute_19 = None
        add_tensor_69: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_69, arg30_1);  mm_default_69 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_39: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_69, [4, 512, 1024]);  add_tensor_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_13: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_9, view_39);  add_9 = view_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_6: "f32[4, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[4, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_6: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_13, getitem_7);  getitem_7 = None
        add_14: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_10: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = rsqrt_3 = None
        mul_11: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_10, arg31_1);  mul_10 = arg31_1 = None
        add_15: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_11, arg32_1);  mul_11 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_15, [2048, 1024]);  add_15 = None
        permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        
        # No stacktrace found for following nodes
        mm_default_68: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_40, permute_20);  view_40 = permute_20 = None
        add_tensor_68: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_68, arg34_1);  mm_default_68 = arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_41: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_68, [4, 512, 4096]);  add_tensor_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_12: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_13: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
        erf_1: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_16: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_14: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_14, [2048, 4096]);  mul_14 = None
        permute_21: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        
        # No stacktrace found for following nodes
        mm_default_67: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_42, permute_21);  view_42 = permute_21 = None
        add_tensor_67: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_67, arg36_1);  mm_default_67 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_43: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_67, [4, 512, 1024]);  add_tensor_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_17: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_13, view_43);  add_13 = view_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8: "f32[4, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[4, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_7: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  getitem_9 = None
        add_18: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_15: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
        mul_16: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_15, arg37_1);  mul_15 = arg37_1 = None
        add_19: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_16, arg38_1);  mul_16 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_44: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_19, [2048, 1024])
        permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_12: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg40_1, view_44, permute_22);  arg40_1 = view_44 = permute_22 = None
        view_45: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_12, [4, 512, 1024]);  addmm_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_52: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_45, [4, 512, 16, 64]);  view_45 = None
        
        # No stacktrace found for following nodes
        permute_default_63: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_46: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_19, [2048, 1024])
        permute_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_13: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg42_1, view_46, permute_23);  arg42_1 = view_46 = permute_23 = None
        view_47: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_13, [4, 512, 1024]);  addmm_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_48: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_47, [4, 512, 16, 64]);  view_47 = None
        
        # No stacktrace found for following nodes
        permute_default_64: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_49: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_19, [2048, 1024]);  add_19 = None
        permute_25: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_14: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg44_1, view_49, permute_25);  arg44_1 = view_49 = permute_25 = None
        view_50: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_14, [4, 512, 1024]);  addmm_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_51: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_50, [4, 512, 16, 64]);  view_50 = None
        
        # No stacktrace found for following nodes
        permute_default_65: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        _scaled_dot_product_efficient_attention_default_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_63, permute_default_64, permute_default_65, None, False, scale = 0.125);  permute_default_63 = permute_default_64 = permute_default_65 = None
        getitem_121: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_21[0];  _scaled_dot_product_efficient_attention_default_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_29: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_121, [0, 2, 1, 3]);  getitem_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_59: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_29, [4, 512, 1024]);  permute_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_59, [2048, 1024]);  view_59 = None
        permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        
        # No stacktrace found for following nodes
        mm_default_66: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_60, permute_30);  view_60 = permute_30 = None
        add_tensor_66: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_66, arg46_1);  mm_default_66 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_61: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_66, [4, 512, 1024]);  add_tensor_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_21: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_17, view_61);  add_17 = view_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_10: "f32[4, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[4, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_9: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_21, getitem_11);  getitem_11 = None
        add_22: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_17: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = rsqrt_5 = None
        mul_18: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_17, arg47_1);  mul_17 = arg47_1 = None
        add_23: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_18, arg48_1);  mul_18 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_23, [2048, 1024]);  add_23 = None
        permute_31: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        
        # No stacktrace found for following nodes
        mm_default_65: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_62, permute_31);  view_62 = permute_31 = None
        add_tensor_65: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_65, arg50_1);  mm_default_65 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_63: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_65, [4, 512, 4096]);  add_tensor_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_19: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_20: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_2: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_24: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_19, add_24);  mul_19 = add_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_21, [2048, 4096]);  mul_21 = None
        permute_32: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        
        # No stacktrace found for following nodes
        mm_default_64: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_64, permute_32);  view_64 = permute_32 = None
        add_tensor_64: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_64, arg52_1);  mm_default_64 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_65: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_64, [4, 512, 1024]);  add_tensor_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_25: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_21, view_65);  add_21 = view_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12: "f32[4, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[4, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_10: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_25, getitem_13);  getitem_13 = None
        add_26: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_22: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
        mul_23: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_22, arg53_1);  mul_22 = arg53_1 = None
        add_27: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_23, arg54_1);  mul_23 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_66: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_27, [2048, 1024])
        permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_18: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg56_1, view_66, permute_33);  arg56_1 = view_66 = permute_33 = None
        view_67: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_18, [4, 512, 1024]);  addmm_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_74: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_67, [4, 512, 16, 64]);  view_67 = None
        
        # No stacktrace found for following nodes
        permute_default_60: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_68: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_27, [2048, 1024])
        permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_19: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg58_1, view_68, permute_34);  arg58_1 = view_68 = permute_34 = None
        view_69: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_19, [4, 512, 1024]);  addmm_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_70: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_69, [4, 512, 16, 64]);  view_69 = None
        
        # No stacktrace found for following nodes
        permute_default_61: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_71: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_27, [2048, 1024]);  add_27 = None
        permute_36: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_20: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg60_1, view_71, permute_36);  arg60_1 = view_71 = permute_36 = None
        view_72: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_20, [4, 512, 1024]);  addmm_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_73: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_72, [4, 512, 16, 64]);  view_72 = None
        
        # No stacktrace found for following nodes
        permute_default_62: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        _scaled_dot_product_efficient_attention_default_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_60, permute_default_61, permute_default_62, None, False, scale = 0.125);  permute_default_60 = permute_default_61 = permute_default_62 = None
        getitem_120: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_20[0];  _scaled_dot_product_efficient_attention_default_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_40: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_120, [0, 2, 1, 3]);  getitem_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_81: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_40, [4, 512, 1024]);  permute_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_82: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_81, [2048, 1024]);  view_81 = None
        permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        
        # No stacktrace found for following nodes
        mm_default_63: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_82, permute_41);  view_82 = permute_41 = None
        add_tensor_63: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_63, arg62_1);  mm_default_63 = arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_83: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_63, [4, 512, 1024]);  add_tensor_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_29: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_25, view_83);  add_25 = view_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_14: "f32[4, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[4, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_12: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_29, getitem_15);  getitem_15 = None
        add_30: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_24: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = rsqrt_7 = None
        mul_25: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_24, arg63_1);  mul_24 = arg63_1 = None
        add_31: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_25, arg64_1);  mul_25 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_31, [2048, 1024]);  add_31 = None
        permute_42: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        
        # No stacktrace found for following nodes
        mm_default_62: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_84, permute_42);  view_84 = permute_42 = None
        add_tensor_62: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_62, arg66_1);  mm_default_62 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_85: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_62, [4, 512, 4096]);  add_tensor_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_26: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_27: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
        erf_3: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_32: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_28: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_86: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_28, [2048, 4096]);  mul_28 = None
        permute_43: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        
        # No stacktrace found for following nodes
        mm_default_61: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_86, permute_43);  view_86 = permute_43 = None
        add_tensor_61: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_61, arg68_1);  mm_default_61 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_87: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_61, [4, 512, 1024]);  add_tensor_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_33: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_29, view_87);  add_29 = view_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16: "f32[4, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[4, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_13: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  getitem_17 = None
        add_34: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_29: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
        mul_30: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_29, arg69_1);  mul_29 = arg69_1 = None
        add_35: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_30, arg70_1);  mul_30 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_88: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_35, [2048, 1024])
        permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_24: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg72_1, view_88, permute_44);  arg72_1 = view_88 = permute_44 = None
        view_89: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_24, [4, 512, 1024]);  addmm_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_96: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_89, [4, 512, 16, 64]);  view_89 = None
        
        # No stacktrace found for following nodes
        permute_default_57: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_90: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_35, [2048, 1024])
        permute_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_25: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg74_1, view_90, permute_45);  arg74_1 = view_90 = permute_45 = None
        view_91: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_25, [4, 512, 1024]);  addmm_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_92: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_91, [4, 512, 16, 64]);  view_91 = None
        
        # No stacktrace found for following nodes
        permute_default_58: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_93: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_35, [2048, 1024]);  add_35 = None
        permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_26: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg76_1, view_93, permute_47);  arg76_1 = view_93 = permute_47 = None
        view_94: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_26, [4, 512, 1024]);  addmm_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_95: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_94, [4, 512, 16, 64]);  view_94 = None
        
        # No stacktrace found for following nodes
        permute_default_59: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        _scaled_dot_product_efficient_attention_default_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_57, permute_default_58, permute_default_59, None, False, scale = 0.125);  permute_default_57 = permute_default_58 = permute_default_59 = None
        getitem_119: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_19[0];  _scaled_dot_product_efficient_attention_default_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_51: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_103: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_51, [4, 512, 1024]);  permute_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_103, [2048, 1024]);  view_103 = None
        permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        
        # No stacktrace found for following nodes
        mm_default_60: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_104, permute_52);  view_104 = permute_52 = None
        add_tensor_60: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_60, arg78_1);  mm_default_60 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_105: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_60, [4, 512, 1024]);  add_tensor_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_37: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_33, view_105);  add_33 = view_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_18: "f32[4, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[4, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_15: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_37, getitem_19);  getitem_19 = None
        add_38: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_31: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = rsqrt_9 = None
        mul_32: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_31, arg79_1);  mul_31 = arg79_1 = None
        add_39: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_32, arg80_1);  mul_32 = arg80_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_39, [2048, 1024]);  add_39 = None
        permute_53: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        
        # No stacktrace found for following nodes
        mm_default_59: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_106, permute_53);  view_106 = permute_53 = None
        add_tensor_59: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_59, arg82_1);  mm_default_59 = arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_107: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_59, [4, 512, 4096]);  add_tensor_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_33: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_34: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
        erf_4: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_40: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_35: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_35, [2048, 4096]);  mul_35 = None
        permute_54: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        
        # No stacktrace found for following nodes
        mm_default_58: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_108, permute_54);  view_108 = permute_54 = None
        add_tensor_58: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_58, arg84_1);  mm_default_58 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_109: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_58, [4, 512, 1024]);  add_tensor_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_41: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_37, view_109);  add_37 = view_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20: "f32[4, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[4, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_16: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_41, getitem_21);  getitem_21 = None
        add_42: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_36: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
        mul_37: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_36, arg85_1);  mul_36 = arg85_1 = None
        add_43: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_37, arg86_1);  mul_37 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_110: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_43, [2048, 1024])
        permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_30: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg88_1, view_110, permute_55);  arg88_1 = view_110 = permute_55 = None
        view_111: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_30, [4, 512, 1024]);  addmm_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_118: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_111, [4, 512, 16, 64]);  view_111 = None
        
        # No stacktrace found for following nodes
        permute_default_54: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_112: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_43, [2048, 1024])
        permute_56: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_31: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg90_1, view_112, permute_56);  arg90_1 = view_112 = permute_56 = None
        view_113: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_31, [4, 512, 1024]);  addmm_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_114: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_113, [4, 512, 16, 64]);  view_113 = None
        
        # No stacktrace found for following nodes
        permute_default_55: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_115: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_43, [2048, 1024]);  add_43 = None
        permute_58: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_32: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg92_1, view_115, permute_58);  arg92_1 = view_115 = permute_58 = None
        view_116: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_32, [4, 512, 1024]);  addmm_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_117: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_116, [4, 512, 16, 64]);  view_116 = None
        
        # No stacktrace found for following nodes
        permute_default_56: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        _scaled_dot_product_efficient_attention_default_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_54, permute_default_55, permute_default_56, None, False, scale = 0.125);  permute_default_54 = permute_default_55 = permute_default_56 = None
        getitem_118: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_18[0];  _scaled_dot_product_efficient_attention_default_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_62: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_118, [0, 2, 1, 3]);  getitem_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_125: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_62, [4, 512, 1024]);  permute_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_125, [2048, 1024]);  view_125 = None
        permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        
        # No stacktrace found for following nodes
        mm_default_57: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_126, permute_63);  view_126 = permute_63 = None
        add_tensor_57: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_57, arg94_1);  mm_default_57 = arg94_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_127: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_57, [4, 512, 1024]);  add_tensor_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_45: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_41, view_127);  add_41 = view_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_22: "f32[4, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[4, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_18: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_45, getitem_23);  getitem_23 = None
        add_46: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_38: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = rsqrt_11 = None
        mul_39: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_38, arg95_1);  mul_38 = arg95_1 = None
        add_47: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_39, arg96_1);  mul_39 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_128: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_47, [2048, 1024]);  add_47 = None
        permute_64: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        
        # No stacktrace found for following nodes
        mm_default_56: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_128, permute_64);  view_128 = permute_64 = None
        add_tensor_56: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_56, arg98_1);  mm_default_56 = arg98_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_129: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_56, [4, 512, 4096]);  add_tensor_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_40: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_41: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_5: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_48: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_42: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_40, add_48);  mul_40 = add_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_130: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_42, [2048, 4096]);  mul_42 = None
        permute_65: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        
        # No stacktrace found for following nodes
        mm_default_55: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_130, permute_65);  view_130 = permute_65 = None
        add_tensor_55: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_55, arg100_1);  mm_default_55 = arg100_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_131: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_55, [4, 512, 1024]);  add_tensor_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_49: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_45, view_131);  add_45 = view_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24: "f32[4, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[4, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_19: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_49, getitem_25);  getitem_25 = None
        add_50: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_43: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
        mul_44: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_43, arg101_1);  mul_43 = arg101_1 = None
        add_51: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_44, arg102_1);  mul_44 = arg102_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_132: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_51, [2048, 1024])
        permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_36: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg104_1, view_132, permute_66);  arg104_1 = view_132 = permute_66 = None
        view_133: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_36, [4, 512, 1024]);  addmm_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_140: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_133, [4, 512, 16, 64]);  view_133 = None
        
        # No stacktrace found for following nodes
        permute_default_51: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_134: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_51, [2048, 1024])
        permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_37: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg106_1, view_134, permute_67);  arg106_1 = view_134 = permute_67 = None
        view_135: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_37, [4, 512, 1024]);  addmm_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_136: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_135, [4, 512, 16, 64]);  view_135 = None
        
        # No stacktrace found for following nodes
        permute_default_52: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_137: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_51, [2048, 1024]);  add_51 = None
        permute_69: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_38: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg108_1, view_137, permute_69);  arg108_1 = view_137 = permute_69 = None
        view_138: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_38, [4, 512, 1024]);  addmm_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_139: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_138, [4, 512, 16, 64]);  view_138 = None
        
        # No stacktrace found for following nodes
        permute_default_53: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
        _scaled_dot_product_efficient_attention_default_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_51, permute_default_52, permute_default_53, None, False, scale = 0.125);  permute_default_51 = permute_default_52 = permute_default_53 = None
        getitem_117: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_17[0];  _scaled_dot_product_efficient_attention_default_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_73: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_147: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_73, [4, 512, 1024]);  permute_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_148: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_147, [2048, 1024]);  view_147 = None
        permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        
        # No stacktrace found for following nodes
        mm_default_54: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_148, permute_74);  view_148 = permute_74 = None
        add_tensor_54: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_54, arg110_1);  mm_default_54 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_149: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_54, [4, 512, 1024]);  add_tensor_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_53: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_49, view_149);  add_49 = view_149 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_26: "f32[4, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[4, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_21: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_53, getitem_27);  getitem_27 = None
        add_54: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_45: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = rsqrt_13 = None
        mul_46: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_45, arg111_1);  mul_45 = arg111_1 = None
        add_55: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_46, arg112_1);  mul_46 = arg112_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_150: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_55, [2048, 1024]);  add_55 = None
        permute_75: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        
        # No stacktrace found for following nodes
        mm_default_53: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_150, permute_75);  view_150 = permute_75 = None
        add_tensor_53: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_53, arg114_1);  mm_default_53 = arg114_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_151: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_53, [4, 512, 4096]);  add_tensor_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_47: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_48: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
        erf_6: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_56: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_49: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_47, add_56);  mul_47 = add_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_49, [2048, 4096]);  mul_49 = None
        permute_76: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        
        # No stacktrace found for following nodes
        mm_default_52: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_152, permute_76);  view_152 = permute_76 = None
        add_tensor_52: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_52, arg116_1);  mm_default_52 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_153: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_52, [4, 512, 1024]);  add_tensor_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_57: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_53, view_153);  add_53 = view_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28: "f32[4, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[4, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_22: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_57, getitem_29);  getitem_29 = None
        add_58: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_50: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
        mul_51: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_50, arg117_1);  mul_50 = arg117_1 = None
        add_59: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_51, arg118_1);  mul_51 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_154: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_59, [2048, 1024])
        permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_42: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg120_1, view_154, permute_77);  arg120_1 = view_154 = permute_77 = None
        view_155: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_42, [4, 512, 1024]);  addmm_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_162: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_155, [4, 512, 16, 64]);  view_155 = None
        
        # No stacktrace found for following nodes
        permute_default_48: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_156: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_59, [2048, 1024])
        permute_78: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_43: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg122_1, view_156, permute_78);  arg122_1 = view_156 = permute_78 = None
        view_157: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_43, [4, 512, 1024]);  addmm_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_158: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_157, [4, 512, 16, 64]);  view_157 = None
        
        # No stacktrace found for following nodes
        permute_default_49: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_159: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_59, [2048, 1024]);  add_59 = None
        permute_80: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_44: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg124_1, view_159, permute_80);  arg124_1 = view_159 = permute_80 = None
        view_160: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_44, [4, 512, 1024]);  addmm_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_161: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_160, [4, 512, 16, 64]);  view_160 = None
        
        # No stacktrace found for following nodes
        permute_default_50: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
        _scaled_dot_product_efficient_attention_default_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_48, permute_default_49, permute_default_50, None, False, scale = 0.125);  permute_default_48 = permute_default_49 = permute_default_50 = None
        getitem_116: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_16[0];  _scaled_dot_product_efficient_attention_default_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_84: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_116, [0, 2, 1, 3]);  getitem_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_169: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_84, [4, 512, 1024]);  permute_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_169, [2048, 1024]);  view_169 = None
        permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        
        # No stacktrace found for following nodes
        mm_default_51: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_170, permute_85);  view_170 = permute_85 = None
        add_tensor_51: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_51, arg126_1);  mm_default_51 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_171: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_51, [4, 512, 1024]);  add_tensor_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_61: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_57, view_171);  add_57 = view_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_30: "f32[4, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[4, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_24: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_61, getitem_31);  getitem_31 = None
        add_62: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_52: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = rsqrt_15 = None
        mul_53: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_52, arg127_1);  mul_52 = arg127_1 = None
        add_63: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_53, arg128_1);  mul_53 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_63, [2048, 1024]);  add_63 = None
        permute_86: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        
        # No stacktrace found for following nodes
        mm_default_50: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_172, permute_86);  view_172 = permute_86 = None
        add_tensor_50: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_50, arg130_1);  mm_default_50 = arg130_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_173: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_50, [4, 512, 4096]);  add_tensor_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_54: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_55: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_7: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_64: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_56: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_54, add_64);  mul_54 = add_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_56, [2048, 4096]);  mul_56 = None
        permute_87: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        
        # No stacktrace found for following nodes
        mm_default_49: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_174, permute_87);  view_174 = permute_87 = None
        add_tensor_49: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_49, arg132_1);  mm_default_49 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_175: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_49, [4, 512, 1024]);  add_tensor_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_65: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_61, view_175);  add_61 = view_175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32: "f32[4, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[4, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_25: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_65, getitem_33);  getitem_33 = None
        add_66: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_57: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
        mul_58: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_57, arg133_1);  mul_57 = arg133_1 = None
        add_67: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_58, arg134_1);  mul_58 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_176: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_67, [2048, 1024])
        permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_48: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg136_1, view_176, permute_88);  arg136_1 = view_176 = permute_88 = None
        view_177: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_48, [4, 512, 1024]);  addmm_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_184: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_177, [4, 512, 16, 64]);  view_177 = None
        
        # No stacktrace found for following nodes
        permute_default_45: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_178: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_67, [2048, 1024])
        permute_89: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_49: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg138_1, view_178, permute_89);  arg138_1 = view_178 = permute_89 = None
        view_179: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_49, [4, 512, 1024]);  addmm_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_180: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_179, [4, 512, 16, 64]);  view_179 = None
        
        # No stacktrace found for following nodes
        permute_default_46: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_181: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_67, [2048, 1024]);  add_67 = None
        permute_91: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_50: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg140_1, view_181, permute_91);  arg140_1 = view_181 = permute_91 = None
        view_182: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_50, [4, 512, 1024]);  addmm_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_183: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_182, [4, 512, 16, 64]);  view_182 = None
        
        # No stacktrace found for following nodes
        permute_default_47: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        _scaled_dot_product_efficient_attention_default_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_45, permute_default_46, permute_default_47, None, False, scale = 0.125);  permute_default_45 = permute_default_46 = permute_default_47 = None
        getitem_115: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_15[0];  _scaled_dot_product_efficient_attention_default_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_95: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_191: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_95, [4, 512, 1024]);  permute_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_192: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_191, [2048, 1024]);  view_191 = None
        permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        
        # No stacktrace found for following nodes
        mm_default_48: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_192, permute_96);  view_192 = permute_96 = None
        add_tensor_48: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_48, arg142_1);  mm_default_48 = arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_193: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_48, [4, 512, 1024]);  add_tensor_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_69: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_65, view_193);  add_65 = view_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_34: "f32[4, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[4, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_27: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_69, getitem_35);  getitem_35 = None
        add_70: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_59: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = rsqrt_17 = None
        mul_60: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_59, arg143_1);  mul_59 = arg143_1 = None
        add_71: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_60, arg144_1);  mul_60 = arg144_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_71, [2048, 1024]);  add_71 = None
        permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        
        # No stacktrace found for following nodes
        mm_default_47: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_194, permute_97);  view_194 = permute_97 = None
        add_tensor_47: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_47, arg146_1);  mm_default_47 = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_195: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_47, [4, 512, 4096]);  add_tensor_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_61: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_62: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
        erf_8: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_72: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_63: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_63, [2048, 4096]);  mul_63 = None
        permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        
        # No stacktrace found for following nodes
        mm_default_46: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_196, permute_98);  view_196 = permute_98 = None
        add_tensor_46: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_46, arg148_1);  mm_default_46 = arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_197: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_46, [4, 512, 1024]);  add_tensor_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_73: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_69, view_197);  add_69 = view_197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_36: "f32[4, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[4, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_28: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_73, getitem_37);  getitem_37 = None
        add_74: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_64: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
        mul_65: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_64, arg149_1);  mul_64 = arg149_1 = None
        add_75: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_65, arg150_1);  mul_65 = arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_198: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_75, [2048, 1024])
        permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_54: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg152_1, view_198, permute_99);  arg152_1 = view_198 = permute_99 = None
        view_199: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_54, [4, 512, 1024]);  addmm_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_206: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_199, [4, 512, 16, 64]);  view_199 = None
        
        # No stacktrace found for following nodes
        permute_default_42: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_200: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_75, [2048, 1024])
        permute_100: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_55: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg154_1, view_200, permute_100);  arg154_1 = view_200 = permute_100 = None
        view_201: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_55, [4, 512, 1024]);  addmm_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_202: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_201, [4, 512, 16, 64]);  view_201 = None
        
        # No stacktrace found for following nodes
        permute_default_43: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_203: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_75, [2048, 1024]);  add_75 = None
        permute_102: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_56: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg156_1, view_203, permute_102);  arg156_1 = view_203 = permute_102 = None
        view_204: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_56, [4, 512, 1024]);  addmm_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_205: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_204, [4, 512, 16, 64]);  view_204 = None
        
        # No stacktrace found for following nodes
        permute_default_44: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        _scaled_dot_product_efficient_attention_default_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_42, permute_default_43, permute_default_44, None, False, scale = 0.125);  permute_default_42 = permute_default_43 = permute_default_44 = None
        getitem_114: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_14[0];  _scaled_dot_product_efficient_attention_default_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_106: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_114, [0, 2, 1, 3]);  getitem_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_213: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_106, [4, 512, 1024]);  permute_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_214: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_213, [2048, 1024]);  view_213 = None
        permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        
        # No stacktrace found for following nodes
        mm_default_45: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_214, permute_107);  view_214 = permute_107 = None
        add_tensor_45: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_45, arg158_1);  mm_default_45 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_215: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_45, [4, 512, 1024]);  add_tensor_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_77: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_73, view_215);  add_73 = view_215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_38: "f32[4, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[4, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_30: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_77, getitem_39);  getitem_39 = None
        add_78: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_66: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = rsqrt_19 = None
        mul_67: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_66, arg159_1);  mul_66 = arg159_1 = None
        add_79: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_67, arg160_1);  mul_67 = arg160_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_216: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_79, [2048, 1024]);  add_79 = None
        permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        
        # No stacktrace found for following nodes
        mm_default_44: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_216, permute_108);  view_216 = permute_108 = None
        add_tensor_44: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_44, arg162_1);  mm_default_44 = arg162_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_217: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_44, [4, 512, 4096]);  add_tensor_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_68: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_69: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
        erf_9: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_80: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_70: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_218: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_70, [2048, 4096]);  mul_70 = None
        permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        
        # No stacktrace found for following nodes
        mm_default_43: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_218, permute_109);  view_218 = permute_109 = None
        add_tensor_43: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_43, arg164_1);  mm_default_43 = arg164_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_219: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_43, [4, 512, 1024]);  add_tensor_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_81: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_77, view_219);  add_77 = view_219 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_40: "f32[4, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[4, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_31: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_81, getitem_41);  getitem_41 = None
        add_82: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_71: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
        mul_72: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_71, arg165_1);  mul_71 = arg165_1 = None
        add_83: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_72, arg166_1);  mul_72 = arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_220: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_83, [2048, 1024])
        permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_60: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg168_1, view_220, permute_110);  arg168_1 = view_220 = permute_110 = None
        view_221: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_60, [4, 512, 1024]);  addmm_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_228: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_221, [4, 512, 16, 64]);  view_221 = None
        
        # No stacktrace found for following nodes
        permute_default_39: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_222: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_83, [2048, 1024])
        permute_111: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_61: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg170_1, view_222, permute_111);  arg170_1 = view_222 = permute_111 = None
        view_223: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_61, [4, 512, 1024]);  addmm_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_224: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_223, [4, 512, 16, 64]);  view_223 = None
        
        # No stacktrace found for following nodes
        permute_default_40: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_225: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_83, [2048, 1024]);  add_83 = None
        permute_113: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_62: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg172_1, view_225, permute_113);  arg172_1 = view_225 = permute_113 = None
        view_226: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_62, [4, 512, 1024]);  addmm_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_227: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_226, [4, 512, 16, 64]);  view_226 = None
        
        # No stacktrace found for following nodes
        permute_default_41: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
        _scaled_dot_product_efficient_attention_default_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_39, permute_default_40, permute_default_41, None, False, scale = 0.125);  permute_default_39 = permute_default_40 = permute_default_41 = None
        getitem_113: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_13[0];  _scaled_dot_product_efficient_attention_default_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_117: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_113, [0, 2, 1, 3]);  getitem_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_235: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_117, [4, 512, 1024]);  permute_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_236: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_235, [2048, 1024]);  view_235 = None
        permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        
        # No stacktrace found for following nodes
        mm_default_42: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_236, permute_118);  view_236 = permute_118 = None
        add_tensor_42: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_42, arg174_1);  mm_default_42 = arg174_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_237: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_42, [4, 512, 1024]);  add_tensor_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_85: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_81, view_237);  add_81 = view_237 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_42: "f32[4, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[4, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_33: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_85, getitem_43);  getitem_43 = None
        add_86: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_73: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = rsqrt_21 = None
        mul_74: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_73, arg175_1);  mul_73 = arg175_1 = None
        add_87: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_74, arg176_1);  mul_74 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_238: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_87, [2048, 1024]);  add_87 = None
        permute_119: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        
        # No stacktrace found for following nodes
        mm_default_41: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_238, permute_119);  view_238 = permute_119 = None
        add_tensor_41: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_41, arg178_1);  mm_default_41 = arg178_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_239: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_41, [4, 512, 4096]);  add_tensor_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_75: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_76: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_10: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_88: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_77: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_75, add_88);  mul_75 = add_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_240: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_77, [2048, 4096]);  mul_77 = None
        permute_120: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        
        # No stacktrace found for following nodes
        mm_default_40: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_240, permute_120);  view_240 = permute_120 = None
        add_tensor_40: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_40, arg180_1);  mm_default_40 = arg180_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_241: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_40, [4, 512, 1024]);  add_tensor_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_89: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_85, view_241);  add_85 = view_241 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_44: "f32[4, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[4, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_34: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_89, getitem_45);  getitem_45 = None
        add_90: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_78: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
        mul_79: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_78, arg181_1);  mul_78 = arg181_1 = None
        add_91: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_79, arg182_1);  mul_79 = arg182_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_242: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_91, [2048, 1024])
        permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_66: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg184_1, view_242, permute_121);  arg184_1 = view_242 = permute_121 = None
        view_243: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_66, [4, 512, 1024]);  addmm_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_250: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_243, [4, 512, 16, 64]);  view_243 = None
        
        # No stacktrace found for following nodes
        permute_default_36: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_244: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_91, [2048, 1024])
        permute_122: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_67: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg186_1, view_244, permute_122);  arg186_1 = view_244 = permute_122 = None
        view_245: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_67, [4, 512, 1024]);  addmm_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_246: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_245, [4, 512, 16, 64]);  view_245 = None
        
        # No stacktrace found for following nodes
        permute_default_37: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_247: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_91, [2048, 1024]);  add_91 = None
        permute_124: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_68: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg188_1, view_247, permute_124);  arg188_1 = view_247 = permute_124 = None
        view_248: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_68, [4, 512, 1024]);  addmm_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_249: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_248, [4, 512, 16, 64]);  view_248 = None
        
        # No stacktrace found for following nodes
        permute_default_38: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
        _scaled_dot_product_efficient_attention_default_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_36, permute_default_37, permute_default_38, None, False, scale = 0.125);  permute_default_36 = permute_default_37 = permute_default_38 = None
        getitem_112: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_12[0];  _scaled_dot_product_efficient_attention_default_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_128: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_112, [0, 2, 1, 3]);  getitem_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_257: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_128, [4, 512, 1024]);  permute_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_258: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_257, [2048, 1024]);  view_257 = None
        permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        
        # No stacktrace found for following nodes
        mm_default_39: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_258, permute_129);  view_258 = permute_129 = None
        add_tensor_39: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_39, arg190_1);  mm_default_39 = arg190_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_259: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_39, [4, 512, 1024]);  add_tensor_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_93: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_89, view_259);  add_89 = view_259 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_46: "f32[4, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[4, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_36: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_93, getitem_47);  getitem_47 = None
        add_94: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_80: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = rsqrt_23 = None
        mul_81: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_80, arg191_1);  mul_80 = arg191_1 = None
        add_95: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_81, arg192_1);  mul_81 = arg192_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_260: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_95, [2048, 1024]);  add_95 = None
        permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        
        # No stacktrace found for following nodes
        mm_default_38: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_260, permute_130);  view_260 = permute_130 = None
        add_tensor_38: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_38, arg194_1);  mm_default_38 = arg194_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_261: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_38, [4, 512, 4096]);  add_tensor_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_82: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_83: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
        erf_11: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_96: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_84: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_82, add_96);  mul_82 = add_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_262: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_84, [2048, 4096]);  mul_84 = None
        permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        
        # No stacktrace found for following nodes
        mm_default_37: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_262, permute_131);  view_262 = permute_131 = None
        add_tensor_37: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_37, arg196_1);  mm_default_37 = arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_263: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_37, [4, 512, 1024]);  add_tensor_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_97: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_93, view_263);  add_93 = view_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48: "f32[4, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[4, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_37: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_97, getitem_49);  getitem_49 = None
        add_98: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_85: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
        mul_86: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_85, arg197_1);  mul_85 = arg197_1 = None
        add_99: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_86, arg198_1);  mul_86 = arg198_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_264: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_99, [2048, 1024])
        permute_132: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_72: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg200_1, view_264, permute_132);  arg200_1 = view_264 = permute_132 = None
        view_265: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_72, [4, 512, 1024]);  addmm_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_272: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_265, [4, 512, 16, 64]);  view_265 = None
        
        # No stacktrace found for following nodes
        permute_default_33: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_266: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_99, [2048, 1024])
        permute_133: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_73: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg202_1, view_266, permute_133);  arg202_1 = view_266 = permute_133 = None
        view_267: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_73, [4, 512, 1024]);  addmm_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_268: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_267, [4, 512, 16, 64]);  view_267 = None
        
        # No stacktrace found for following nodes
        permute_default_34: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_269: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_99, [2048, 1024]);  add_99 = None
        permute_135: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_74: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg204_1, view_269, permute_135);  arg204_1 = view_269 = permute_135 = None
        view_270: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_74, [4, 512, 1024]);  addmm_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_271: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_270, [4, 512, 16, 64]);  view_270 = None
        
        # No stacktrace found for following nodes
        permute_default_35: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None
        _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_33, permute_default_34, permute_default_35, None, False, scale = 0.125);  permute_default_33 = permute_default_34 = permute_default_35 = None
        getitem_111: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_139: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_111, [0, 2, 1, 3]);  getitem_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_279: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_139, [4, 512, 1024]);  permute_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_280: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_279, [2048, 1024]);  view_279 = None
        permute_140: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        
        # No stacktrace found for following nodes
        mm_default_36: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_280, permute_140);  view_280 = permute_140 = None
        add_tensor_36: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_36, arg206_1);  mm_default_36 = arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_281: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_36, [4, 512, 1024]);  add_tensor_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_101: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_97, view_281);  add_97 = view_281 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_25 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_50: "f32[4, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[4, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_39: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_101, getitem_51);  getitem_51 = None
        add_102: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_87: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_25);  sub_39 = rsqrt_25 = None
        mul_88: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_87, arg207_1);  mul_87 = arg207_1 = None
        add_103: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_88, arg208_1);  mul_88 = arg208_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_282: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_103, [2048, 1024]);  add_103 = None
        permute_141: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        
        # No stacktrace found for following nodes
        mm_default_35: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_282, permute_141);  view_282 = permute_141 = None
        add_tensor_35: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_35, arg210_1);  mm_default_35 = arg210_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_283: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_35, [4, 512, 4096]);  add_tensor_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_89: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, 0.5)
        mul_90: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476);  view_283 = None
        erf_12: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_104: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_91: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_89, add_104);  mul_89 = add_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_284: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_91, [2048, 4096]);  mul_91 = None
        permute_142: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        
        # No stacktrace found for following nodes
        mm_default_34: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_284, permute_142);  view_284 = permute_142 = None
        add_tensor_34: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_34, arg212_1);  mm_default_34 = arg212_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_285: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_34, [4, 512, 1024]);  add_tensor_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_105: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_101, view_285);  add_101 = view_285 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_26 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_52: "f32[4, 512, 1]" = var_mean_26[0]
        getitem_53: "f32[4, 512, 1]" = var_mean_26[1];  var_mean_26 = None
        sub_40: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_105, getitem_53);  getitem_53 = None
        add_106: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_26: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_92: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = rsqrt_26 = None
        mul_93: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_92, arg213_1);  mul_92 = arg213_1 = None
        add_107: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_93, arg214_1);  mul_93 = arg214_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_286: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_107, [2048, 1024])
        permute_143: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_78: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg216_1, view_286, permute_143);  arg216_1 = view_286 = permute_143 = None
        view_287: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_78, [4, 512, 1024]);  addmm_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_294: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_287, [4, 512, 16, 64]);  view_287 = None
        
        # No stacktrace found for following nodes
        permute_default_30: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_288: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_107, [2048, 1024])
        permute_144: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_79: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg218_1, view_288, permute_144);  arg218_1 = view_288 = permute_144 = None
        view_289: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_79, [4, 512, 1024]);  addmm_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_290: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_289, [4, 512, 16, 64]);  view_289 = None
        
        # No stacktrace found for following nodes
        permute_default_31: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_291: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_107, [2048, 1024]);  add_107 = None
        permute_146: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_80: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg220_1, view_291, permute_146);  arg220_1 = view_291 = permute_146 = None
        view_292: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_80, [4, 512, 1024]);  addmm_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_293: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_292, [4, 512, 16, 64]);  view_292 = None
        
        # No stacktrace found for following nodes
        permute_default_32: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
        _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_30, permute_default_31, permute_default_32, None, False, scale = 0.125);  permute_default_30 = permute_default_31 = permute_default_32 = None
        getitem_110: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_150: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_110, [0, 2, 1, 3]);  getitem_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_301: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_150, [4, 512, 1024]);  permute_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_302: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_301, [2048, 1024]);  view_301 = None
        permute_151: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        
        # No stacktrace found for following nodes
        mm_default_33: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_302, permute_151);  view_302 = permute_151 = None
        add_tensor_33: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_33, arg222_1);  mm_default_33 = arg222_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_303: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_33, [4, 512, 1024]);  add_tensor_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_109: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_105, view_303);  add_105 = view_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_27 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
        getitem_54: "f32[4, 512, 1]" = var_mean_27[0]
        getitem_55: "f32[4, 512, 1]" = var_mean_27[1];  var_mean_27 = None
        sub_42: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_109, getitem_55);  getitem_55 = None
        add_110: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_27: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        mul_94: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_27);  sub_42 = rsqrt_27 = None
        mul_95: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_94, arg223_1);  mul_94 = arg223_1 = None
        add_111: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_95, arg224_1);  mul_95 = arg224_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_304: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_111, [2048, 1024]);  add_111 = None
        permute_152: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        
        # No stacktrace found for following nodes
        mm_default_32: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_304, permute_152);  view_304 = permute_152 = None
        add_tensor_32: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_32, arg226_1);  mm_default_32 = arg226_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_305: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_32, [4, 512, 4096]);  add_tensor_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_96: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        mul_97: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476);  view_305 = None
        erf_13: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_112: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_98: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_96, add_112);  mul_96 = add_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_306: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_98, [2048, 4096]);  mul_98 = None
        permute_153: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        
        # No stacktrace found for following nodes
        mm_default_31: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_306, permute_153);  view_306 = permute_153 = None
        add_tensor_31: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_31, arg228_1);  mm_default_31 = arg228_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_307: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_31, [4, 512, 1024]);  add_tensor_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_113: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_109, view_307);  add_109 = view_307 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_28 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
        getitem_56: "f32[4, 512, 1]" = var_mean_28[0]
        getitem_57: "f32[4, 512, 1]" = var_mean_28[1];  var_mean_28 = None
        sub_43: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_113, getitem_57);  getitem_57 = None
        add_114: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_28: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_99: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_28);  sub_43 = rsqrt_28 = None
        mul_100: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_99, arg229_1);  mul_99 = arg229_1 = None
        add_115: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_100, arg230_1);  mul_100 = arg230_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_308: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_115, [2048, 1024])
        permute_154: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_84: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg232_1, view_308, permute_154);  arg232_1 = view_308 = permute_154 = None
        view_309: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_84, [4, 512, 1024]);  addmm_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_316: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_309, [4, 512, 16, 64]);  view_309 = None
        
        # No stacktrace found for following nodes
        permute_default_27: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_310: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_115, [2048, 1024])
        permute_155: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_85: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg234_1, view_310, permute_155);  arg234_1 = view_310 = permute_155 = None
        view_311: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_85, [4, 512, 1024]);  addmm_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_312: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_311, [4, 512, 16, 64]);  view_311 = None
        
        # No stacktrace found for following nodes
        permute_default_28: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_313: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_115, [2048, 1024]);  add_115 = None
        permute_157: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_86: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg236_1, view_313, permute_157);  arg236_1 = view_313 = permute_157 = None
        view_314: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_86, [4, 512, 1024]);  addmm_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_315: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_314, [4, 512, 16, 64]);  view_314 = None
        
        # No stacktrace found for following nodes
        permute_default_29: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_27, permute_default_28, permute_default_29, None, False, scale = 0.125);  permute_default_27 = permute_default_28 = permute_default_29 = None
        getitem_109: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_161: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_109, [0, 2, 1, 3]);  getitem_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_323: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_161, [4, 512, 1024]);  permute_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_324: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_323, [2048, 1024]);  view_323 = None
        permute_162: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        
        # No stacktrace found for following nodes
        mm_default_30: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_324, permute_162);  view_324 = permute_162 = None
        add_tensor_30: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_30, arg238_1);  mm_default_30 = arg238_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_325: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_30, [4, 512, 1024]);  add_tensor_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_117: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_113, view_325);  add_113 = view_325 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_29 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_58: "f32[4, 512, 1]" = var_mean_29[0]
        getitem_59: "f32[4, 512, 1]" = var_mean_29[1];  var_mean_29 = None
        sub_45: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_117, getitem_59);  getitem_59 = None
        add_118: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
        rsqrt_29: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_101: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_29);  sub_45 = rsqrt_29 = None
        mul_102: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_101, arg239_1);  mul_101 = arg239_1 = None
        add_119: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_102, arg240_1);  mul_102 = arg240_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_326: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_119, [2048, 1024]);  add_119 = None
        permute_163: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        
        # No stacktrace found for following nodes
        mm_default_29: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_326, permute_163);  view_326 = permute_163 = None
        add_tensor_29: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_29, arg242_1);  mm_default_29 = arg242_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_327: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_29, [4, 512, 4096]);  add_tensor_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_103: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
        mul_104: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476);  view_327 = None
        erf_14: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
        add_120: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_105: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_103, add_120);  mul_103 = add_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_328: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_105, [2048, 4096]);  mul_105 = None
        permute_164: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        
        # No stacktrace found for following nodes
        mm_default_28: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_328, permute_164);  view_328 = permute_164 = None
        add_tensor_28: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_28, arg244_1);  mm_default_28 = arg244_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_329: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_28, [4, 512, 1024]);  add_tensor_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_121: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_117, view_329);  add_117 = view_329 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_30 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
        getitem_60: "f32[4, 512, 1]" = var_mean_30[0]
        getitem_61: "f32[4, 512, 1]" = var_mean_30[1];  var_mean_30 = None
        sub_46: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_121, getitem_61);  getitem_61 = None
        add_122: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
        rsqrt_30: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_106: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = rsqrt_30 = None
        mul_107: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_106, arg245_1);  mul_106 = arg245_1 = None
        add_123: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_107, arg246_1);  mul_107 = arg246_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_330: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_123, [2048, 1024])
        permute_165: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_90: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg248_1, view_330, permute_165);  arg248_1 = view_330 = permute_165 = None
        view_331: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_90, [4, 512, 1024]);  addmm_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_338: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_331, [4, 512, 16, 64]);  view_331 = None
        
        # No stacktrace found for following nodes
        permute_default_24: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_332: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_123, [2048, 1024])
        permute_166: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_91: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg250_1, view_332, permute_166);  arg250_1 = view_332 = permute_166 = None
        view_333: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_91, [4, 512, 1024]);  addmm_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_334: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_333, [4, 512, 16, 64]);  view_333 = None
        
        # No stacktrace found for following nodes
        permute_default_25: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_335: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_123, [2048, 1024]);  add_123 = None
        permute_168: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_92: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg252_1, view_335, permute_168);  arg252_1 = view_335 = permute_168 = None
        view_336: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_92, [4, 512, 1024]);  addmm_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_337: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_336, [4, 512, 16, 64]);  view_336 = None
        
        # No stacktrace found for following nodes
        permute_default_26: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
        _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_24, permute_default_25, permute_default_26, None, False, scale = 0.125);  permute_default_24 = permute_default_25 = permute_default_26 = None
        getitem_108: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_172: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3]);  getitem_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_345: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_172, [4, 512, 1024]);  permute_172 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_346: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_345, [2048, 1024]);  view_345 = None
        permute_173: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        
        # No stacktrace found for following nodes
        mm_default_27: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_346, permute_173);  view_346 = permute_173 = None
        add_tensor_27: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_27, arg254_1);  mm_default_27 = arg254_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_347: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_27, [4, 512, 1024]);  add_tensor_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_125: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_121, view_347);  add_121 = view_347 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_31 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_62: "f32[4, 512, 1]" = var_mean_31[0]
        getitem_63: "f32[4, 512, 1]" = var_mean_31[1];  var_mean_31 = None
        sub_48: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_125, getitem_63);  getitem_63 = None
        add_126: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_31: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_108: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_31);  sub_48 = rsqrt_31 = None
        mul_109: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_108, arg255_1);  mul_108 = arg255_1 = None
        add_127: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_109, arg256_1);  mul_109 = arg256_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_348: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_127, [2048, 1024]);  add_127 = None
        permute_174: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        
        # No stacktrace found for following nodes
        mm_default_26: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_348, permute_174);  view_348 = permute_174 = None
        add_tensor_26: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_26, arg258_1);  mm_default_26 = arg258_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_349: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_26, [4, 512, 4096]);  add_tensor_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_110: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, 0.5)
        mul_111: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476);  view_349 = None
        erf_15: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_128: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_112: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_110, add_128);  mul_110 = add_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_350: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_112, [2048, 4096]);  mul_112 = None
        permute_175: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        
        # No stacktrace found for following nodes
        mm_default_25: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_350, permute_175);  view_350 = permute_175 = None
        add_tensor_25: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_25, arg260_1);  mm_default_25 = arg260_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_351: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_25, [4, 512, 1024]);  add_tensor_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_129: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_125, view_351);  add_125 = view_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_32 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_64: "f32[4, 512, 1]" = var_mean_32[0]
        getitem_65: "f32[4, 512, 1]" = var_mean_32[1];  var_mean_32 = None
        sub_49: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_129, getitem_65);  getitem_65 = None
        add_130: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_32: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        mul_113: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = rsqrt_32 = None
        mul_114: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_113, arg261_1);  mul_113 = arg261_1 = None
        add_131: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_114, arg262_1);  mul_114 = arg262_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_352: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_131, [2048, 1024])
        permute_176: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_96: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg264_1, view_352, permute_176);  arg264_1 = view_352 = permute_176 = None
        view_353: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_96, [4, 512, 1024]);  addmm_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_360: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_353, [4, 512, 16, 64]);  view_353 = None
        
        # No stacktrace found for following nodes
        permute_default_21: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_354: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_131, [2048, 1024])
        permute_177: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_97: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg266_1, view_354, permute_177);  arg266_1 = view_354 = permute_177 = None
        view_355: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_97, [4, 512, 1024]);  addmm_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_356: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_355, [4, 512, 16, 64]);  view_355 = None
        
        # No stacktrace found for following nodes
        permute_default_22: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_357: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_131, [2048, 1024]);  add_131 = None
        permute_179: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_98: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg268_1, view_357, permute_179);  arg268_1 = view_357 = permute_179 = None
        view_358: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_98, [4, 512, 1024]);  addmm_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_359: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_358, [4, 512, 16, 64]);  view_358 = None
        
        # No stacktrace found for following nodes
        permute_default_23: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_21, permute_default_22, permute_default_23, None, False, scale = 0.125);  permute_default_21 = permute_default_22 = permute_default_23 = None
        getitem_107: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_183: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_107, [0, 2, 1, 3]);  getitem_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_367: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_183, [4, 512, 1024]);  permute_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_368: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_367, [2048, 1024]);  view_367 = None
        permute_184: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        
        # No stacktrace found for following nodes
        mm_default_24: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_368, permute_184);  view_368 = permute_184 = None
        add_tensor_24: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_24, arg270_1);  mm_default_24 = arg270_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_369: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_24, [4, 512, 1024]);  add_tensor_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_133: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_129, view_369);  add_129 = view_369 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_33 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_66: "f32[4, 512, 1]" = var_mean_33[0]
        getitem_67: "f32[4, 512, 1]" = var_mean_33[1];  var_mean_33 = None
        sub_51: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_133, getitem_67);  getitem_67 = None
        add_134: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
        rsqrt_33: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        mul_115: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_33);  sub_51 = rsqrt_33 = None
        mul_116: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_115, arg271_1);  mul_115 = arg271_1 = None
        add_135: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_116, arg272_1);  mul_116 = arg272_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_370: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_135, [2048, 1024]);  add_135 = None
        permute_185: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_370, permute_185);  view_370 = permute_185 = None
        add_tensor_23: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_23, arg274_1);  mm_default_23 = arg274_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_371: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_23, [4, 512, 4096]);  add_tensor_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_117: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, 0.5)
        mul_118: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476);  view_371 = None
        erf_16: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_136: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_119: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_117, add_136);  mul_117 = add_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_372: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_119, [2048, 4096]);  mul_119 = None
        permute_186: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_372, permute_186);  view_372 = permute_186 = None
        add_tensor_22: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_22, arg276_1);  mm_default_22 = arg276_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_373: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_22, [4, 512, 1024]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_137: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_133, view_373);  add_133 = view_373 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_34 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_68: "f32[4, 512, 1]" = var_mean_34[0]
        getitem_69: "f32[4, 512, 1]" = var_mean_34[1];  var_mean_34 = None
        sub_52: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_137, getitem_69);  getitem_69 = None
        add_138: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
        rsqrt_34: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_120: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = rsqrt_34 = None
        mul_121: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_120, arg277_1);  mul_120 = arg277_1 = None
        add_139: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_121, arg278_1);  mul_121 = arg278_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_374: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_139, [2048, 1024])
        permute_187: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_102: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg280_1, view_374, permute_187);  arg280_1 = view_374 = permute_187 = None
        view_375: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_102, [4, 512, 1024]);  addmm_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_382: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_375, [4, 512, 16, 64]);  view_375 = None
        
        # No stacktrace found for following nodes
        permute_default_18: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_376: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_139, [2048, 1024])
        permute_188: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_103: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg282_1, view_376, permute_188);  arg282_1 = view_376 = permute_188 = None
        view_377: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_103, [4, 512, 1024]);  addmm_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_378: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_377, [4, 512, 16, 64]);  view_377 = None
        
        # No stacktrace found for following nodes
        permute_default_19: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_379: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_139, [2048, 1024]);  add_139 = None
        permute_190: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_104: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg284_1, view_379, permute_190);  arg284_1 = view_379 = permute_190 = None
        view_380: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_104, [4, 512, 1024]);  addmm_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_381: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_380, [4, 512, 16, 64]);  view_380 = None
        
        # No stacktrace found for following nodes
        permute_default_20: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_381, [0, 2, 1, 3]);  view_381 = None
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_18, permute_default_19, permute_default_20, None, False, scale = 0.125);  permute_default_18 = permute_default_19 = permute_default_20 = None
        getitem_106: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_194: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_106, [0, 2, 1, 3]);  getitem_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_389: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_194, [4, 512, 1024]);  permute_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_390: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_389, [2048, 1024]);  view_389 = None
        permute_195: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_390, permute_195);  view_390 = permute_195 = None
        add_tensor_21: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_21, arg286_1);  mm_default_21 = arg286_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_391: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_21, [4, 512, 1024]);  add_tensor_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_141: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_137, view_391);  add_137 = view_391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_35 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
        getitem_70: "f32[4, 512, 1]" = var_mean_35[0]
        getitem_71: "f32[4, 512, 1]" = var_mean_35[1];  var_mean_35 = None
        sub_54: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_141, getitem_71);  getitem_71 = None
        add_142: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_35: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        mul_122: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_35);  sub_54 = rsqrt_35 = None
        mul_123: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_122, arg287_1);  mul_122 = arg287_1 = None
        add_143: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_123, arg288_1);  mul_123 = arg288_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_392: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_143, [2048, 1024]);  add_143 = None
        permute_196: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_392, permute_196);  view_392 = permute_196 = None
        add_tensor_20: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_20, arg290_1);  mm_default_20 = arg290_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_393: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_20, [4, 512, 4096]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_124: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.5)
        mul_125: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476);  view_393 = None
        erf_17: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_144: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_126: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_124, add_144);  mul_124 = add_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_394: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_126, [2048, 4096]);  mul_126 = None
        permute_197: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_394, permute_197);  view_394 = permute_197 = None
        add_tensor_19: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_19, arg292_1);  mm_default_19 = arg292_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_395: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_19, [4, 512, 1024]);  add_tensor_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_145: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_141, view_395);  add_141 = view_395 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_36 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
        getitem_72: "f32[4, 512, 1]" = var_mean_36[0]
        getitem_73: "f32[4, 512, 1]" = var_mean_36[1];  var_mean_36 = None
        sub_55: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_145, getitem_73);  getitem_73 = None
        add_146: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_36: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        mul_127: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_36);  sub_55 = rsqrt_36 = None
        mul_128: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_127, arg293_1);  mul_127 = arg293_1 = None
        add_147: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_128, arg294_1);  mul_128 = arg294_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_396: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_147, [2048, 1024])
        permute_198: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_108: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg296_1, view_396, permute_198);  arg296_1 = view_396 = permute_198 = None
        view_397: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_108, [4, 512, 1024]);  addmm_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_404: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_397, [4, 512, 16, 64]);  view_397 = None
        
        # No stacktrace found for following nodes
        permute_default_15: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_398: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_147, [2048, 1024])
        permute_199: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_109: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg298_1, view_398, permute_199);  arg298_1 = view_398 = permute_199 = None
        view_399: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_109, [4, 512, 1024]);  addmm_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_400: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_399, [4, 512, 16, 64]);  view_399 = None
        
        # No stacktrace found for following nodes
        permute_default_16: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_401: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_147, [2048, 1024]);  add_147 = None
        permute_201: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_110: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg300_1, view_401, permute_201);  arg300_1 = view_401 = permute_201 = None
        view_402: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_110, [4, 512, 1024]);  addmm_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_403: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_402, [4, 512, 16, 64]);  view_402 = None
        
        # No stacktrace found for following nodes
        permute_default_17: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_15, permute_default_16, permute_default_17, None, False, scale = 0.125);  permute_default_15 = permute_default_16 = permute_default_17 = None
        getitem_105: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_205: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_105, [0, 2, 1, 3]);  getitem_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_411: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_205, [4, 512, 1024]);  permute_205 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_412: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_411, [2048, 1024]);  view_411 = None
        permute_206: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_412, permute_206);  view_412 = permute_206 = None
        add_tensor_18: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_18, arg302_1);  mm_default_18 = arg302_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_413: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_18, [4, 512, 1024]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_149: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_145, view_413);  add_145 = view_413 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_37 = torch.ops.aten.var_mean.correction(add_149, [2], correction = 0, keepdim = True)
        getitem_74: "f32[4, 512, 1]" = var_mean_37[0]
        getitem_75: "f32[4, 512, 1]" = var_mean_37[1];  var_mean_37 = None
        sub_57: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_149, getitem_75);  getitem_75 = None
        add_150: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
        rsqrt_37: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        mul_129: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_37);  sub_57 = rsqrt_37 = None
        mul_130: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_129, arg303_1);  mul_129 = arg303_1 = None
        add_151: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_130, arg304_1);  mul_130 = arg304_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_414: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_151, [2048, 1024]);  add_151 = None
        permute_207: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_414, permute_207);  view_414 = permute_207 = None
        add_tensor_17: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_17, arg306_1);  mm_default_17 = arg306_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_415: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_17, [4, 512, 4096]);  add_tensor_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_131: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
        mul_132: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476);  view_415 = None
        erf_18: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
        add_152: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_133: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_131, add_152);  mul_131 = add_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_416: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_133, [2048, 4096]);  mul_133 = None
        permute_208: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_416, permute_208);  view_416 = permute_208 = None
        add_tensor_16: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_16, arg308_1);  mm_default_16 = arg308_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_417: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_16, [4, 512, 1024]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_153: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_149, view_417);  add_149 = view_417 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_38 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
        getitem_76: "f32[4, 512, 1]" = var_mean_38[0]
        getitem_77: "f32[4, 512, 1]" = var_mean_38[1];  var_mean_38 = None
        sub_58: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_153, getitem_77);  getitem_77 = None
        add_154: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
        rsqrt_38: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        mul_134: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_38);  sub_58 = rsqrt_38 = None
        mul_135: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_134, arg309_1);  mul_134 = arg309_1 = None
        add_155: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_135, arg310_1);  mul_135 = arg310_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_418: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_155, [2048, 1024])
        permute_209: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_114: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg312_1, view_418, permute_209);  arg312_1 = view_418 = permute_209 = None
        view_419: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_114, [4, 512, 1024]);  addmm_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_426: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_419, [4, 512, 16, 64]);  view_419 = None
        
        # No stacktrace found for following nodes
        permute_default_12: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_420: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_155, [2048, 1024])
        permute_210: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_115: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg314_1, view_420, permute_210);  arg314_1 = view_420 = permute_210 = None
        view_421: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_115, [4, 512, 1024]);  addmm_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_422: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_421, [4, 512, 16, 64]);  view_421 = None
        
        # No stacktrace found for following nodes
        permute_default_13: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_423: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_155, [2048, 1024]);  add_155 = None
        permute_212: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_116: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg316_1, view_423, permute_212);  arg316_1 = view_423 = permute_212 = None
        view_424: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_116, [4, 512, 1024]);  addmm_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_425: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_424, [4, 512, 16, 64]);  view_424 = None
        
        # No stacktrace found for following nodes
        permute_default_14: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_12, permute_default_13, permute_default_14, None, False, scale = 0.125);  permute_default_12 = permute_default_13 = permute_default_14 = None
        getitem_104: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_216: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_433: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_216, [4, 512, 1024]);  permute_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_434: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_433, [2048, 1024]);  view_433 = None
        permute_217: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_434, permute_217);  view_434 = permute_217 = None
        add_tensor_15: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_15, arg318_1);  mm_default_15 = arg318_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_435: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_15, [4, 512, 1024]);  add_tensor_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_157: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_153, view_435);  add_153 = view_435 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_39 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_78: "f32[4, 512, 1]" = var_mean_39[0]
        getitem_79: "f32[4, 512, 1]" = var_mean_39[1];  var_mean_39 = None
        sub_60: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_157, getitem_79);  getitem_79 = None
        add_158: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_39: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        mul_136: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_39);  sub_60 = rsqrt_39 = None
        mul_137: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_136, arg319_1);  mul_136 = arg319_1 = None
        add_159: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_137, arg320_1);  mul_137 = arg320_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_436: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_159, [2048, 1024]);  add_159 = None
        permute_218: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_436, permute_218);  view_436 = permute_218 = None
        add_tensor_14: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_14, arg322_1);  mm_default_14 = arg322_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_437: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_14, [4, 512, 4096]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_138: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, 0.5)
        mul_139: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476);  view_437 = None
        erf_19: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
        add_160: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_140: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_138, add_160);  mul_138 = add_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_438: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_140, [2048, 4096]);  mul_140 = None
        permute_219: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_438, permute_219);  view_438 = permute_219 = None
        add_tensor_13: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_13, arg324_1);  mm_default_13 = arg324_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_439: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_13, [4, 512, 1024]);  add_tensor_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_161: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_157, view_439);  add_157 = view_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_40 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_80: "f32[4, 512, 1]" = var_mean_40[0]
        getitem_81: "f32[4, 512, 1]" = var_mean_40[1];  var_mean_40 = None
        sub_61: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_161, getitem_81);  getitem_81 = None
        add_162: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_40: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        mul_141: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_40);  sub_61 = rsqrt_40 = None
        mul_142: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_141, arg325_1);  mul_141 = arg325_1 = None
        add_163: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_142, arg326_1);  mul_142 = arg326_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_440: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_163, [2048, 1024])
        permute_220: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_120: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg328_1, view_440, permute_220);  arg328_1 = view_440 = permute_220 = None
        view_441: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_120, [4, 512, 1024]);  addmm_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_448: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_441, [4, 512, 16, 64]);  view_441 = None
        
        # No stacktrace found for following nodes
        permute_default_9: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_442: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_163, [2048, 1024])
        permute_221: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        addmm_121: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg330_1, view_442, permute_221);  arg330_1 = view_442 = permute_221 = None
        view_443: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_121, [4, 512, 1024]);  addmm_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_444: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_443, [4, 512, 16, 64]);  view_443 = None
        
        # No stacktrace found for following nodes
        permute_default_10: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_445: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_163, [2048, 1024]);  add_163 = None
        permute_223: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_122: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg332_1, view_445, permute_223);  arg332_1 = view_445 = permute_223 = None
        view_446: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_122, [4, 512, 1024]);  addmm_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_447: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_446, [4, 512, 16, 64]);  view_446 = None
        
        # No stacktrace found for following nodes
        permute_default_11: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_9, permute_default_10, permute_default_11, None, False, scale = 0.125);  permute_default_9 = permute_default_10 = permute_default_11 = None
        getitem_103: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_227: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_103, [0, 2, 1, 3]);  getitem_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_455: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_227, [4, 512, 1024]);  permute_227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_456: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_455, [2048, 1024]);  view_455 = None
        permute_228: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_456, permute_228);  view_456 = permute_228 = None
        add_tensor_12: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_12, arg334_1);  mm_default_12 = arg334_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_457: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_12, [4, 512, 1024]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_165: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_161, view_457);  add_161 = view_457 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_41 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
        getitem_82: "f32[4, 512, 1]" = var_mean_41[0]
        getitem_83: "f32[4, 512, 1]" = var_mean_41[1];  var_mean_41 = None
        sub_63: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_165, getitem_83);  getitem_83 = None
        add_166: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
        rsqrt_41: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        mul_143: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_41);  sub_63 = rsqrt_41 = None
        mul_144: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_143, arg335_1);  mul_143 = arg335_1 = None
        add_167: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_144, arg336_1);  mul_144 = arg336_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_458: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_167, [2048, 1024]);  add_167 = None
        permute_229: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_458, permute_229);  view_458 = permute_229 = None
        add_tensor_11: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_11, arg338_1);  mm_default_11 = arg338_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_459: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_11, [4, 512, 4096]);  add_tensor_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_145: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, 0.5)
        mul_146: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476);  view_459 = None
        erf_20: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
        add_168: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_147: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_145, add_168);  mul_145 = add_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_460: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_147, [2048, 4096]);  mul_147 = None
        permute_230: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_460, permute_230);  view_460 = permute_230 = None
        add_tensor_10: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_10, arg340_1);  mm_default_10 = arg340_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_461: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_10, [4, 512, 1024]);  add_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_169: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_165, view_461);  add_165 = view_461 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_42 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
        getitem_84: "f32[4, 512, 1]" = var_mean_42[0]
        getitem_85: "f32[4, 512, 1]" = var_mean_42[1];  var_mean_42 = None
        sub_64: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_169, getitem_85);  getitem_85 = None
        add_170: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-12);  getitem_84 = None
        rsqrt_42: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        mul_148: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_42);  sub_64 = rsqrt_42 = None
        mul_149: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_148, arg341_1);  mul_148 = arg341_1 = None
        add_171: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_149, arg342_1);  mul_149 = arg342_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_462: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_171, [2048, 1024])
        permute_231: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_126: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg344_1, view_462, permute_231);  arg344_1 = view_462 = permute_231 = None
        view_463: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_126, [4, 512, 1024]);  addmm_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_470: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_463, [4, 512, 16, 64]);  view_463 = None
        
        # No stacktrace found for following nodes
        permute_default_6: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_464: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_171, [2048, 1024])
        permute_232: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_127: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg346_1, view_464, permute_232);  arg346_1 = view_464 = permute_232 = None
        view_465: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_127, [4, 512, 1024]);  addmm_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_466: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_465, [4, 512, 16, 64]);  view_465 = None
        
        # No stacktrace found for following nodes
        permute_default_7: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_467: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_171, [2048, 1024]);  add_171 = None
        permute_234: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
        addmm_128: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg348_1, view_467, permute_234);  arg348_1 = view_467 = permute_234 = None
        view_468: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_128, [4, 512, 1024]);  addmm_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_469: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_468, [4, 512, 16, 64]);  view_468 = None
        
        # No stacktrace found for following nodes
        permute_default_8: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_6, permute_default_7, permute_default_8, None, False, scale = 0.125);  permute_default_6 = permute_default_7 = permute_default_8 = None
        getitem_102: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_238: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_102, [0, 2, 1, 3]);  getitem_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_477: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_238, [4, 512, 1024]);  permute_238 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_478: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_477, [2048, 1024]);  view_477 = None
        permute_239: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_478, permute_239);  view_478 = permute_239 = None
        add_tensor_9: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_9, arg350_1);  mm_default_9 = arg350_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_479: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_9, [4, 512, 1024]);  add_tensor_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_173: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_169, view_479);  add_169 = view_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_43 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_86: "f32[4, 512, 1]" = var_mean_43[0]
        getitem_87: "f32[4, 512, 1]" = var_mean_43[1];  var_mean_43 = None
        sub_66: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_173, getitem_87);  getitem_87 = None
        add_174: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_43: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_150: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_43);  sub_66 = rsqrt_43 = None
        mul_151: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_150, arg351_1);  mul_150 = arg351_1 = None
        add_175: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_151, arg352_1);  mul_151 = arg352_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_480: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_175, [2048, 1024]);  add_175 = None
        permute_240: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_480, permute_240);  view_480 = permute_240 = None
        add_tensor_8: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_8, arg354_1);  mm_default_8 = arg354_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_481: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_8, [4, 512, 4096]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_152: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, 0.5)
        mul_153: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476);  view_481 = None
        erf_21: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
        add_176: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_154: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_152, add_176);  mul_152 = add_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_482: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_154, [2048, 4096]);  mul_154 = None
        permute_241: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_482, permute_241);  view_482 = permute_241 = None
        add_tensor_7: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_7, arg356_1);  mm_default_7 = arg356_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_483: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_7, [4, 512, 1024]);  add_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_177: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_173, view_483);  add_173 = view_483 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_44 = torch.ops.aten.var_mean.correction(add_177, [2], correction = 0, keepdim = True)
        getitem_88: "f32[4, 512, 1]" = var_mean_44[0]
        getitem_89: "f32[4, 512, 1]" = var_mean_44[1];  var_mean_44 = None
        sub_67: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_177, getitem_89);  getitem_89 = None
        add_178: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_44: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        mul_155: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_44);  sub_67 = rsqrt_44 = None
        mul_156: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_155, arg357_1);  mul_155 = arg357_1 = None
        add_179: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_156, arg358_1);  mul_156 = arg358_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_484: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_179, [2048, 1024])
        permute_242: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_132: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg360_1, view_484, permute_242);  arg360_1 = view_484 = permute_242 = None
        view_485: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_132, [4, 512, 1024]);  addmm_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_492: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_485, [4, 512, 16, 64]);  view_485 = None
        
        # No stacktrace found for following nodes
        permute_default_3: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_486: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_179, [2048, 1024])
        permute_243: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
        addmm_133: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg362_1, view_486, permute_243);  arg362_1 = view_486 = permute_243 = None
        view_487: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_133, [4, 512, 1024]);  addmm_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_488: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_487, [4, 512, 16, 64]);  view_487 = None
        
        # No stacktrace found for following nodes
        permute_default_4: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_489: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_179, [2048, 1024]);  add_179 = None
        permute_245: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_134: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg364_1, view_489, permute_245);  arg364_1 = view_489 = permute_245 = None
        view_490: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_134, [4, 512, 1024]);  addmm_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_491: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_490, [4, 512, 16, 64]);  view_490 = None
        
        # No stacktrace found for following nodes
        permute_default_5: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_3, permute_default_4, permute_default_5, None, False, scale = 0.125);  permute_default_3 = permute_default_4 = permute_default_5 = None
        getitem_101: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_249: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_499: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_249, [4, 512, 1024]);  permute_249 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_500: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_499, [2048, 1024]);  view_499 = None
        permute_250: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg365_1, [1, 0]);  arg365_1 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_500, permute_250);  view_500 = permute_250 = None
        add_tensor_6: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_6, arg366_1);  mm_default_6 = arg366_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_501: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_6, [4, 512, 1024]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_181: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_177, view_501);  add_177 = view_501 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_45 = torch.ops.aten.var_mean.correction(add_181, [2], correction = 0, keepdim = True)
        getitem_90: "f32[4, 512, 1]" = var_mean_45[0]
        getitem_91: "f32[4, 512, 1]" = var_mean_45[1];  var_mean_45 = None
        sub_69: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_181, getitem_91);  getitem_91 = None
        add_182: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-12);  getitem_90 = None
        rsqrt_45: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        mul_157: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_45);  sub_69 = rsqrt_45 = None
        mul_158: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_157, arg367_1);  mul_157 = arg367_1 = None
        add_183: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_158, arg368_1);  mul_158 = arg368_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_502: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_183, [2048, 1024]);  add_183 = None
        permute_251: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_502, permute_251);  view_502 = permute_251 = None
        add_tensor_5: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_5, arg370_1);  mm_default_5 = arg370_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_503: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_5, [4, 512, 4096]);  add_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_159: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
        mul_160: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476);  view_503 = None
        erf_22: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
        add_184: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_161: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_159, add_184);  mul_159 = add_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_504: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_161, [2048, 4096]);  mul_161 = None
        permute_252: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg371_1, [1, 0]);  arg371_1 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_504, permute_252);  view_504 = permute_252 = None
        add_tensor_4: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_4, arg372_1);  mm_default_4 = arg372_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_505: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_4, [4, 512, 1024]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_185: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_181, view_505);  add_181 = view_505 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:373 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_46 = torch.ops.aten.var_mean.correction(add_185, [2], correction = 0, keepdim = True)
        getitem_92: "f32[4, 512, 1]" = var_mean_46[0]
        getitem_93: "f32[4, 512, 1]" = var_mean_46[1];  var_mean_46 = None
        sub_70: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_185, getitem_93);  getitem_93 = None
        add_186: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
        rsqrt_46: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        mul_162: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_46);  sub_70 = rsqrt_46 = None
        mul_163: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_162, arg373_1);  mul_162 = arg373_1 = None
        add_187: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_163, arg374_1);  mul_163 = arg374_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:231 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_506: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_187, [2048, 1024])
        permute_253: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg375_1, [1, 0]);  arg375_1 = None
        addmm_138: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg376_1, view_506, permute_253);  arg376_1 = view_506 = permute_253 = None
        view_507: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_138, [4, 512, 1024]);  addmm_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_514: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_507, [4, 512, 16, 64]);  view_507 = None
        
        # No stacktrace found for following nodes
        permute_default: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:253 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_508: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_187, [2048, 1024])
        permute_254: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
        addmm_139: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg378_1, view_508, permute_254);  arg378_1 = view_508 = permute_254 = None
        view_509: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_139, [4, 512, 1024]);  addmm_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_510: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_509, [4, 512, 16, 64]);  view_509 = None
        
        # No stacktrace found for following nodes
        permute_default_1: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:254 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_511: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_187, [2048, 1024]);  add_187 = None
        permute_256: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg379_1, [1, 0]);  arg379_1 = None
        addmm_140: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg380_1, view_511, permute_256);  arg380_1 = view_511 = permute_256 = None
        view_512: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_140, [4, 512, 1024]);  addmm_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:218 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_513: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_512, [4, 512, 16, 64]);  view_512 = None
        
        # No stacktrace found for following nodes
        permute_default_2: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default, permute_default_1, permute_default_2, None, False, scale = 0.125);  permute_default = permute_default_1 = permute_default_2 = None
        getitem_100: "f32[4, 16, 512, 64]" = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:312 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_260: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_100, [0, 2, 1, 3]);  getitem_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:314 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_521: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_260, [4, 512, 1024]);  permute_260 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_522: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_521, [2048, 1024]);  view_521 = None
        permute_261: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg381_1, [1, 0]);  arg381_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_522, permute_261);  view_522 = permute_261 = None
        add_tensor_3: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_3, arg382_1);  mm_default_3 = arg382_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_523: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, [4, 512, 1024]);  add_tensor_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:333 in forward, code: return residual + hidden_states
        add_189: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_185, view_523);  add_185 = view_523 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:500 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_47 = torch.ops.aten.var_mean.correction(add_189, [2], correction = 0, keepdim = True)
        getitem_94: "f32[4, 512, 1]" = var_mean_47[0]
        getitem_95: "f32[4, 512, 1]" = var_mean_47[1];  var_mean_47 = None
        sub_72: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_189, getitem_95);  getitem_95 = None
        add_190: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_47: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        mul_164: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_47);  sub_72 = rsqrt_47 = None
        mul_165: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_164, arg383_1);  mul_164 = arg383_1 = None
        add_191: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_165, arg384_1);  mul_165 = arg384_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_524: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_191, [2048, 1024]);  add_191 = None
        permute_262: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_524, permute_262);  view_524 = permute_262 = None
        add_tensor_2: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_2, arg386_1);  mm_default_2 = arg386_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_525: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_2, [4, 512, 4096]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_166: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, 0.5)
        mul_167: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476);  view_525 = None
        erf_23: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_192: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_168: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_166, add_192);  mul_166 = add_192 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_526: "f32[2048, 4096]" = torch.ops.aten.reshape.default(mul_168, [2048, 4096]);  mul_168 = None
        permute_263: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_526, permute_263);  view_526 = permute_263 = None
        add_tensor_1: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default_1, arg388_1);  mm_default_1 = arg388_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:412 in forward, code: hidden_states = self.dense(hidden_states)
        view_527: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [4, 512, 1024]);  add_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:414 in forward, code: return input_tensor + hidden_states
        add_193: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_189, view_527);  add_189 = view_527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:582 in forward, code: hidden_states = self.ln(hidden_states)
        var_mean_48 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
        getitem_96: "f32[4, 512, 1]" = var_mean_48[0]
        getitem_97: "f32[4, 512, 1]" = var_mean_48[1];  var_mean_48 = None
        sub_73: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(add_193, getitem_97);  add_193 = getitem_97 = None
        add_194: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_48: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        mul_169: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_48);  sub_73 = rsqrt_48 = None
        mul_170: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_169, arg389_1);  mul_169 = arg389_1 = None
        add_195: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_170, arg390_1);  mul_170 = arg390_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:636 in forward, code: hidden_states = self.dense(hidden_states)
        view_528: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_195, [2048, 1024]);  add_195 = None
        permute_264: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg392_1, [1, 0]);  arg392_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_528, permute_264);  view_528 = permute_264 = None
        add_tensor: "f32[2048, 1024]" = torch.ops.aten.add.Tensor(mm_default, arg393_1);  mm_default = arg393_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:636 in forward, code: hidden_states = self.dense(hidden_states)
        view_529: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor, [4, 512, 1024]);  add_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_171: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        mul_172: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476);  view_529 = None
        erf_24: "f32[4, 512, 1024]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
        add_196: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_173: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_171, add_196);  mul_171 = add_196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:638 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_49 = torch.ops.aten.var_mean.correction(mul_173, [2], correction = 0, keepdim = True)
        getitem_98: "f32[4, 512, 1]" = var_mean_49[0]
        getitem_99: "f32[4, 512, 1]" = var_mean_49[1];  var_mean_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1222 in forward, code: labels = labels[:, 1:].contiguous()
        slice_8: "i64[4, 511]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 1, 9223372036854775807);  arg0_1 = None
        clone_170: "i64[4, 511]" = torch.ops.aten.clone.default(slice_8, memory_format = torch.contiguous_format);  slice_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1224 in forward, code: lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_533: "i64[2044]" = torch.ops.aten.reshape.default(clone_170, [-1]);  clone_170 = None
        ne_1: "b8[2044]" = torch.ops.aten.ne.Scalar(view_533, -100)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:638 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_74: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_173, getitem_99);  mul_173 = getitem_99 = None
        add_197: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
        rsqrt_49: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        mul_174: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_49);  sub_74 = rsqrt_49 = None
        mul_175: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_174, arg394_1);  mul_174 = arg394_1 = None
        add_198: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_175, arg395_1);  mul_175 = arg395_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:662 in forward, code: hidden_states = self.decoder(hidden_states)
        view_530: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_198, [2048, 1024]);  add_198 = None
        permute_265: "f32[1024, 29056]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        addmm_145: "f32[2048, 29056]" = torch.ops.aten.addmm.default(arg396_1, view_530, permute_265);  arg396_1 = view_530 = permute_265 = None
        view_531: "f32[4, 512, 29056]" = torch.ops.aten.reshape.default(addmm_145, [4, 512, 29056]);  addmm_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1221 in forward, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        slice_5: "f32[4, 511, 29056]" = torch.ops.aten.slice.Tensor(view_531, 1, 0, -1)
        clone_169: "f32[4, 511, 29056]" = torch.ops.aten.clone.default(slice_5, memory_format = torch.contiguous_format);  slice_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1224 in forward, code: lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_532: "f32[2044, 29056]" = torch.ops.aten.reshape.default(clone_169, [-1, 29056]);  clone_169 = None
        amax_24: "f32[2044, 1]" = torch.ops.aten.amax.default(view_532, [1], True)
        sub_75: "f32[2044, 29056]" = torch.ops.aten.sub.Tensor(view_532, amax_24);  view_532 = amax_24 = None
        exp_24: "f32[2044, 29056]" = torch.ops.aten.exp.default(sub_75)
        sum_25: "f32[2044, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[2044, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_76: "f32[2044, 29056]" = torch.ops.aten.sub.Tensor(sub_75, log);  sub_75 = log = None
        ne: "b8[2044]" = torch.ops.aten.ne.Scalar(view_533, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[2044]" = torch.ops.aten.where.self(ne, view_533, full_default_2);  ne = full_default_2 = None
        unsqueeze_2: "i64[2044, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[2044, 1]" = torch.ops.aten.gather.default(sub_76, 1, unsqueeze_2);  sub_76 = unsqueeze_2 = None
        squeeze: "f32[2044]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[2044]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[2044]" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[2044]" = torch.ops.aten.ne.Scalar(view_533, -100);  view_533 = None
        sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        div_48: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None
        return (div_48, view_531)
        