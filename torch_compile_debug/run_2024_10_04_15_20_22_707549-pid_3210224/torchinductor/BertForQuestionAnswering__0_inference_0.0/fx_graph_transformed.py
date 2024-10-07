class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "i64[1, 512]", arg2_1: "i64[1, 512]", arg3_1: "f32[30522, 768]", arg4_1: "f32[2, 768]", arg5_1: "f32[512, 768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[768, 768]", arg11_1: "f32[768]", arg12_1: "f32[768, 768]", arg13_1: "f32[768]", arg14_1: "f32[768, 768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[3072, 768]", arg19_1: "f32[3072]", arg20_1: "f32[768, 3072]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768, 768]", arg25_1: "f32[768]", arg26_1: "f32[768, 768]", arg27_1: "f32[768]", arg28_1: "f32[768, 768]", arg29_1: "f32[768]", arg30_1: "f32[768, 768]", arg31_1: "f32[768]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[3072, 768]", arg35_1: "f32[3072]", arg36_1: "f32[768, 3072]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768, 768]", arg41_1: "f32[768]", arg42_1: "f32[768, 768]", arg43_1: "f32[768]", arg44_1: "f32[768, 768]", arg45_1: "f32[768]", arg46_1: "f32[768, 768]", arg47_1: "f32[768]", arg48_1: "f32[768]", arg49_1: "f32[768]", arg50_1: "f32[3072, 768]", arg51_1: "f32[3072]", arg52_1: "f32[768, 3072]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768, 768]", arg59_1: "f32[768]", arg60_1: "f32[768, 768]", arg61_1: "f32[768]", arg62_1: "f32[768, 768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[3072, 768]", arg67_1: "f32[3072]", arg68_1: "f32[768, 3072]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768, 768]", arg73_1: "f32[768]", arg74_1: "f32[768, 768]", arg75_1: "f32[768]", arg76_1: "f32[768, 768]", arg77_1: "f32[768]", arg78_1: "f32[768, 768]", arg79_1: "f32[768]", arg80_1: "f32[768]", arg81_1: "f32[768]", arg82_1: "f32[3072, 768]", arg83_1: "f32[3072]", arg84_1: "f32[768, 3072]", arg85_1: "f32[768]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[768, 768]", arg89_1: "f32[768]", arg90_1: "f32[768, 768]", arg91_1: "f32[768]", arg92_1: "f32[768, 768]", arg93_1: "f32[768]", arg94_1: "f32[768, 768]", arg95_1: "f32[768]", arg96_1: "f32[768]", arg97_1: "f32[768]", arg98_1: "f32[3072, 768]", arg99_1: "f32[3072]", arg100_1: "f32[768, 3072]", arg101_1: "f32[768]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[768, 768]", arg105_1: "f32[768]", arg106_1: "f32[768, 768]", arg107_1: "f32[768]", arg108_1: "f32[768, 768]", arg109_1: "f32[768]", arg110_1: "f32[768, 768]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[3072, 768]", arg115_1: "f32[3072]", arg116_1: "f32[768, 3072]", arg117_1: "f32[768]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768, 768]", arg121_1: "f32[768]", arg122_1: "f32[768, 768]", arg123_1: "f32[768]", arg124_1: "f32[768, 768]", arg125_1: "f32[768]", arg126_1: "f32[768, 768]", arg127_1: "f32[768]", arg128_1: "f32[768]", arg129_1: "f32[768]", arg130_1: "f32[3072, 768]", arg131_1: "f32[3072]", arg132_1: "f32[768, 3072]", arg133_1: "f32[768]", arg134_1: "f32[768]", arg135_1: "f32[768]", arg136_1: "f32[768, 768]", arg137_1: "f32[768]", arg138_1: "f32[768, 768]", arg139_1: "f32[768]", arg140_1: "f32[768, 768]", arg141_1: "f32[768]", arg142_1: "f32[768, 768]", arg143_1: "f32[768]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg146_1: "f32[3072, 768]", arg147_1: "f32[3072]", arg148_1: "f32[768, 3072]", arg149_1: "f32[768]", arg150_1: "f32[768]", arg151_1: "f32[768]", arg152_1: "f32[768, 768]", arg153_1: "f32[768]", arg154_1: "f32[768, 768]", arg155_1: "f32[768]", arg156_1: "f32[768, 768]", arg157_1: "f32[768]", arg158_1: "f32[768, 768]", arg159_1: "f32[768]", arg160_1: "f32[768]", arg161_1: "f32[768]", arg162_1: "f32[3072, 768]", arg163_1: "f32[3072]", arg164_1: "f32[768, 3072]", arg165_1: "f32[768]", arg166_1: "f32[768]", arg167_1: "f32[768]", arg168_1: "f32[768, 768]", arg169_1: "f32[768]", arg170_1: "f32[768, 768]", arg171_1: "f32[768]", arg172_1: "f32[768, 768]", arg173_1: "f32[768]", arg174_1: "f32[768, 768]", arg175_1: "f32[768]", arg176_1: "f32[768]", arg177_1: "f32[768]", arg178_1: "f32[3072, 768]", arg179_1: "f32[3072]", arg180_1: "f32[768, 3072]", arg181_1: "f32[768]", arg182_1: "f32[768]", arg183_1: "f32[768]", arg184_1: "f32[768, 768]", arg185_1: "f32[768]", arg186_1: "f32[768, 768]", arg187_1: "f32[768]", arg188_1: "f32[768, 768]", arg189_1: "f32[768]", arg190_1: "f32[768, 768]", arg191_1: "f32[768]", arg192_1: "f32[768]", arg193_1: "f32[768]", arg194_1: "f32[3072, 768]", arg195_1: "f32[3072]", arg196_1: "f32[768, 3072]", arg197_1: "f32[768]", arg198_1: "f32[768]", arg199_1: "f32[768]", arg200_1: "f32[2, 768]", arg201_1: "f32[2]", arg202_1: "i64[16]", arg203_1: "i64[16]"):
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:211 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[16, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg3_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1073 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand: "i64[16, 512]" = torch.ops.aten.expand.default(arg1_1, [16, 512]);  arg1_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:212 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[16, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, expand);  arg4_1 = expand = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:214 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:216 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, arg2_1);  arg5_1 = arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:217 in forward, code: embeddings += position_embeddings
        add_1: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:218 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean[0]
        getitem_1: "f32[16, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        add_2: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_3: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:395 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_3, [8192, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg9_1, view, permute);  arg9_1 = view = permute = None
        view_1: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm, [16, 512, 768]);  addmm = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_2: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_1, [16, 512, 12, 64]);  view_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_1: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:408 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_3: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_3, [8192, 768])
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_1: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg11_1, view_3, permute_2);  arg11_1 = view_3 = permute_2 = None
        view_4: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [16, 512, 768]);  addmm_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_5: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_4, [16, 512, 12, 64]);  view_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_3: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:409 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_6: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_3, [8192, 768])
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_2: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg13_1, view_6, permute_4);  arg13_1 = view_6 = permute_4 = None
        view_7: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [16, 512, 768]);  addmm_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_8: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_7, [16, 512, 12, 64]);  view_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_5: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1087 in forward, code: attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
        full: "f32[16, 512]" = torch.ops.aten.full.default([16, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:184 in _expand_mask, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        unsqueeze: "f32[16, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1: "f32[16, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand_1: "f32[16, 1, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_1, [16, 1, 512, 512]);  unsqueeze_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:186 in _expand_mask, code: inverted_mask = 1.0 - expanded_mask
        sub_1: "f32[16, 1, 512, 512]" = torch.ops.aten.sub.Tensor(1.0, expand_1);  expand_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:188 in _expand_mask, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        convert_element_type: "b8[16, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where: "f32[16, 1, 512, 512]" = torch.ops.aten.where.self(convert_element_type, scalar_tensor, sub_1);  convert_element_type = scalar_tensor = sub_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:440 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_2: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_3, permute_5, expand_2, False);  permute_1 = permute_3 = permute_5 = expand_2 = None
        getitem_2: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:449 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_6: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:450 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_9: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(permute_6, [16, 512, 768]);  permute_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_10: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_9, [8192, 768]);  view_9 = None
        permute_7: "f32[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        
        # No stacktrace found for following nodes
        mm_default_36: "f32[8192, 768]" = torch.ops.aten.mm.default(view_10, permute_7);  view_10 = permute_7 = None
        add_tensor_36: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_36, arg15_1);  mm_default_36 = arg15_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_11: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_36, [16, 512, 768]);  add_tensor_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:468 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_4: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_11, add_3);  view_11 = add_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_6: "f32[16, 512, 1]" = var_mean_1[0]
        getitem_7: "f32[16, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
        add_5: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_2: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_3: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg16_1);  mul_2 = arg16_1 = None
        add_6: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, arg17_1);  mul_3 = arg17_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_12: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_6, [8192, 768])
        permute_8: "f32[768, 3072]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        
        # No stacktrace found for following nodes
        mm_default_35: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_12, permute_8);  view_12 = permute_8 = None
        add_tensor_35: "f32[8192, 3072]" = torch.ops.aten.add.Tensor(mm_default_35, arg19_1);  mm_default_35 = arg19_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_13: "f32[16, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_35, [16, 512, 3072]);  add_tensor_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_4: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_5: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_7: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_14: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_6, [8192, 3072]);  mul_6 = None
        permute_9: "f32[3072, 768]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        
        # No stacktrace found for following nodes
        mm_default_34: "f32[8192, 768]" = torch.ops.aten.mm.default(view_14, permute_9);  view_14 = permute_9 = None
        add_tensor_34: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_34, arg21_1);  mm_default_34 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_15: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_34, [16, 512, 768]);  add_tensor_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:554 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_8: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_15, add_6);  view_15 = add_6 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_8: "f32[16, 512, 1]" = var_mean_2[0]
        getitem_9: "f32[16, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_9);  add_8 = getitem_9 = None
        add_9: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_2: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_7: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_8: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_7, arg22_1);  mul_7 = arg22_1 = None
        add_10: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_8, arg23_1);  mul_8 = arg23_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:395 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_16: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_10, [8192, 768])
        permute_10: "f32[768, 768]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_6: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg25_1, view_16, permute_10);  arg25_1 = view_16 = permute_10 = None
        view_17: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [16, 512, 768]);  addmm_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_18: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_17, [16, 512, 12, 64]);  view_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_11: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:408 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_19: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_10, [8192, 768])
        permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_7: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg27_1, view_19, permute_12);  arg27_1 = view_19 = permute_12 = None
        view_20: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [16, 512, 768]);  addmm_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_21: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_20, [16, 512, 12, 64]);  view_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_13: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:409 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_22: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_10, [8192, 768])
        permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_8: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg29_1, view_22, permute_14);  arg29_1 = view_22 = permute_14 = None
        view_23: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_8, [16, 512, 768]);  addmm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_24: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_23, [16, 512, 12, 64]);  view_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_15: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:440 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_3: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_11, permute_13, permute_15, expand_3, False);  permute_11 = permute_13 = permute_15 = expand_3 = None
        getitem_10: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:449 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_16: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:450 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_25: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(permute_16, [16, 512, 768]);  permute_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_26: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_25, [8192, 768]);  view_25 = None
        permute_17: "f32[768, 768]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        
        # No stacktrace found for following nodes
        mm_default_33: "f32[8192, 768]" = torch.ops.aten.mm.default(view_26, permute_17);  view_26 = permute_17 = None
        add_tensor_33: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_33, arg31_1);  mm_default_33 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_27: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_33, [16, 512, 768]);  add_tensor_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:468 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_11: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_27, add_10);  view_27 = add_10 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_14: "f32[16, 512, 1]" = var_mean_3[0]
        getitem_15: "f32[16, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_4: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_15);  add_11 = getitem_15 = None
        add_12: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_9: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_10: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg32_1);  mul_9 = arg32_1 = None
        add_13: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, arg33_1);  mul_10 = arg33_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_28: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_13, [8192, 768])
        permute_18: "f32[768, 3072]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        
        # No stacktrace found for following nodes
        mm_default_32: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_28, permute_18);  view_28 = permute_18 = None
        add_tensor_32: "f32[8192, 3072]" = torch.ops.aten.add.Tensor(mm_default_32, arg35_1);  mm_default_32 = arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_29: "f32[16, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_32, [16, 512, 3072]);  add_tensor_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_11: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.5)
        mul_12: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.7071067811865476);  view_29 = None
        erf_1: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_14: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_14);  mul_11 = add_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_30: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_13, [8192, 3072]);  mul_13 = None
        permute_19: "f32[3072, 768]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        
        # No stacktrace found for following nodes
        mm_default_31: "f32[8192, 768]" = torch.ops.aten.mm.default(view_30, permute_19);  view_30 = permute_19 = None
        add_tensor_31: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_31, arg37_1);  mm_default_31 = arg37_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_31: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_31, [16, 512, 768]);  add_tensor_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:554 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_31, add_13);  view_31 = add_13 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_16: "f32[16, 512, 1]" = var_mean_4[0]
        getitem_17: "f32[16, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_5: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_17);  add_15 = getitem_17 = None
        add_16: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_4: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_14: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = rsqrt_4 = None
        mul_15: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg38_1);  mul_14 = arg38_1 = None
        add_17: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_15, arg39_1);  mul_15 = arg39_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:395 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_32: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_17, [8192, 768])
        permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_12: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg41_1, view_32, permute_20);  arg41_1 = view_32 = permute_20 = None
        view_33: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [16, 512, 768]);  addmm_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_34: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_33, [16, 512, 12, 64]);  view_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_21: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:408 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_35: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_17, [8192, 768])
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_13: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg43_1, view_35, permute_22);  arg43_1 = view_35 = permute_22 = None
        view_36: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [16, 512, 768]);  addmm_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_37: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_36, [16, 512, 12, 64]);  view_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_23: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:409 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_38: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_17, [8192, 768])
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_14: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg45_1, view_38, permute_24);  arg45_1 = view_38 = permute_24 = None
        view_39: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [16, 512, 768]);  addmm_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_40: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_39, [16, 512, 12, 64]);  view_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_25: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:440 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_4: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_21, permute_23, permute_25, expand_4, False);  permute_21 = permute_23 = permute_25 = expand_4 = None
        getitem_18: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:449 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_26: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:450 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_41: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(permute_26, [16, 512, 768]);  permute_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_41, [8192, 768]);  view_41 = None
        permute_27: "f32[768, 768]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        
        # No stacktrace found for following nodes
        mm_default_30: "f32[8192, 768]" = torch.ops.aten.mm.default(view_42, permute_27);  view_42 = permute_27 = None
        add_tensor_30: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_30, arg47_1);  mm_default_30 = arg47_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_43: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_30, [16, 512, 768]);  add_tensor_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:468 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_18: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_43, add_17);  view_43 = add_17 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_22: "f32[16, 512, 1]" = var_mean_5[0]
        getitem_23: "f32[16, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_6: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_18, getitem_23);  add_18 = getitem_23 = None
        add_19: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_16: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = rsqrt_5 = None
        mul_17: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg48_1);  mul_16 = arg48_1 = None
        add_20: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_17, arg49_1);  mul_17 = arg49_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_44: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_20, [8192, 768])
        permute_28: "f32[768, 3072]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        
        # No stacktrace found for following nodes
        mm_default_29: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_44, permute_28);  view_44 = permute_28 = None
        add_tensor_29: "f32[8192, 3072]" = torch.ops.aten.add.Tensor(mm_default_29, arg51_1);  mm_default_29 = arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_45: "f32[16, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_29, [16, 512, 3072]);  add_tensor_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_18: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_19: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476);  view_45 = None
        erf_2: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_21: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_21);  mul_18 = add_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_46: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_20, [8192, 3072]);  mul_20 = None
        permute_29: "f32[3072, 768]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        
        # No stacktrace found for following nodes
        mm_default_28: "f32[8192, 768]" = torch.ops.aten.mm.default(view_46, permute_29);  view_46 = permute_29 = None
        add_tensor_28: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_28, arg53_1);  mm_default_28 = arg53_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_47: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_28, [16, 512, 768]);  add_tensor_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:554 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_22: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_47, add_20);  view_47 = add_20 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_24: "f32[16, 512, 1]" = var_mean_6[0]
        getitem_25: "f32[16, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_7: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_25);  add_22 = getitem_25 = None
        add_23: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_6: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_21: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = rsqrt_6 = None
        mul_22: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_21, arg54_1);  mul_21 = arg54_1 = None
        add_24: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_22, arg55_1);  mul_22 = arg55_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:395 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_48: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_24, [8192, 768])
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_18: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg57_1, view_48, permute_30);  arg57_1 = view_48 = permute_30 = None
        view_49: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [16, 512, 768]);  addmm_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_50: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_49, [16, 512, 12, 64]);  view_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_31: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:408 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_51: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_24, [8192, 768])
        permute_32: "f32[768, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_19: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg59_1, view_51, permute_32);  arg59_1 = view_51 = permute_32 = None
        view_52: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [16, 512, 768]);  addmm_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_53: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_52, [16, 512, 12, 64]);  view_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_33: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:409 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_54: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_24, [8192, 768])
        permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_20: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg61_1, view_54, permute_34);  arg61_1 = view_54 = permute_34 = None
        view_55: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [16, 512, 768]);  addmm_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_56: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_55, [16, 512, 12, 64]);  view_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_35: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:440 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_5: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_31, permute_33, permute_35, expand_5, False);  permute_31 = permute_33 = permute_35 = expand_5 = None
        getitem_26: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:449 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_36: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3]);  getitem_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:450 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_57: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(permute_36, [16, 512, 768]);  permute_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_58: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_57, [8192, 768]);  view_57 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        
        # No stacktrace found for following nodes
        mm_default_27: "f32[8192, 768]" = torch.ops.aten.mm.default(view_58, permute_37);  view_58 = permute_37 = None
        add_tensor_27: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_27, arg63_1);  mm_default_27 = arg63_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_59: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_27, [16, 512, 768]);  add_tensor_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:468 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_25: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_59, add_24);  view_59 = add_24 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_30: "f32[16, 512, 1]" = var_mean_7[0]
        getitem_31: "f32[16, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_8: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_31);  add_25 = getitem_31 = None
        add_26: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_23: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = rsqrt_7 = None
        mul_24: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_23, arg64_1);  mul_23 = arg64_1 = None
        add_27: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_24, arg65_1);  mul_24 = arg65_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_27, [8192, 768])
        permute_38: "f32[768, 3072]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        
        # No stacktrace found for following nodes
        mm_default_26: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_60, permute_38);  view_60 = permute_38 = None
        add_tensor_26: "f32[8192, 3072]" = torch.ops.aten.add.Tensor(mm_default_26, arg67_1);  mm_default_26 = arg67_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_61: "f32[16, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_26, [16, 512, 3072]);  add_tensor_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_25: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_61, 0.5)
        mul_26: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476);  view_61 = None
        erf_3: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_28: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_28);  mul_25 = add_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_27, [8192, 3072]);  mul_27 = None
        permute_39: "f32[3072, 768]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        
        # No stacktrace found for following nodes
        mm_default_25: "f32[8192, 768]" = torch.ops.aten.mm.default(view_62, permute_39);  view_62 = permute_39 = None
        add_tensor_25: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_25, arg69_1);  mm_default_25 = arg69_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_63: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_25, [16, 512, 768]);  add_tensor_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:554 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_29: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_63, add_27);  view_63 = add_27 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_32: "f32[16, 512, 1]" = var_mean_8[0]
        getitem_33: "f32[16, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_9: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_29, getitem_33);  add_29 = getitem_33 = None
        add_30: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_8: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_28: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = rsqrt_8 = None
        mul_29: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_28, arg70_1);  mul_28 = arg70_1 = None
        add_31: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_29, arg71_1);  mul_29 = arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:395 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_64: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_31, [8192, 768])
        permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_24: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg73_1, view_64, permute_40);  arg73_1 = view_64 = permute_40 = None
        view_65: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [16, 512, 768]);  addmm_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_66: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_65, [16, 512, 12, 64]);  view_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_41: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:408 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_67: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_31, [8192, 768])
        permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_25: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg75_1, view_67, permute_42);  arg75_1 = view_67 = permute_42 = None
        view_68: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_25, [16, 512, 768]);  addmm_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_69: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_68, [16, 512, 12, 64]);  view_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_43: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:409 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_70: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_31, [8192, 768])
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_26: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg77_1, view_70, permute_44);  arg77_1 = view_70 = permute_44 = None
        view_71: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [16, 512, 768]);  addmm_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_72: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_71, [16, 512, 12, 64]);  view_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_45: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:440 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_6: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_41, permute_43, permute_45, expand_6, False);  permute_41 = permute_43 = permute_45 = expand_6 = None
        getitem_34: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:449 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_46: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:450 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_73: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(permute_46, [16, 512, 768]);  permute_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_74: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_73, [8192, 768]);  view_73 = None
        permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        
        # No stacktrace found for following nodes
        mm_default_24: "f32[8192, 768]" = torch.ops.aten.mm.default(view_74, permute_47);  view_74 = permute_47 = None
        add_tensor_24: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_24, arg79_1);  mm_default_24 = arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_75: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [16, 512, 768]);  add_tensor_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:468 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_32: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_75, add_31);  view_75 = add_31 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_38: "f32[16, 512, 1]" = var_mean_9[0]
        getitem_39: "f32[16, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_10: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_39);  add_32 = getitem_39 = None
        add_33: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_30: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = rsqrt_9 = None
        mul_31: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg80_1);  mul_30 = arg80_1 = None
        add_34: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_31, arg81_1);  mul_31 = arg81_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_76: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_34, [8192, 768])
        permute_48: "f32[768, 3072]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_76, permute_48);  view_76 = permute_48 = None
        add_tensor_23: "f32[8192, 3072]" = torch.ops.aten.add.Tensor(mm_default_23, arg83_1);  mm_default_23 = arg83_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_77: "f32[16, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_23, [16, 512, 3072]);  add_tensor_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_32: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.5)
        mul_33: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.7071067811865476);  view_77 = None
        erf_4: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_35: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_35);  mul_32 = add_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_78: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_34, [8192, 3072]);  mul_34 = None
        permute_49: "f32[3072, 768]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[8192, 768]" = torch.ops.aten.mm.default(view_78, permute_49);  view_78 = permute_49 = None
        add_tensor_22: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_22, arg85_1);  mm_default_22 = arg85_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_79: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [16, 512, 768]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:554 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_36: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_79, add_34);  view_79 = add_34 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_40: "f32[16, 512, 1]" = var_mean_10[0]
        getitem_41: "f32[16, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_11: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_41);  add_36 = getitem_41 = None
        add_37: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_10: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_35: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = rsqrt_10 = None
        mul_36: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_35, arg86_1);  mul_35 = arg86_1 = None
        add_38: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_36, arg87_1);  mul_36 = arg87_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:395 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_80: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_38, [8192, 768])
        permute_50: "f32[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_30: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg89_1, view_80, permute_50);  arg89_1 = view_80 = permute_50 = None
        view_81: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_30, [16, 512, 768]);  addmm_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_82: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_81, [16, 512, 12, 64]);  view_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_51: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:408 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_83: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_38, [8192, 768])
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_31: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg91_1, view_83, permute_52);  arg91_1 = view_83 = permute_52 = None
        view_84: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_31, [16, 512, 768]);  addmm_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_85: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_84, [16, 512, 12, 64]);  view_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_53: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:409 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_86: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_38, [8192, 768])
        permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_32: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg93_1, view_86, permute_54);  arg93_1 = view_86 = permute_54 = None
        view_87: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_32, [16, 512, 768]);  addmm_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_88: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_87, [16, 512, 12, 64]);  view_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_55: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:440 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_7: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_51, permute_53, permute_55, expand_7, False);  permute_51 = permute_53 = permute_55 = expand_7 = None
        getitem_42: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:449 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_56: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:450 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_89: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(permute_56, [16, 512, 768]);  permute_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_90: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_89, [8192, 768]);  view_89 = None
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[8192, 768]" = torch.ops.aten.mm.default(view_90, permute_57);  view_90 = permute_57 = None
        add_tensor_21: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_21, arg95_1);  mm_default_21 = arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_91: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [16, 512, 768]);  add_tensor_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:468 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_39: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_91, add_38);  view_91 = add_38 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_46: "f32[16, 512, 1]" = var_mean_11[0]
        getitem_47: "f32[16, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_12: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_47);  add_39 = getitem_47 = None
        add_40: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_37: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = rsqrt_11 = None
        mul_38: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_37, arg96_1);  mul_37 = arg96_1 = None
        add_41: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_38, arg97_1);  mul_38 = arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_92: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_41, [8192, 768])
        permute_58: "f32[768, 3072]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_92, permute_58);  view_92 = permute_58 = None
        add_tensor_20: "f32[8192, 3072]" = torch.ops.aten.add.Tensor(mm_default_20, arg99_1);  mm_default_20 = arg99_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_93: "f32[16, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_20, [16, 512, 3072]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_39: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.5)
        mul_40: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.7071067811865476);  view_93 = None
        erf_5: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_42: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_42);  mul_39 = add_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_94: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_41, [8192, 3072]);  mul_41 = None
        permute_59: "f32[3072, 768]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[8192, 768]" = torch.ops.aten.mm.default(view_94, permute_59);  view_94 = permute_59 = None
        add_tensor_19: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_19, arg101_1);  mm_default_19 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_95: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_19, [16, 512, 768]);  add_tensor_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:554 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_43: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_95, add_41);  view_95 = add_41 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_48: "f32[16, 512, 1]" = var_mean_12[0]
        getitem_49: "f32[16, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_13: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_49);  add_43 = getitem_49 = None
        add_44: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_12: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_42: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = rsqrt_12 = None
        mul_43: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg102_1);  mul_42 = arg102_1 = None
        add_45: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_43, arg103_1);  mul_43 = arg103_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:395 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_96: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_45, [8192, 768])
        permute_60: "f32[768, 768]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_36: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg105_1, view_96, permute_60);  arg105_1 = view_96 = permute_60 = None
        view_97: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_36, [16, 512, 768]);  addmm_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_98: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_97, [16, 512, 12, 64]);  view_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_61: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:408 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_99: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_45, [8192, 768])
        permute_62: "f32[768, 768]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_37: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg107_1, view_99, permute_62);  arg107_1 = view_99 = permute_62 = None
        view_100: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_37, [16, 512, 768]);  addmm_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_101: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_100, [16, 512, 12, 64]);  view_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_63: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:409 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_102: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_45, [8192, 768])
        permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_38: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg109_1, view_102, permute_64);  arg109_1 = view_102 = permute_64 = None
        view_103: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_38, [16, 512, 768]);  addmm_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_104: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_103, [16, 512, 12, 64]);  view_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_65: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:440 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_8: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_61, permute_63, permute_65, expand_8, False);  permute_61 = permute_63 = permute_65 = expand_8 = None
        getitem_50: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:449 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_66: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:450 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_105: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(permute_66, [16, 512, 768]);  permute_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_105, [8192, 768]);  view_105 = None
        permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[8192, 768]" = torch.ops.aten.mm.default(view_106, permute_67);  view_106 = permute_67 = None
        add_tensor_18: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_18, arg111_1);  mm_default_18 = arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_107: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [16, 512, 768]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:468 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_46: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_107, add_45);  view_107 = add_45 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_54: "f32[16, 512, 1]" = var_mean_13[0]
        getitem_55: "f32[16, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_14: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_55);  add_46 = getitem_55 = None
        add_47: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_13: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_44: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_13);  sub_14 = rsqrt_13 = None
        mul_45: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg112_1);  mul_44 = arg112_1 = None
        add_48: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_45, arg113_1);  mul_45 = arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_48, [8192, 768])
        permute_68: "f32[768, 3072]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_108, permute_68);  view_108 = permute_68 = None
        add_tensor_17: "f32[8192, 3072]" = torch.ops.aten.add.Tensor(mm_default_17, arg115_1);  mm_default_17 = arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_109: "f32[16, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_17, [16, 512, 3072]);  add_tensor_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_46: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_47: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_6: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_49: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_110: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_48, [8192, 3072]);  mul_48 = None
        permute_69: "f32[3072, 768]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[8192, 768]" = torch.ops.aten.mm.default(view_110, permute_69);  view_110 = permute_69 = None
        add_tensor_16: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_16, arg117_1);  mm_default_16 = arg117_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_111: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [16, 512, 768]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:554 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_50: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_111, add_48);  view_111 = add_48 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_56: "f32[16, 512, 1]" = var_mean_14[0]
        getitem_57: "f32[16, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_15: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_50, getitem_57);  add_50 = getitem_57 = None
        add_51: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_14: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_49: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_14);  sub_15 = rsqrt_14 = None
        mul_50: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg118_1);  mul_49 = arg118_1 = None
        add_52: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, arg119_1);  mul_50 = arg119_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:395 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_112: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_52, [8192, 768])
        permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_42: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg121_1, view_112, permute_70);  arg121_1 = view_112 = permute_70 = None
        view_113: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_42, [16, 512, 768]);  addmm_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_114: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_113, [16, 512, 12, 64]);  view_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_71: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:408 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_115: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_52, [8192, 768])
        permute_72: "f32[768, 768]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_43: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg123_1, view_115, permute_72);  arg123_1 = view_115 = permute_72 = None
        view_116: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_43, [16, 512, 768]);  addmm_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_117: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_116, [16, 512, 12, 64]);  view_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_73: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:409 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_118: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_52, [8192, 768])
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_44: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg125_1, view_118, permute_74);  arg125_1 = view_118 = permute_74 = None
        view_119: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_44, [16, 512, 768]);  addmm_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_120: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_119, [16, 512, 12, 64]);  view_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_75: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:440 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_9: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_71, permute_73, permute_75, expand_9, False);  permute_71 = permute_73 = permute_75 = expand_9 = None
        getitem_58: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:449 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_76: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:450 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_121: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(permute_76, [16, 512, 768]);  permute_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_122: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_121, [8192, 768]);  view_121 = None
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[8192, 768]" = torch.ops.aten.mm.default(view_122, permute_77);  view_122 = permute_77 = None
        add_tensor_15: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_15, arg127_1);  mm_default_15 = arg127_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_123: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [16, 512, 768]);  add_tensor_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:468 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_53: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_123, add_52);  view_123 = add_52 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_62: "f32[16, 512, 1]" = var_mean_15[0]
        getitem_63: "f32[16, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_16: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_53, getitem_63);  add_53 = getitem_63 = None
        add_54: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_15: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_51: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_15);  sub_16 = rsqrt_15 = None
        mul_52: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_51, arg128_1);  mul_51 = arg128_1 = None
        add_55: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_52, arg129_1);  mul_52 = arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_124: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_55, [8192, 768])
        permute_78: "f32[768, 3072]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_124, permute_78);  view_124 = permute_78 = None
        add_tensor_14: "f32[8192, 3072]" = torch.ops.aten.add.Tensor(mm_default_14, arg131_1);  mm_default_14 = arg131_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_125: "f32[16, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_14, [16, 512, 3072]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_53: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.5)
        mul_54: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.7071067811865476);  view_125 = None
        erf_7: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_56: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_56);  mul_53 = add_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_55, [8192, 3072]);  mul_55 = None
        permute_79: "f32[3072, 768]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[8192, 768]" = torch.ops.aten.mm.default(view_126, permute_79);  view_126 = permute_79 = None
        add_tensor_13: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_13, arg133_1);  mm_default_13 = arg133_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_127: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_13, [16, 512, 768]);  add_tensor_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:554 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_57: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_127, add_55);  view_127 = add_55 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_64: "f32[16, 512, 1]" = var_mean_16[0]
        getitem_65: "f32[16, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_17: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_65);  add_57 = getitem_65 = None
        add_58: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_16: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_56: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_16);  sub_17 = rsqrt_16 = None
        mul_57: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg134_1);  mul_56 = arg134_1 = None
        add_59: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_57, arg135_1);  mul_57 = arg135_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:395 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_128: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_59, [8192, 768])
        permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_48: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg137_1, view_128, permute_80);  arg137_1 = view_128 = permute_80 = None
        view_129: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_48, [16, 512, 768]);  addmm_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_130: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_129, [16, 512, 12, 64]);  view_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_81: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:408 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_131: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_59, [8192, 768])
        permute_82: "f32[768, 768]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_49: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg139_1, view_131, permute_82);  arg139_1 = view_131 = permute_82 = None
        view_132: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_49, [16, 512, 768]);  addmm_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_133: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_132, [16, 512, 12, 64]);  view_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_83: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:409 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_134: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_59, [8192, 768])
        permute_84: "f32[768, 768]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_50: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg141_1, view_134, permute_84);  arg141_1 = view_134 = permute_84 = None
        view_135: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_50, [16, 512, 768]);  addmm_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_136: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_135, [16, 512, 12, 64]);  view_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_85: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:440 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_10: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_81, permute_83, permute_85, expand_10, False);  permute_81 = permute_83 = permute_85 = expand_10 = None
        getitem_66: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:449 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_86: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3]);  getitem_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:450 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_137: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(permute_86, [16, 512, 768]);  permute_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_138: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_137, [8192, 768]);  view_137 = None
        permute_87: "f32[768, 768]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[8192, 768]" = torch.ops.aten.mm.default(view_138, permute_87);  view_138 = permute_87 = None
        add_tensor_12: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_12, arg143_1);  mm_default_12 = arg143_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_139: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [16, 512, 768]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:468 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_60: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_139, add_59);  view_139 = add_59 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_70: "f32[16, 512, 1]" = var_mean_17[0]
        getitem_71: "f32[16, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_18: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_71);  add_60 = getitem_71 = None
        add_61: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_17: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_58: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_17);  sub_18 = rsqrt_17 = None
        mul_59: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg144_1);  mul_58 = arg144_1 = None
        add_62: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_59, arg145_1);  mul_59 = arg145_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_140: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_62, [8192, 768])
        permute_88: "f32[768, 3072]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_140, permute_88);  view_140 = permute_88 = None
        add_tensor_11: "f32[8192, 3072]" = torch.ops.aten.add.Tensor(mm_default_11, arg147_1);  mm_default_11 = arg147_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_141: "f32[16, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_11, [16, 512, 3072]);  add_tensor_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_60: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
        mul_61: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
        erf_8: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_63: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_142: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_62, [8192, 3072]);  mul_62 = None
        permute_89: "f32[3072, 768]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[8192, 768]" = torch.ops.aten.mm.default(view_142, permute_89);  view_142 = permute_89 = None
        add_tensor_10: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_10, arg149_1);  mm_default_10 = arg149_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_143: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [16, 512, 768]);  add_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:554 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_64: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_143, add_62);  view_143 = add_62 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_72: "f32[16, 512, 1]" = var_mean_18[0]
        getitem_73: "f32[16, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_19: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_73);  add_64 = getitem_73 = None
        add_65: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_18: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_63: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_18);  sub_19 = rsqrt_18 = None
        mul_64: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_63, arg150_1);  mul_63 = arg150_1 = None
        add_66: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_64, arg151_1);  mul_64 = arg151_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:395 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_144: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_66, [8192, 768])
        permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_54: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg153_1, view_144, permute_90);  arg153_1 = view_144 = permute_90 = None
        view_145: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_54, [16, 512, 768]);  addmm_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_146: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_145, [16, 512, 12, 64]);  view_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_91: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:408 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_147: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_66, [8192, 768])
        permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_55: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg155_1, view_147, permute_92);  arg155_1 = view_147 = permute_92 = None
        view_148: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_55, [16, 512, 768]);  addmm_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_149: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_148, [16, 512, 12, 64]);  view_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_93: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:409 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_150: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_66, [8192, 768])
        permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_56: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg157_1, view_150, permute_94);  arg157_1 = view_150 = permute_94 = None
        view_151: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_56, [16, 512, 768]);  addmm_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_152: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_151, [16, 512, 12, 64]);  view_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_95: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:440 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_11: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_91, permute_93, permute_95, expand_11, False);  permute_91 = permute_93 = permute_95 = expand_11 = None
        getitem_74: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:449 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_96: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:450 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_153: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(permute_96, [16, 512, 768]);  permute_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_154: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_153, [8192, 768]);  view_153 = None
        permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[8192, 768]" = torch.ops.aten.mm.default(view_154, permute_97);  view_154 = permute_97 = None
        add_tensor_9: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_9, arg159_1);  mm_default_9 = arg159_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_155: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [16, 512, 768]);  add_tensor_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:468 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_67: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_155, add_66);  view_155 = add_66 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_78: "f32[16, 512, 1]" = var_mean_19[0]
        getitem_79: "f32[16, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_20: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_79);  add_67 = getitem_79 = None
        add_68: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_19: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_65: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = rsqrt_19 = None
        mul_66: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg160_1);  mul_65 = arg160_1 = None
        add_69: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_66, arg161_1);  mul_66 = arg161_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_156: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_69, [8192, 768])
        permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_156, permute_98);  view_156 = permute_98 = None
        add_tensor_8: "f32[8192, 3072]" = torch.ops.aten.add.Tensor(mm_default_8, arg163_1);  mm_default_8 = arg163_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_157: "f32[16, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_8, [16, 512, 3072]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_67: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_157, 0.5)
        mul_68: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_157, 0.7071067811865476);  view_157 = None
        erf_9: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_70: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_70);  mul_67 = add_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_158: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_69, [8192, 3072]);  mul_69 = None
        permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[8192, 768]" = torch.ops.aten.mm.default(view_158, permute_99);  view_158 = permute_99 = None
        add_tensor_7: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_7, arg165_1);  mm_default_7 = arg165_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_159: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [16, 512, 768]);  add_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:554 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_71: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_159, add_69);  view_159 = add_69 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_80: "f32[16, 512, 1]" = var_mean_20[0]
        getitem_81: "f32[16, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_21: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_81);  add_71 = getitem_81 = None
        add_72: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_20: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_70: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_20);  sub_21 = rsqrt_20 = None
        mul_71: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg166_1);  mul_70 = arg166_1 = None
        add_73: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_71, arg167_1);  mul_71 = arg167_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:395 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_160: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_73, [8192, 768])
        permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_60: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg169_1, view_160, permute_100);  arg169_1 = view_160 = permute_100 = None
        view_161: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_60, [16, 512, 768]);  addmm_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_162: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_161, [16, 512, 12, 64]);  view_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_101: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:408 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_163: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_73, [8192, 768])
        permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_61: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg171_1, view_163, permute_102);  arg171_1 = view_163 = permute_102 = None
        view_164: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_61, [16, 512, 768]);  addmm_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_165: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_164, [16, 512, 12, 64]);  view_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_103: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:409 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_166: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_73, [8192, 768])
        permute_104: "f32[768, 768]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_62: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg173_1, view_166, permute_104);  arg173_1 = view_166 = permute_104 = None
        view_167: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_62, [16, 512, 768]);  addmm_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_168: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_167, [16, 512, 12, 64]);  view_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_105: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:440 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_12: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_101, permute_103, permute_105, expand_12, False);  permute_101 = permute_103 = permute_105 = expand_12 = None
        getitem_82: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:449 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_106: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:450 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_169: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(permute_106, [16, 512, 768]);  permute_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_169, [8192, 768]);  view_169 = None
        permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[8192, 768]" = torch.ops.aten.mm.default(view_170, permute_107);  view_170 = permute_107 = None
        add_tensor_6: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_6, arg175_1);  mm_default_6 = arg175_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_171: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [16, 512, 768]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:468 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_74: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_171, add_73);  view_171 = add_73 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_86: "f32[16, 512, 1]" = var_mean_21[0]
        getitem_87: "f32[16, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_22: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_74, getitem_87);  add_74 = getitem_87 = None
        add_75: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_21: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_72: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_21);  sub_22 = rsqrt_21 = None
        mul_73: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg176_1);  mul_72 = arg176_1 = None
        add_76: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_73, arg177_1);  mul_73 = arg177_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_76, [8192, 768])
        permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_172, permute_108);  view_172 = permute_108 = None
        add_tensor_5: "f32[8192, 3072]" = torch.ops.aten.add.Tensor(mm_default_5, arg179_1);  mm_default_5 = arg179_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_173: "f32[16, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_5, [16, 512, 3072]);  add_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_74: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_75: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_10: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_77: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_74, add_77);  mul_74 = add_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_76, [8192, 3072]);  mul_76 = None
        permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[8192, 768]" = torch.ops.aten.mm.default(view_174, permute_109);  view_174 = permute_109 = None
        add_tensor_4: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_4, arg181_1);  mm_default_4 = arg181_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_175: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [16, 512, 768]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:554 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_78: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_175, add_76);  view_175 = add_76 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
        getitem_88: "f32[16, 512, 1]" = var_mean_22[0]
        getitem_89: "f32[16, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_23: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_78, getitem_89);  add_78 = getitem_89 = None
        add_79: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_22: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_77: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = rsqrt_22 = None
        mul_78: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_77, arg182_1);  mul_77 = arg182_1 = None
        add_80: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_78, arg183_1);  mul_78 = arg183_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:395 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_176: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_80, [8192, 768])
        permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_66: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg185_1, view_176, permute_110);  arg185_1 = view_176 = permute_110 = None
        view_177: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_66, [16, 512, 768]);  addmm_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_178: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_177, [16, 512, 12, 64]);  view_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_111: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:408 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_179: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_80, [8192, 768])
        permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_67: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg187_1, view_179, permute_112);  arg187_1 = view_179 = permute_112 = None
        view_180: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_67, [16, 512, 768]);  addmm_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_181: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_180, [16, 512, 12, 64]);  view_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_113: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:409 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_182: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_80, [8192, 768])
        permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_68: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg189_1, view_182, permute_114);  arg189_1 = view_182 = permute_114 = None
        view_183: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(addmm_68, [16, 512, 768]);  addmm_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:252 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_184: "f32[16, 512, 12, 64]" = torch.ops.aten.reshape.default(view_183, [16, 512, 12, 64]);  view_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:253 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_115: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:440 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_13: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512]);  where = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_111, permute_113, permute_115, expand_13, False);  permute_111 = permute_113 = permute_115 = expand_13 = None
        getitem_90: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:449 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_116: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:450 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_185: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(permute_116, [16, 512, 768]);  permute_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_186: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_185, [8192, 768]);  view_185 = None
        permute_117: "f32[768, 768]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[8192, 768]" = torch.ops.aten.mm.default(view_186, permute_117);  view_186 = permute_117 = None
        add_tensor_3: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_3, arg191_1);  mm_default_3 = arg191_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466 in forward, code: hidden_states = self.dense(hidden_states)
        view_187: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [16, 512, 768]);  add_tensor_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:468 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_81: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_187, add_80);  view_187 = add_80 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_94: "f32[16, 512, 1]" = var_mean_23[0]
        getitem_95: "f32[16, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_24: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_95);  add_81 = getitem_95 = None
        add_82: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_23: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_79: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = rsqrt_23 = None
        mul_80: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_79, arg192_1);  mul_79 = arg192_1 = None
        add_83: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_80, arg193_1);  mul_80 = arg193_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_188: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_83, [8192, 768])
        permute_118: "f32[768, 3072]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_188, permute_118);  view_188 = permute_118 = None
        add_tensor_2: "f32[8192, 3072]" = torch.ops.aten.add.Tensor(mm_default_2, arg195_1);  mm_default_2 = arg195_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:539 in forward, code: hidden_states = self.dense(hidden_states)
        view_189: "f32[16, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_2, [16, 512, 3072]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_81: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_82: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_11: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_84: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_81, add_84);  mul_81 = add_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_190: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_83, [8192, 3072]);  mul_83 = None
        permute_119: "f32[3072, 768]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[8192, 768]" = torch.ops.aten.mm.default(view_190, permute_119);  view_190 = permute_119 = None
        add_tensor_1: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mm_default_1, arg197_1);  mm_default_1 = arg197_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:552 in forward, code: hidden_states = self.dense(hidden_states)
        view_191: "f32[16, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [16, 512, 768]);  add_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:554 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_85: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_191, add_83);  view_191 = add_83 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_96: "f32[16, 512, 1]" = var_mean_24[0]
        getitem_97: "f32[16, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_25: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_97);  add_85 = getitem_97 = None
        add_86: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_24: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_84: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = rsqrt_24 = None
        mul_85: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, arg198_1);  mul_84 = arg198_1 = None
        add_87: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_85, arg199_1);  mul_85 = arg199_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1992 in forward, code: logits = self.qa_outputs(sequence_output)
        view_192: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_87, [8192, 768]);  add_87 = None
        permute_120: "f32[768, 2]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[8192, 2]" = torch.ops.aten.mm.default(view_192, permute_120);  view_192 = permute_120 = None
        add_tensor: "f32[8192, 2]" = torch.ops.aten.add.Tensor(mm_default, arg201_1);  mm_default = arg201_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1992 in forward, code: logits = self.qa_outputs(sequence_output)
        view_193: "f32[16, 512, 2]" = torch.ops.aten.reshape.default(add_tensor, [16, 512, 2]);  add_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1993 in forward, code: start_logits, end_logits = logits.split(1, dim=-1)
        split = torch.ops.aten.split.Tensor(view_193, 1, -1);  view_193 = None
        getitem_98: "f32[16, 512, 1]" = split[0]
        getitem_99: "f32[16, 512, 1]" = split[1];  split = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:2006 in forward, code: start_positions = start_positions.clamp(0, ignored_index)
        clamp_min: "i64[16]" = torch.ops.aten.clamp_min.default(arg202_1, 0);  arg202_1 = None
        clamp_max: "i64[16]" = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:2010 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        ne_1: "b8[16]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1994 in forward, code: start_logits = start_logits.squeeze(-1).contiguous()
        squeeze: "f32[16, 512]" = torch.ops.aten.squeeze.dim(getitem_98, -1);  getitem_98 = None
        clone_25: "f32[16, 512]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:2010 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        amax: "f32[16, 1]" = torch.ops.aten.amax.default(clone_25, [1], True)
        sub_26: "f32[16, 512]" = torch.ops.aten.sub.Tensor(clone_25, amax);  amax = None
        exp: "f32[16, 512]" = torch.ops.aten.exp.default(sub_26)
        sum_1: "f32[16, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[16, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_27: "f32[16, 512]" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
        ne: "b8[16]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[16]" = torch.ops.aten.where.self(ne, clamp_max, full_default);  ne = full_default = None
        unsqueeze_2: "i64[16, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather: "f32[16, 1]" = torch.ops.aten.gather.default(sub_27, 1, unsqueeze_2);  sub_27 = unsqueeze_2 = None
        squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[16]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[16]" = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        ne_2: "b8[16]" = torch.ops.aten.ne.Scalar(clamp_max, 512);  clamp_max = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_1);  sum_3 = convert_element_type_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:2007 in forward, code: end_positions = end_positions.clamp(0, ignored_index)
        clamp_min_1: "i64[16]" = torch.ops.aten.clamp_min.default(arg203_1, 0);  arg203_1 = None
        clamp_max_1: "i64[16]" = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:2011 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        ne_4: "b8[16]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1995 in forward, code: end_logits = end_logits.squeeze(-1).contiguous()
        squeeze_1: "f32[16, 512]" = torch.ops.aten.squeeze.dim(getitem_99, -1);  getitem_99 = None
        clone_26: "f32[16, 512]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:2011 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        amax_1: "f32[16, 1]" = torch.ops.aten.amax.default(clone_26, [1], True)
        sub_28: "f32[16, 512]" = torch.ops.aten.sub.Tensor(clone_26, amax_1);  amax_1 = None
        exp_1: "f32[16, 512]" = torch.ops.aten.exp.default(sub_28)
        sum_4: "f32[16, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [1], True);  exp_1 = None
        log_1: "f32[16, 1]" = torch.ops.aten.log.default(sum_4);  sum_4 = None
        sub_29: "f32[16, 512]" = torch.ops.aten.sub.Tensor(sub_28, log_1);  sub_28 = log_1 = None
        ne_3: "b8[16]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "i64[16]" = torch.ops.aten.where.self(ne_3, clamp_max_1, full_default_2);  ne_3 = full_default_2 = None
        unsqueeze_3: "i64[16, 1]" = torch.ops.aten.unsqueeze.default(where_3, 1);  where_3 = None
        gather_1: "f32[16, 1]" = torch.ops.aten.gather.default(sub_29, 1, unsqueeze_3);  sub_29 = unsqueeze_3 = None
        squeeze_3: "f32[16]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1: "f32[16]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[16]" = torch.ops.aten.where.self(ne_4, neg_1, full_default_3);  ne_4 = neg_1 = full_default_3 = None
        sum_6: "f32[]" = torch.ops.aten.sum.default(where_4);  where_4 = None
        ne_5: "b8[16]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512);  clamp_max_1 = None
        sum_5: "i64[]" = torch.ops.aten.sum.default(ne_5);  ne_5 = None
        convert_element_type_2: "f32[]" = torch.ops.prims.convert_element_type.default(sum_5, torch.float32);  sum_5 = None
        div_1: "f32[]" = torch.ops.aten.div.Tensor(sum_6, convert_element_type_2);  sum_6 = convert_element_type_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:2012 in forward, code: total_loss = (start_loss + end_loss) / 2
        add_88: "f32[]" = torch.ops.aten.add.Tensor(div, div_1);  div = div_1 = None
        div_2: "f32[]" = torch.ops.aten.div.Tensor(add_88, 2);  add_88 = None
        return (div_2, clone_25, clone_26)
        